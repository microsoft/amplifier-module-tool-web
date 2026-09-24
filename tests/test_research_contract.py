"""Offline search/fetch contracts; real providers are replaced at the boundary."""

import asyncio
import gzip
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from ddgs.exceptions import RatelimitException, TimeoutException

import amplifier_module_tool_web as module
from amplifier_module_tool_web import WebFetchTool, WebSearchTool


def provider(monkeypatch, rows=None, error=None):
    calls = []

    class SearchClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def text(self, query, **kwargs):
            calls.append({"query": query, **kwargs})
            if error:
                raise error
            return rows

    monkeypatch.setattr(module, "DDGS", SearchClient)
    return calls


@pytest.mark.asyncio
async def test_provider_failure_never_becomes_synthetic_success(monkeypatch):
    provider(
        monkeypatch, error=RuntimeError("private proxy credentials and a huge page")
    )
    tool = WebSearchTool({})
    tool._mock_search = AsyncMock(side_effect=AssertionError("must not fabricate"))
    result = await tool.execute({"query": "test"})
    assert not result.success
    assert result.error["code"] == "search_failed"
    assert result.error["provider"] == "ddgs"
    assert result.error["retryable"] is True
    assert "private" not in str(result)
    tool._mock_search.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,code",
    [
        (TimeoutException(), "search_timeout"),
        (RatelimitException(), "search_rate_limited"),
    ],
)
async def test_classified_provider_failures(monkeypatch, error, code):
    provider(monkeypatch, error=error)
    result = await WebSearchTool({}).execute({"query": "test"})
    assert not result.success
    assert result.error["code"] == code


@pytest.mark.asyncio
async def test_empty_result_is_an_empty_success(monkeypatch):
    provider(monkeypatch, rows=[])
    result = await WebSearchTool({}).execute({"query": "nothing"})
    assert result.success
    assert result.output["results"] == []
    assert result.output["count"] == 0
    assert result.output["mock"] is False


@pytest.mark.asyncio
async def test_default_real_search_has_bounded_attributable_results(monkeypatch):
    url = "https://source.example/article?edition=2#section"
    calls = provider(
        monkeypatch,
        rows=[
            {"title": "T" * 2000, "href": url, "body": "S" * 10000},
            {
                "title": "Extra",
                "href": "https://extra.example/",
                "body": "not requested",
            },
        ],
    )
    tool = WebSearchTool({"max_results": 1, "search_timeout": 3})
    first = await tool.execute({"query": "research"})
    second = await tool.execute({"query": "another query"})
    assert first.success
    assert first.output["count"] == 1
    assert first.output["provider"] == "ddgs"
    assert first.output["backend"] == "auto"
    assert first.output["mock"] is False
    row = first.output["results"][0]
    assert row["url"] == row["source_url"] == url
    assert row["source_id"] == second.output["results"][0]["source_id"]
    assert len(row["title"]) == 512
    assert len(row["snippet"]) == 2000
    assert row["truncated"] is True
    assert calls[:2] == [
        {"timeout": 3},
        {"query": "research", "max_results": 1, "backend": "auto"},
    ]


@pytest.mark.asyncio
async def test_duckduckgo_setting_selects_that_backend(monkeypatch):
    calls = provider(monkeypatch, rows=[])
    result = await WebSearchTool({"search_engine": "duckduckgo"}).execute(
        {"query": "test"}
    )
    assert result.success
    assert calls[1]["backend"] == result.output["backend"] == "duckduckgo"


@pytest.mark.asyncio
async def test_mock_requires_explicit_setting_and_labels_every_result(monkeypatch):
    provider(monkeypatch, error=AssertionError("must not call real backend"))
    result = await WebSearchTool({"search_engine": "mock", "max_results": 1}).execute(
        {"query": "test"}
    )
    assert result.success
    assert result.output["mock"] is True
    assert result.output["provider"] == "mock"
    assert "not web evidence" in result.output["warning"]
    assert len(result.output["results"]) == 1
    assert result.output["results"][0]["mock"] is True
    assert result.output["results"][0]["title"].startswith("[MOCK]")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [
        None,
        {},
        [{"title": "No URL"}],
        [{"href": "javascript:alert(1)"}],
        [{"href": "https://username:password@example.com/"}],
        [{"href": "https://example.com/", "body": None}],
    ],
)
async def test_malformed_provider_response_fails_instead_of_claiming_evidence(
    monkeypatch, rows
):
    provider(monkeypatch, rows=rows)
    result = await WebSearchTool({}).execute({"query": "test"})
    assert not result.success
    assert result.error["code"] == "search_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [None, "", "  ", 12, ["test"], "x" * 4097])
async def test_invalid_query_never_calls_provider(monkeypatch, query):
    calls = provider(monkeypatch, error=AssertionError("unexpected backend call"))
    result = await WebSearchTool({}).execute({"query": query})
    assert not result.success
    assert result.error["code"] == "invalid_input"
    assert calls == []


@pytest.mark.parametrize(
    "config",
    [
        {"max_results": 0},
        {"max_results": True},
        {"max_results": 51},
        {"search_timeout": 0},
        {"search_timeout": float("inf")},
        {"search_engine": "unknown"},
    ],
)
def test_invalid_search_config_is_explicit(config):
    with pytest.raises(ValueError):
        WebSearchTool(config)


@pytest.mark.asyncio
async def test_search_deadline_returns_timeout():
    tool = WebSearchTool({"search_timeout": 0.01})

    async def blocked(_):
        await asyncio.Event().wait()

    tool._real_search = blocked
    result = await tool.execute({"query": "test"})
    assert not result.success
    assert result.error["code"] == "search_timeout"


@pytest.mark.asyncio
async def test_cancellation_propagates_while_sync_search_is_running(monkeypatch):
    started = asyncio.Event()
    release = threading.Event()
    finished = asyncio.Event()
    loop = asyncio.get_running_loop()

    class BlockingSearch:
        def __init__(self, **kwargs):
            pass

        def text(self, *args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            try:
                release.wait(timeout=5)
                return [
                    {"href": "https://source.example/", "title": "late", "body": "late"}
                ]
            finally:
                loop.call_soon_threadsafe(finished.set)

    monkeypatch.setattr(module, "DDGS", BlockingSearch)
    task = asyncio.create_task(WebSearchTool({}).execute({"query": "test"}))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 2)


@pytest_asyncio.fixture
async def http_server():
    app = web.Application()
    runner = web.AppRunner(app)
    requests = []
    release = asyncio.Event()
    started = asyncio.Event()

    async def handler(request):
        requests.append(request.path)
        if request.path == "/redirect":
            raise web.HTTPFound("/final")
        if request.path == "/blocked-redirect":
            raise web.HTTPFound("https://blocked.example/private")
        if request.path == "/redirect-loop":
            raise web.HTTPFound("/redirect-loop")
        if request.path == "/missing":
            raise web.HTTPNotFound()
        if request.path == "/stream":
            response = web.StreamResponse(headers={"Content-Type": "text/plain"})
            await response.prepare(request)
            await response.write(b"0123456789")
            started.set()
            await release.wait()
            return response
        if request.path == "/compressed":
            return web.Response(
                body=gzip.compress(b"x" * 1000),
                headers={"Content-Encoding": "gzip"},
                content_type="text/plain",
            )
        if request.path == "/binary":
            return web.Response(
                body=b"%PDF-test\x00binary", content_type="application/pdf"
            )
        return web.Response(
            body=b"<title>Source</title><p>Verified text</p>", content_type="text/html"
        )

    app.router.add_get("/{path:.*}", handler)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests, started, release
    finally:
        release.set()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_fetch_attribution_uses_final_url_and_matches_search_identity(
    http_server, monkeypatch
):
    base, requests, _, _ = http_server
    tool = WebFetchTool({"blocked_domains": [], "allow_private_networks": True})
    result = await tool.execute({"url": base + "/redirect"})
    assert result.success
    assert result.output["url"] == result.output["requested_url"] == base + "/redirect"
    assert result.output["source_url"] == base + "/final"
    assert "Verified text" in result.output["content"]
    assert result.output["status_code"] == 200
    assert requests == ["/redirect", "/final"]
    provider(
        monkeypatch, rows=[{"href": base + "/final", "title": "Source", "body": ""}]
    )
    search = await WebSearchTool({}).execute({"query": "source"})
    assert search.output["results"][0]["source_id"] == result.output["source_id"]


@pytest.mark.asyncio
async def test_fetch_limit_does_not_drain_an_unfinished_response(http_server):
    base, _, _, release = http_server
    result = await asyncio.wait_for(
        WebFetchTool({"blocked_domains": [], "allow_private_networks": True}).execute(
            {"url": base + "/stream", "limit": 4}
        ),
        1,
    )
    assert result.success
    assert not release.is_set()
    assert result.output["returned_bytes"] == 4
    assert result.output["content"].startswith("0123")
    assert result.output["truncated"] is True
    assert result.output["total_bytes"] is None


@pytest.mark.asyncio
async def test_fetch_keeps_known_size_for_truncated_response(http_server):
    base, _, _, _ = http_server
    result = await WebFetchTool(
        {"blocked_domains": [], "allow_private_networks": True}
    ).execute({"url": base + "/final", "limit": 4})
    assert result.success
    assert result.output["total_bytes"] == len(
        b"<title>Source</title><p>Verified text</p>"
    )
    assert result.output["returned_bytes"] == 4


@pytest.mark.asyncio
async def test_redirect_cannot_bypass_domain_policy(http_server):
    base, requests, _, _ = http_server
    result = await WebFetchTool(
        {
            "blocked_domains": ["blocked.example"],
            "allow_private_networks": True,
        }
    ).execute({"url": base + "/blocked-redirect"})
    assert not result.success
    assert "blocked" in result.error["message"]
    assert requests == ["/blocked-redirect"]


@pytest.mark.asyncio
async def test_redirect_count_is_bounded(http_server):
    base, requests, _, _ = http_server
    result = await WebFetchTool(
        {"blocked_domains": [], "allow_private_networks": True}
    ).execute({"url": base + "/redirect-loop"})
    assert not result.success
    assert "Too many redirects" in result.error["message"]
    assert len(requests) == 11


@pytest.mark.asyncio
async def test_fetch_failure_is_not_content(http_server):
    base, _, _, _ = http_server
    result = await WebFetchTool(
        {"blocked_domains": [], "allow_private_networks": True}
    ).execute({"url": base + "/missing"})
    assert not result.success
    assert "HTTP 404" in result.error["message"]


@pytest.mark.asyncio
async def test_fetch_cancellation_propagates_and_keeps_shared_session_open(http_server):
    base, _, started, _ = http_server
    async with aiohttp.ClientSession() as session:
        tool = WebFetchTool(
            {"blocked_domains": [], "allow_private_networks": True},
            shared_session=session,
        )
        task = asyncio.create_task(tool.execute({"url": base + "/stream"}))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not session.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra", [{"offset": -1}, {"offset": True}, {"limit": 0}, {"limit": "4"}]
)
async def test_fetch_rejects_invalid_window_without_request(http_server, extra):
    base, requests, _, _ = http_server
    result = await WebFetchTool(
        {"blocked_domains": [], "allow_private_networks": True}
    ).execute({"url": base + "/final", **extra})
    assert not result.success
    assert result.error["code"] == "invalid_input"
    assert requests == []


@pytest.mark.asyncio
async def test_binary_download_preserves_bytes_and_source_metadata(
    http_server, tmp_path
):
    base, _, _, _ = http_server
    tool = WebFetchTool(
        {
            "blocked_domains": [],
            "allow_private_networks": True,
            "working_dir": tmp_path,
        }
    )
    inline = await tool.execute({"url": base + "/binary"})
    assert not inline.success
    path = tmp_path / "download.pdf"
    saved = await tool.execute({"url": base + "/binary", "save_to_file": path.name})
    assert saved.success
    assert path.read_bytes() == b"%PDF-test\x00binary"
    assert saved.output["source_url"] == base + "/binary"
    assert saved.output["saved_bytes"] == len(path.read_bytes())


@pytest.mark.asyncio
async def test_compressed_length_is_not_reported_as_decoded_size(http_server):
    base, _, _, _ = http_server
    tool = WebFetchTool({"blocked_domains": [], "allow_private_networks": True})
    partial = await tool.execute({"url": base + "/compressed", "limit": 4})
    assert partial.success
    assert partial.output["total_bytes"] is None
    assert partial.output["returned_bytes"] == 4
    complete = await tool.execute({"url": base + "/compressed"})
    assert complete.success
    assert complete.output["total_bytes"] == 1000
    assert complete.output["truncated"] is False


@pytest.mark.asyncio
async def test_exact_window_and_pagination_preserve_existing_fields(http_server):
    base, _, _, _ = http_server
    tool = WebFetchTool(
        {
            "blocked_domains": [],
            "allow_private_networks": True,
            "extract_text": False,
        }
    )
    content = b"<title>Source</title><p>Verified text</p>"
    result = await tool.execute(
        {"url": base + "/final", "offset": 7, "limit": len(content) - 7}
    )
    assert result.success
    assert result.output["content"] == content[7:].decode()
    assert result.output["offset"] == 7
    assert result.output["returned_bytes"] == len(content) - 7
    assert result.output["truncated"] is False
    assert result.output["total_bytes"] == len(content)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/binary", "/compressed", "/stream"])
async def test_download_cap_preserves_existing_file_for_known_compressed_and_chunked_bodies(
    http_server, tmp_path, route
):
    base, _, _, release = http_server
    path = tmp_path / "original.pdf"
    path.write_bytes(b"original evidence")
    tool = WebFetchTool(
        {
            "blocked_domains": [],
            "allow_private_networks": True,
            "max_download_bytes": 100,
            "working_dir": tmp_path,
        }
    )
    result = await asyncio.wait_for(
        tool.execute(
            {"url": base + route, "save_to_file": path.name, "download_limit": 4}
        ),
        1,
    )
    assert not result.success and result.error["code"] == "download_too_large"
    assert result.error["max_download_bytes"] == 4
    assert path.read_bytes() == b"original evidence"
    assert not list(tmp_path.glob(".amplifier-download-*"))
    if route == "/stream":
        assert not release.is_set()  # Does not drain a never-finished body.


@pytest.mark.asyncio
async def test_download_cancellation_keeps_destination_and_shared_transport(
    http_server, tmp_path
):
    base, _, started, _ = http_server
    path = tmp_path / "original.txt"
    path.write_bytes(b"original")
    async with aiohttp.ClientSession() as session:
        tool = WebFetchTool(
            {
                "blocked_domains": [],
                "allow_private_networks": True,
                "working_dir": tmp_path,
            },
            shared_session=session,
        )
        task = asyncio.create_task(
            tool.execute({"url": base + "/stream", "save_to_file": path.name})
        )
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not session.closed
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".amplifier-download-*"))


@pytest.mark.asyncio
async def test_complete_pdf_at_exact_cap_retains_bytes_hash_and_source(
    http_server, tmp_path
):
    base, _, _, _ = http_server
    body = b"%PDF-test\x00binary"
    target = tmp_path / "download.pdf"
    target.write_bytes(b"old")
    result = await WebFetchTool(
        {
            "blocked_domains": [],
            "allow_private_networks": True,
            "max_download_bytes": len(body),
            "working_dir": tmp_path,
        }
    ).execute({"url": base + "/binary", "save_to_file": target.name})
    assert result.success and target.read_bytes() == body
    assert result.output["truncated"] is False
    assert result.output["download_limit"] == result.output["total_bytes"] == len(body)
    assert (
        result.output["content_sha256"]
        == result.output["saved_sha256"]
        == module.sha256(body).hexdigest()
    )
    assert result.output["source_url"] == base + "/binary"


@pytest.mark.asyncio
async def test_download_replace_failure_cleans_temporary_and_preserves_destination(
    http_server, tmp_path, monkeypatch
):
    import os

    base, _, _, _ = http_server
    path = tmp_path / "original.pdf"
    path.write_bytes(b"original")

    def fail(*args):
        raise OSError("synthetic replacement failure")

    monkeypatch.setattr(os, "replace", fail)
    result = await WebFetchTool(
        {
            "blocked_domains": [],
            "allow_private_networks": True,
            "working_dir": tmp_path,
        }
    ).execute({"url": base + "/binary", "save_to_file": path.name})
    assert not result.success
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".amplifier-download-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, True, "10", 101])
async def test_download_limit_cannot_raise_host_cap_or_accept_invalid_values(
    http_server, limit
):
    base, requests, _, _ = http_server
    result = await WebFetchTool(
        {
            "blocked_domains": [],
            "allow_private_networks": True,
            "max_download_bytes": 100,
        }
    ).execute(
        {
            "url": base + "/binary",
            "save_to_file": "never-created.pdf",
            "download_limit": limit,
        }
    )
    assert not result.success and result.error["code"] == "invalid_input"
    assert requests == []


@pytest.mark.parametrize(
    "cap", [0, -1, True, "10", WebFetchTool.MAX_DOWNLOAD_LIMIT + 1]
)
def test_invalid_host_download_cap_is_rejected(cap):
    with pytest.raises(ValueError, match="max_download_bytes"):
        WebFetchTool({"max_download_bytes": cap})


@pytest.mark.parametrize(
    "url",
    [
        "http://contoso.com.evil.example/",
        "http://evilcontoso.com/",
        "http://contoso.com@127.0.0.2/",
    ],
)
def test_allowlist_requires_hostname_boundaries_and_rejects_userinfo(url):
    tool = WebFetchTool({"allowed_domains": ["contoso.com"]})
    assert not tool._is_valid_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://contoso.com/",
        "https://api.contoso.com/",
        "https://CONTOSO.COM./",
    ],
)
def test_allowlist_accepts_exact_host_and_subdomains(url):
    tool = WebFetchTool({"allowed_domains": ["contoso.com"]})
    assert tool._is_valid_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://2130706433/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://172.31.0.1/",
    ],
)
def test_default_policy_rejects_non_public_ip_literals(url):
    assert not WebFetchTool({})._is_valid_url(url)


@pytest.mark.asyncio
async def test_connection_resolver_rejects_any_non_public_dns_answer():
    class MixedResolver:
        async def resolve(self, host, port, family):
            return [
                {"host": "93.184.216.34"},
                {"host": "10.0.0.1"},
            ]

    resolver = module._PublicAddressResolver()
    resolver._resolver = MixedResolver()
    with pytest.raises(OSError, match="non-public"):
        await resolver.resolve("mixed.example", 443)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_path",
    [
        "../escaped.txt",
        "~/escaped.txt",
        "C:\\escaped.txt",
        "\\\\server\\share\\escaped.txt",
    ],
)
async def test_download_rejects_paths_outside_working_dir_without_request(
    http_server, tmp_path, file_path
):
    base, requests, _, _ = http_server
    result = await WebFetchTool(
        {
            "blocked_domains": [],
            "allow_private_networks": True,
            "working_dir": tmp_path,
        }
    ).execute({"url": base + "/final", "save_to_file": file_path})
    assert not result.success
    assert result.error["code"] == "invalid_input"
    assert requests == []


@pytest.mark.asyncio
async def test_download_saves_nested_relative_path_within_working_dir(
    http_server, tmp_path
):
    base, _, _, _ = http_server
    result = await WebFetchTool(
        {
            "blocked_domains": [],
            "allow_private_networks": True,
            "working_dir": tmp_path,
        }
    ).execute({"url": base + "/final", "save_to_file": "nested/page.txt"})
    destination = tmp_path / "nested" / "page.txt"
    assert result.success
    assert destination.exists()
    assert Path(result.output["saved_to"]) == destination
