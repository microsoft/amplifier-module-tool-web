"""Integration tests for web_search — verifies the fix against real DuckDuckGo API.

These tests hit the live DuckDuckGo search API. They prove the deadlock from
amplifier-support#70 is actually resolved, not just theoretically prevented.

Run with: pytest tests/test_search_integration.py -v -m integration
Skip with: pytest -m "not integration"
"""

import asyncio
import time

import pytest

from amplifier_module_tool_web import WebSearchTool


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_searches_no_deadlock():
    """Two concurrent searches complete without deadlock.

    This is the direct regression test for amplifier-support#70.
    Uses the exact queries from the issue's minimal reproduction steps.

    BEFORE fix: both searches hang indefinitely (primp deadlock).
    AFTER fix: both complete sequentially via the asyncio.Lock.
    """
    tool = WebSearchTool({"max_results": 3})

    start = time.monotonic()
    results = await asyncio.gather(
        tool.execute({"query": "DuckPGQ extension stability"}),
        tool.execute({"query": "pyoxigraph latest release"}),
    )
    elapsed = time.monotonic() - start

    # Hard assertion: completed without deadlock
    assert elapsed < 30, f"Searches took {elapsed:.1f}s — possible deadlock"

    for i, r in enumerate(results):
        # Soft: if search failed, it's a DuckDuckGo issue, not a deadlock
        if not r.success:
            pytest.skip(f"Search {i} failed (external API issue): {r.error}")
        assert r.output is not None, f"Search {i} returned no output"
        assert len(r.output["results"]) > 0, f"Search {i} returned empty results"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sequential_searches_return_results():
    """Sequential searches return valid results (happy path sanity check).

    The fix must not break the normal single-search use case.
    Each result must have the expected structure: title, url, snippet.
    """
    tool = WebSearchTool({"max_results": 3})

    queries = ["Python asyncio tutorial", "Rust tokio runtime"]
    for query in queries:
        result = await tool.execute({"query": query})

        assert result.success, f"Search failed for '{query}': {result.error}"
        assert result.output is not None

        search_results = result.output["results"]
        assert len(search_results) > 0, f"No results for '{query}'"

        for r in search_results:
            assert "title" in r, f"Missing 'title' key in result: {r}"
            assert "url" in r, f"Missing 'url' key in result: {r}"
            assert "snippet" in r, f"Missing 'snippet' key in result: {r}"
