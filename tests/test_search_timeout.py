"""Regression test for the web_search hang (amplifier#219).

`_real_search()` offloads a blocking DuckDuckGo call to a worker thread. It used
to do so with `run_in_executor(None, ...)` and no timeout, so a thread that
wedged -- as happened with the parallel `primp.Client` init deadlock fixed in
deedy5/primp#142 -- blocked the awaiting coroutine forever and took the session
with it. asyncio cannot cancel a running thread, so the guarantee here is
narrow but the one that matters: the *caller* is always released.
"""

import time

import pytest

import amplifier_module_tool_web
from amplifier_module_tool_web import WebSearchTool


class _WedgedDDGS:
    """Stand-in for a DDGS client whose call never returns in time."""

    def text(self, query: str, max_results: int = 5):
        # Well above the 0.1s budget the test configures, but short enough that
        # the deliberately-leaked worker thread does not slow suite teardown.
        time.sleep(2)
        return []


@pytest.mark.asyncio
async def test_search_timeout_releases_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged search must time out and fall back, not hang the caller."""
    monkeypatch.setattr(
        amplifier_module_tool_web, "DDGS", lambda *a, **kw: _WedgedDDGS()
    )
    tool = WebSearchTool({"search_timeout": 0.1, "max_results": 2})

    started = time.monotonic()
    results = await tool._real_search("anything")
    elapsed = time.monotonic() - started

    assert elapsed < 5, f"caller was not released promptly (took {elapsed:.1f}s)"
    assert results == await tool._mock_search("anything")


@pytest.mark.asyncio
async def test_search_timeout_is_configurable() -> None:
    """search_timeout comes from config, defaulting to DEFAULT_SEARCH_TIMEOUT."""
    assert (
        WebSearchTool({}).search_timeout
        == amplifier_module_tool_web.DEFAULT_SEARCH_TIMEOUT
    )
    assert WebSearchTool({"search_timeout": 5}).search_timeout == 5
