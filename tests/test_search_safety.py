"""Unit tests for web_search concurrency safety (lock + timeout)."""

import asyncio
import threading
import time
from unittest.mock import patch

import pytest
from ddgs.ddgs import DDGS  # Real class, not the lazy-loading _DDGSProxy

from amplifier_module_tool_web import WebSearchTool


@pytest.mark.asyncio
async def test_timeout_returns_error_result():
    """Search timeout returns ToolResult(success=False), not a hang.

    Verifies:
    - Timeout fires and completes (does NOT hang forever)
    - Returns ToolResult with success=False (not a raw exception)
    - Error message contains the query string
    - Error message mentions "timed out"
    - Does NOT fall back to mock results silently
    """
    # Event lets us unblock the mock thread after the test so pytest can exit cleanly.
    # Without this, time.sleep(999) would keep the executor thread alive.
    stall = threading.Event()

    def blocking_search(self, *args, **kwargs):
        stall.wait(timeout=999)  # Block until signaled or 999s
        return []

    # Use a 2-second timeout for fast test execution (production default is 30s)
    tool = WebSearchTool({"search_timeout": 2, "max_results": 1})

    try:
        with patch.object(DDGS, "text", blocking_search):
            start = time.monotonic()
            # Safety net: if the tool's timeout doesn't fire, this outer timeout
            # will catch it. In RED phase, this fires -> test fails with TimeoutError.
            # In GREEN phase, the tool's own 2s timeout fires first -> returns ToolResult.
            result = await asyncio.wait_for(
                tool.execute({"query": "test timeout query"}),
                timeout=10,
            )
            elapsed = time.monotonic() - start
    finally:
        stall.set()  # Unblock the mock thread so it can exit

    # Must return a failed ToolResult, not raise
    assert not result.success, "Timeout should return success=False"
    assert result.error is not None, "Timeout should include error details"
    assert "timed out" in result.error["message"].lower()
    assert "test timeout query" in result.error["message"]
    # Should complete in ~2s (the configured timeout), not 10s or 999s
    assert elapsed < 5, f"Took {elapsed:.1f}s -- timeout didn't fire promptly"
