"""
Web tool module for Amplifier.
Provides web search and fetch capabilities.
"""

# Amplifier module metadata
__amplifier_module_type__ = "tool"

import asyncio
import logging
from typing import Any
from typing import Optional
from urllib.parse import urlparse

import aiohttp
from amplifier_core import ModuleCoordinator
from amplifier_core import ToolResult
from bs4 import BeautifulSoup
from ddgs import DDGS

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount web tools."""
    config = config or {}

    # Get session.working_dir capability if not explicitly configured
    # This ensures save_to_file paths are resolved against the session's working directory
    if "working_dir" not in config:
        working_dir = coordinator.get_capability("session.working_dir")
        if working_dir:
            config["working_dir"] = working_dir
            logger.debug(f"Using session.working_dir: {working_dir}")

    # Create shared session at mount time for connection reuse
    shared_session = aiohttp.ClientSession()

    tools = [
        WebSearchTool(config),
        WebFetchTool(config, shared_session=shared_session),
    ]

    for tool in tools:
        await coordinator.mount("tools", tool, name=tool.name)

    logger.info(f"Mounted {len(tools)} web tools")

    # Return cleanup function to properly close the shared session
    # Use asyncio.shield to protect close() from cancellation during Ctrl+C
    async def cleanup():
        if not shared_session.closed:
            try:
                await asyncio.shield(shared_session.close())
            except asyncio.CancelledError:
                pass  # Swallow cancellation during cleanup

    return cleanup


class WebSearchTool:
    """Simple web search tool (mock implementation)."""

    name = "web_search"
    description = "Search the web for information"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.search_engine = config.get("search_engine", "mock")
        self.api_key = config.get("api_key")
        self.max_results = config.get("max_results", 5)
        self._search_timeout = config.get("search_timeout", 30)

    @property
    def input_schema(self) -> dict:
        """Return JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query to execute"}
            },
            "required": ["query"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute web search."""
        query = input.get("query")
        if not query:
            return ToolResult(success=False, error={"message": "Query is required"})

        try:
            # Try real search first, fall back to mock if it fails
            results = await self._real_search(query)

            return ToolResult(
                success=True,
                output={"query": query, "results": results, "count": len(results)},
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error={
                    "message": (
                        f"web_search timed out after {self._search_timeout}s for query: {query!r}. "
                        "The search backend may be unresponsive. "
                        "Try again or use a different query."
                    )
                },
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return ToolResult(success=False, error={"message": str(e)})

    async def _real_search(self, query: str) -> list:
        """Perform real web search using DuckDuckGo."""
        try:
            # Use sync DDGS in async context
            def search_sync():
                ddgs = DDGS()
                results = []
                for r in ddgs.text(query, max_results=self.max_results):  # pyright: ignore[reportAttributeAccessIssue]
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", ""),
                        }
                    )
                return results

            # Run in thread pool with timeout safety net.
            # NOTE: We use asyncio.wait() instead of asyncio.wait_for() because
            # Python 3.11's wait_for waits for cancellation to complete, but
            # run_in_executor threads cannot be cancelled — causing a hang.
            # asyncio.wait() returns promptly on timeout without cancelling.
            loop = asyncio.get_event_loop()
            fut = loop.run_in_executor(None, search_sync)
            done, _ = await asyncio.wait({fut}, timeout=self._search_timeout)
            if not done:
                raise asyncio.TimeoutError()
            return fut.result()

        except asyncio.TimeoutError:
            raise  # Propagate to execute() — timeout is a serious signal, not a mock fallback
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}, falling back to mock")
            # Fallback to mock on error
            return await self._mock_search(query)

    async def _mock_search(self, query: str) -> list:
        """Mock search implementation."""
        # In production, replace with actual search API call
        return [
            {
                "title": f"Result 1 for {query}",
                "url": "https://example.com/1",
                "snippet": f"This is a mock search result for {query}...",
            },
            {
                "title": f"Result 2 for {query}",
                "url": "https://example.com/2",
                "snippet": f"Another mock result about {query}...",
            },
            {
                "title": f"Result 3 for {query}",
                "url": "https://example.com/3",
                "snippet": f"More information about {query}...",
            },
        ][: self.max_results]


class WebFetchTool:
    """Fetch and parse web pages with streaming support and truncation handling."""

    name = "web_fetch"
    description = """Fetch content from a web URL.

Content is limited to 200KB by default to avoid overwhelming responses.
For larger content:
- Use save_to_file parameter to save full content to a file
- Use offset/limit parameters to paginate through large content

Response includes:
- truncated: boolean indicating if content was cut off
- total_bytes: original content size (when available)
- Use these to decide if you need the full content via save_to_file"""

    # Default limit: 200KB is reasonable for web content
    DEFAULT_LIMIT = 200 * 1024
    CHUNK_SIZE = 8192
    PREVIEW_SIZE = 1000

    def __init__(
        self,
        config: dict[str, Any],
        shared_session: Optional[aiohttp.ClientSession] = None,
    ):
        self.config = config
        self.timeout = config.get("timeout", 10)
        self.default_limit = config.get("default_limit", self.DEFAULT_LIMIT)
        self.allowed_domains = config.get("allowed_domains", [])
        self.blocked_domains = config.get(
            "blocked_domains",
            [
                "localhost",
                "127.0.0.1",
                "0.0.0.0",
                "192.168.",
                "10.",
                "172.16.",
            ],
        )
        self.extract_text = config.get("extract_text", True)
        self._shared_session = shared_session
        # Working directory for resolving relative paths (from session.working_dir capability)
        self.working_dir = config.get("working_dir")

    @property
    def input_schema(self) -> dict:
        """Return JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch content from",
                },
                "save_to_file": {
                    "type": "string",
                    "description": "Save full content to this file path instead of returning in response. "
                    "Useful for large pages. Returns metadata + preview when set.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Start reading from byte N (default 0). Use for pagination.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max bytes to return (default 200KB). Use for pagination.",
                    "default": 204800,
                },
            },
            "required": ["url"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Fetch content from URL with streaming and truncation support."""
        url = input.get("url")
        if not url:
            return ToolResult(success=False, error={"message": "URL is required"})

        save_to_file = input.get("save_to_file")
        offset = input.get("offset", 0)
        limit = input.get("limit", self.default_limit)

        # Validate URL
        if not self._is_valid_url(url):
            return ToolResult(
                success=False, error={"message": f"Invalid or blocked URL: {url}"}
            )

        try:
            # Use shared session if available, otherwise create one for this request
            session = self._shared_session
            owns_session = False
            if session is None or session.closed:
                session = aiohttp.ClientSession()
                owns_session = True

            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={"User-Agent": "Amplifier/1.0"},
                ) as response:
                    # Check response
                    if response.status != 200:
                        return ToolResult(
                            success=False,
                            error={
                                "message": f"HTTP {response.status}: {response.reason}"
                            },
                        )

                    # Get content length hint (may not be accurate for compressed/chunked)
                    content_length_header = response.headers.get("Content-Length")
                    declared_size = (
                        int(content_length_header) if content_length_header else None
                    )

                    # Stream content with hard limit to avoid memory issues
                    if save_to_file:
                        return await self._fetch_to_file(
                            response, url, save_to_file, declared_size
                        )
                    else:
                        return await self._fetch_with_limit(
                            response, url, offset, limit, declared_size
                        )
            finally:
                # Only close if we created the session ourselves
                if owns_session and not session.closed:
                    await session.close()

        except TimeoutError:
            return ToolResult(
                success=False, error={"message": f"Timeout fetching {url}"}
            )
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return ToolResult(success=False, error={"message": str(e)})

    async def _fetch_with_limit(
        self,
        response: aiohttp.ClientResponse,
        url: str,
        offset: int,
        limit: int,
        declared_size: Optional[int],
    ) -> ToolResult:
        """Fetch content with streaming and hard byte limit."""
        chunks: list[bytes] = []
        total_read = 0
        truncated = False

        # Calculate how much we need to read: offset + limit + 1 (to detect truncation)
        max_to_read = offset + limit + 1

        # Stream with hard limit to avoid loading huge responses into memory
        async for chunk in response.content.iter_chunked(self.CHUNK_SIZE):
            chunk_len = len(chunk)
            chunk_end = total_read + chunk_len

            # Only keep bytes within our window [offset, offset + limit)
            if chunk_end > offset and total_read < offset + limit:
                # Calculate slice within this chunk
                start_in_chunk = max(0, offset - total_read)
                end_in_chunk = min(chunk_len, offset + limit - total_read)
                chunks.append(chunk[start_in_chunk:end_in_chunk])

            total_read += chunk_len

            # Stop once we've read enough to know if there's more
            if total_read >= max_to_read:
                truncated = True
                break

        # If we stopped early, try to get actual total size
        actual_total: Optional[int] = None
        if truncated:
            # Read remaining to get total size (but don't store it)
            remaining_size = 0
            async for chunk in response.content.iter_chunked(self.CHUNK_SIZE):
                remaining_size += len(chunk)
            actual_total = total_read + remaining_size
        else:
            actual_total = total_read
            # Check if content was truncated based on what we read vs limit
            if total_read > offset + limit:
                truncated = True

        # Use declared size if available and larger
        if declared_size and (actual_total is None or declared_size > actual_total):
            actual_total = declared_size

        # Combine chunks and decode
        raw_content = b"".join(chunks)
        try:
            content = raw_content.decode("utf-8", errors="replace")
        except Exception:
            content = raw_content.decode("latin-1", errors="replace")

        # Extract text if requested and HTML
        content_type = response.content_type or ""
        if self.extract_text:
            text = self._extract_text(content, content_type)
        else:
            text = content

        # Build result content with truncation indicator
        result_content = text
        if truncated:
            result_content = (
                f"{text}\n\n"
                f"[Content truncated at {limit} bytes. "
                f"Total: {actual_total or 'unknown'} bytes. "
                f"Use offset/limit to paginate or save_to_file for full content.]"
            )

        return ToolResult(
            success=True,
            output={
                "url": url,
                "content": result_content,
                "content_type": content_type,
                "truncated": truncated,
                "total_bytes": actual_total,
                "offset": offset,
                "limit": limit,
                "returned_bytes": len(raw_content),
            },
        )

    async def _fetch_to_file(
        self,
        response: aiohttp.ClientResponse,
        url: str,
        file_path: str,
        declared_size: Optional[int],
    ) -> ToolResult:
        """Fetch full content and save to file, return metadata + preview."""
        from pathlib import Path

        chunks: list[bytes] = []
        total_bytes = 0

        # Stream entire content
        async for chunk in response.content.iter_chunked(self.CHUNK_SIZE):
            chunks.append(chunk)
            total_bytes += len(chunk)

        raw_content = b"".join(chunks)

        # Decode content
        try:
            content = raw_content.decode("utf-8", errors="replace")
        except Exception:
            content = raw_content.decode("latin-1", errors="replace")

        # Extract text if HTML
        content_type = response.content_type or ""
        if self.extract_text:
            text = self._extract_text(content, content_type)
        else:
            text = content

        # Write to file
        try:
            path = Path(file_path).expanduser()
            # Resolve relative paths against working_dir (from session.working_dir capability)
            if not path.is_absolute() and self.working_dir:
                path = Path(self.working_dir) / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except Exception as e:
            return ToolResult(
                success=False, error={"message": f"Failed to write file: {e}"}
            )

        # Create preview
        preview = text[: self.PREVIEW_SIZE]
        if len(text) > self.PREVIEW_SIZE:
            preview += f"\n\n[... {len(text) - self.PREVIEW_SIZE} more characters saved to {file_path}]"

        return ToolResult(
            success=True,
            output={
                "url": url,
                "content": preview,
                "content_type": content_type,
                "truncated": False,
                "total_bytes": total_bytes,
                "saved_to": str(path),
                "saved_bytes": len(text.encode("utf-8")),
            },
        )

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL for safety."""
        try:
            parsed = urlparse(url)

            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False

            # Only allow http/https
            if parsed.scheme not in ["http", "https"]:
                return False

            # Check blocked domains
            for blocked in self.blocked_domains:
                if blocked in parsed.netloc:
                    logger.warning(f"Blocked domain: {parsed.netloc}")
                    return False

            # Check allowed domains if configured
            if self.allowed_domains:
                allowed = any(
                    domain in parsed.netloc for domain in self.allowed_domains
                )
                if not allowed:
                    logger.warning(f"Domain not in allowlist: {parsed.netloc}")
                    return False

            return True

        except Exception:
            return False

    def _extract_text(self, content: str, content_type: str) -> str:
        """Extract text from HTML content."""
        if "html" in content_type:
            try:
                soup = BeautifulSoup(content, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = "\n".join(chunk for chunk in chunks if chunk)

                return text

            except Exception as e:
                logger.warning(f"Failed to extract text: {e}")
                return content
        else:
            return content
