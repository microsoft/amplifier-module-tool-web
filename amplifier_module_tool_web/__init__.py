"""
Web tool module for Amplifier.
Provides web search and fetch capabilities.
"""

# Amplifier module metadata
__amplifier_module_type__ = "tool"

import asyncio
import ipaddress
import logging
import socket
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hashlib import sha256
from math import isfinite
from pathlib import Path, PureWindowsPath
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from amplifier_core import ModuleCoordinator, ToolResult
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import RatelimitException, TimeoutException

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

    search_tool = WebSearchTool(config)

    # Resolve destinations at connection time so DNS rebinding cannot bypass
    # the default private-network restriction.
    connector = (
        None
        if config.get("allow_private_networks", False)
        else aiohttp.TCPConnector(resolver=_PublicAddressResolver())
    )
    shared_session = aiohttp.ClientSession(connector=connector)

    tools = [
        search_tool,
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


def _source_metadata(url: str) -> dict[str, str]:
    """Stable identity for an exact source URL; never rewrite attribution URLs."""
    return {
        "source_url": url,
        "source_id": "web-" + sha256(url.encode("utf-8")).hexdigest()[:16],
    }


def _address_is_public(host: str) -> bool:
    """Return whether a resolved address is safe for an outbound web fetch."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global


def _parse_ip_literal(
    host: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse canonical and legacy IPv4 forms accepted by network stacks."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass

    if host and all(character in "0123456789." for character in host):
        try:
            return ipaddress.IPv4Address(socket.inet_aton(host))
        except OSError:
            pass
    return None


class _PublicAddressResolver(aiohttp.abc.AbstractResolver):
    """Resolve at connection time and reject DNS answers for non-public networks."""

    def __init__(self):
        self._resolver = aiohttp.resolver.DefaultResolver()

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_INET
    ) -> list[dict[str, Any]]:
        records = await self._resolver.resolve(host, port, family)
        if not records or any(
            not _address_is_public(record["host"]) for record in records
        ):
            raise OSError(f"Blocked non-public address for host: {host}")
        return records

    async def close(self) -> None:
        await self._resolver.close()


class WebSearchTool:
    """Bounded DDGS search with explicit, labeled development fixtures."""

    name = "web_search"
    description = "Search the web for information"
    MAX_QUERY_LENGTH = 4096
    MAX_TITLE_LENGTH = 512
    MAX_SNIPPET_LENGTH = 2000
    MAX_URL_LENGTH = 8192

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.search_engine = config.get("search_engine", "ddgs")
        if self.search_engine not in {"ddgs", "duckduckgo", "mock"}:
            raise ValueError("search_engine must be ddgs, duckduckgo, or mock")
        self.backend = "duckduckgo" if self.search_engine == "duckduckgo" else "auto"
        self.max_results = config.get("max_results", 5)
        if type(self.max_results) is not int or not 1 <= self.max_results <= 50:
            raise ValueError("max_results must be an integer between 1 and 50")
        self.timeout = config.get("search_timeout", config.get("timeout", 10))
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("search_timeout must be a finite positive number")

    @property
    def input_schema(self) -> dict:
        """Return JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to execute",
                    "minLength": 1,
                    "maxLength": self.MAX_QUERY_LENGTH,
                }
            },
            "required": ["query"],
        }

    def _failure(
        self, code: str, message: str, *, retryable: bool = False
    ) -> ToolResult:
        return ToolResult(
            success=False,
            output=message,
            error={
                "code": code,
                "message": message,
                "provider": "mock" if self.search_engine == "mock" else "ddgs",
                "retryable": retryable,
            },
        )

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Return actual results or an explicit failure, never synthetic fallback."""
        query = input.get("query")
        if not isinstance(query, str) or not query.strip():
            return self._failure(
                "invalid_input", "A non-empty query string is required"
            )
        if len(query) > self.MAX_QUERY_LENGTH:
            return self._failure("invalid_input", "Query exceeds 4096 characters")

        is_mock = self.search_engine == "mock"
        try:
            results = (
                await self._mock_search(query)
                if is_mock
                else await asyncio.wait_for(self._real_search(query), self.timeout)
            )
            output = {
                "query": query,
                "results": results,
                "count": len(results),
                "provider": "mock" if is_mock else "ddgs",
                "backend": None if is_mock else self.backend,
                "mock": is_mock,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
            if is_mock:
                output["warning"] = "Synthetic development fixtures; not web evidence."
            return ToolResult(success=True, output=output)
        except (TimeoutError, TimeoutException):
            return self._failure(
                "search_timeout", "Web search timed out", retryable=True
            )
        except RatelimitException:
            return self._failure(
                "search_rate_limited",
                "Search provider rate limited the request",
                retryable=True,
            )
        except Exception as exc:
            # Provider exceptions may contain credentials/proxy URLs or full HTML.
            # Keep the returned failure bounded and do not log the raw exception.
            logger.warning("Web search failed (%s)", type(exc).__name__)
            return self._failure(
                "search_failed",
                "Web search failed; no results were retrieved",
                retryable=True,
            )
        # CancelledError deliberately propagates to the calling task.

    async def _real_search(self, query: str) -> list[dict[str, Any]]:
        """Use DDGS's actual backend, bounded both at its client and async boundary."""

        def search_sync():
            rows = DDGS(timeout=self.timeout).text(
                query, max_results=self.max_results, backend=self.backend
            )
            if not isinstance(rows, list):
                raise ValueError("Invalid search provider response")
            results = []
            seen = set()
            for row in rows[: self.max_results]:
                if not isinstance(row, dict):
                    raise ValueError("Invalid search result")
                url = row.get("href") or row.get("url")
                if not isinstance(url, str) or len(url) > self.MAX_URL_LENGTH:
                    raise ValueError("Missing or oversized source URL")
                parsed = urlparse(url)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.hostname
                    or parsed.username
                ):
                    raise ValueError("Invalid source URL")
                if url in seen:
                    continue
                seen.add(url)
                title, snippet = row.get("title", ""), row.get("body", "")
                if not isinstance(title, str) or not isinstance(snippet, str):
                    raise ValueError("Invalid source text")
                results.append(
                    {
                        "title": title[: self.MAX_TITLE_LENGTH],
                        "url": url,
                        "snippet": snippet[: self.MAX_SNIPPET_LENGTH],
                        **_source_metadata(url),
                        "truncated": len(title) > self.MAX_TITLE_LENGTH
                        or len(snippet) > self.MAX_SNIPPET_LENGTH,
                    }
                )
            return results

        # Cancellation stops the await immediately. DDGS's synchronous HTTP work
        # may finish in its thread, subject to its own timeout; it cannot publish
        # a late tool result or turn cancellation into a successful fixture.
        return await asyncio.to_thread(search_sync)

    async def _mock_search(self, query: str) -> list[dict[str, Any]]:
        """Opt-in fixtures only. Never reached from a real provider's error path."""
        return [
            {
                "title": f"[MOCK] Result {index} for {query}"[: self.MAX_TITLE_LENGTH],
                "url": f"https://example.com/{index}",
                "snippet": "Synthetic development fixture; not retrieved web content.",
                **_source_metadata(f"https://example.com/{index}"),
                "mock": True,
                "truncated": len(f"[MOCK] Result {index} for {query}")
                > self.MAX_TITLE_LENGTH,
            }
            for index in range(1, min(self.max_results, 3) + 1)
        ]


class WebFetchTool:
    """Fetch and parse web pages with streaming support and truncation handling."""

    name = "web_fetch"
    description = """Fetch content from a web URL.

Inline content defaults to 200KB; paginate with offset/limit. save_to_file writes the full response within the configured download cap (20MB default), returning metadata + preview. download_limit can lower that cap.

The response includes `truncated` (was content cut off) and `total_bytes` (original size, when available) - use them to decide whether to re-fetch with save_to_file.

Binary content (PDFs, images, archives) cannot be returned inline as text and will be refused; use save_to_file to download it intact - the bytes are written to disk exactly as received."""

    # Default limit: 200KB is reasonable for web content
    DEFAULT_LIMIT = 200 * 1024
    CHUNK_SIZE = 8192
    PREVIEW_SIZE = 1000
    DEFAULT_DOWNLOAD_LIMIT = 20 * 1024 * 1024
    MAX_DOWNLOAD_LIMIT = 256 * 1024 * 1024

    # Content types that are always binary. Used only as a supporting signal --
    # the NUL-byte check in _looks_binary is the primary, structural test.
    #
    # "application/octet-stream" is deliberately NOT listed. It means "unknown",
    # not "binary": servers commonly fall back to it for plain text they failed
    # to identify. Treating it as binary would refuse legitimate text. Real
    # binary served under it is still caught by the NUL-byte check.
    BINARY_TYPE_PREFIXES = ("image/", "audio/", "video/", "font/")
    BINARY_TYPES = frozenset(
        {
            "application/pdf",
            "application/zip",
            "application/gzip",
            "application/x-gzip",
            "application/x-tar",
            "application/x-bzip2",
            "application/x-7z-compressed",
            "application/vnd.rar",
            "application/msword",
            "application/vnd.ms-excel",
            "application/vnd.ms-powerpoint",
            "application/epub+zip",
            "application/wasm",
            "application/java-archive",
            "application/x-shockwave-flash",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
    )

    def __init__(
        self,
        config: dict[str, Any],
        shared_session: Optional[aiohttp.ClientSession] = None,
    ):
        self.config = config
        self.timeout = config.get("timeout", 10)
        self.default_limit = config.get("default_limit", self.DEFAULT_LIMIT)
        self.max_download_bytes = config.get(
            "max_download_bytes", self.DEFAULT_DOWNLOAD_LIMIT
        )
        if (
            type(self.max_download_bytes) is not int
            or not 1 <= self.max_download_bytes <= self.MAX_DOWNLOAD_LIMIT
        ):
            raise ValueError(
                "max_download_bytes must be an integer from 1 through 268435456"
            )
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
        self.allow_private_networks = config.get("allow_private_networks", False)
        if type(self.allow_private_networks) is not bool:
            raise ValueError("allow_private_networks must be a boolean")
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
                    "description": "Save full content to this relative path beneath working_dir instead of returning in response. "
                    "Useful for large pages. Returns metadata + preview when set.",
                },
                "download_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self.max_download_bytes,
                    "description": "Maximum decoded body bytes for save_to_file. May lower, never raise, the configured cap. An incomplete download does not replace the destination.",
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
        if not isinstance(url, str) or not url:
            error_msg = "URL string is required"
            return ToolResult(
                success=False, output=error_msg, error={"message": error_msg}
            )

        save_to_file = input.get("save_to_file")
        offset = input.get("offset", 0)
        limit = input.get("limit", self.default_limit)
        download_limit = input.get("download_limit", self.max_download_bytes)
        if (
            type(download_limit) is not int
            or not 1 <= download_limit <= self.max_download_bytes
        ):
            return ToolResult(
                success=False,
                error={
                    "code": "invalid_input",
                    "message": "download_limit must be a positive integer no greater than the configured download cap",
                },
            )

        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or limit <= 0
        ):
            return ToolResult(
                success=False,
                error={
                    "code": "invalid_input",
                    "message": "offset must be a non-negative integer and limit a positive integer",
                },
            )

        destination = None
        if save_to_file is not None:
            try:
                destination = self._resolve_download_path(save_to_file)
            except ValueError as error:
                return ToolResult(
                    success=False,
                    error={"code": "invalid_input", "message": str(error)},
                )

        # Validate URL
        if not self._is_valid_url(url):
            return ToolResult(
                success=False, error={"message": f"Invalid or blocked URL: {url}"}
            )

        try:
            # Use shared session if available, otherwise create one for this request
            session = self._shared_session
            owns_session = False
            secure_shared_session = session is not None and isinstance(
                getattr(session.connector, "_resolver", None),
                _PublicAddressResolver,
            )
            if (
                session is None
                or session.closed
                or (not self.allow_private_networks and not secure_shared_session)
            ):
                connector = (
                    None
                    if self.allow_private_networks
                    else aiohttp.TCPConnector(resolver=_PublicAddressResolver())
                )
                session = aiohttp.ClientSession(connector=connector)
                owns_session = True

            try:
                async with self._request(session, url) as response:
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
                    # Content-Length describes the encoded representation, not
                    # necessarily the decompressed bytes returned by aiohttp.
                    declared_size = (
                        int(content_length_header)
                        if content_length_header
                        and content_length_header.isdigit()
                        and not response.headers.get("Content-Encoding")
                        else None
                    )

                    # Stream content with hard limit to avoid memory issues
                    if save_to_file:
                        return await self._fetch_to_file(
                            response,
                            url,
                            destination,
                            declared_size,
                            download_limit,
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
            error_msg = f"Timeout fetching {url}"
            return ToolResult(
                success=False, output=error_msg, error={"message": error_msg}
            )
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            error_msg = str(e)
            return ToolResult(
                success=False, output=error_msg, error={"message": error_msg}
            )

    @asynccontextmanager
    async def _request(self, session: aiohttp.ClientSession, url: str):
        """Check each redirect before requesting it, retaining the domain policy."""
        current_url = url
        # One deadline covers the entire redirect chain and body consumption.
        async with asyncio.timeout(self.timeout):
            for redirect_count in range(11):
                async with session.get(
                    current_url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={"User-Agent": "Amplifier/1.0"},
                    allow_redirects=False,
                ) as response:
                    location = response.headers.get("Location")
                    if response.status in {301, 302, 303, 307, 308} and location:
                        if redirect_count == 10:
                            raise ValueError("Too many redirects")
                        current_url = urljoin(str(response.url), location)
                        if not self._is_valid_url(current_url):
                            raise ValueError(
                                "Redirect targets an invalid or blocked URL"
                            )
                        continue
                    yield response
                    return

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

        # Never drain a large/infinite response just to count its bytes. When
        # truncated, only a trustworthy length header can supply the total.
        actual_total = declared_size if truncated else total_read

        # Combine chunks
        raw_content = b"".join(chunks)
        content_type = response.content_type or ""

        # Refuse to inline binary rather than force-decoding it into the
        # transcript. Decoding with errors="replace" would emit a wall of
        # replacement characters that is both useless and expensive in context.
        if self._looks_binary(raw_content, content_type):
            error_msg = (
                f"Refusing to return binary content as text "
                f"(content-type: {content_type or 'unknown'}, "
                f"{actual_total if actual_total is not None else len(raw_content)} bytes). "
                f"Decoding it would produce replacement characters, not usable text. "
                f"Use save_to_file to download it intact instead."
            )
            return ToolResult(
                success=False, output=error_msg, error={"message": error_msg}
            )

        # Decode text content
        try:
            content = raw_content.decode("utf-8", errors="replace")
        except Exception:
            content = raw_content.decode("latin-1", errors="replace")

        # Extract text if requested and HTML
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
                "requested_url": url,
                **_source_metadata(str(response.url)),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "status_code": response.status,
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
        destination: Path,
        declared_size: Optional[int],
        download_limit: Optional[int] = None,
    ) -> ToolResult:
        """Bound a complete download and atomically publish only complete content."""
        import os
        import tempfile

        cap = self.max_download_bytes if download_limit is None else download_limit

        def oversized():
            return ToolResult(
                success=False,
                error={
                    "code": "download_too_large",
                    "message": "The response exceeds the download cap; the destination was not changed",
                    "max_download_bytes": cap,
                },
            )

        if declared_size is not None and declared_size > cap:
            return oversized()
        chunks: list[bytes] = []
        total_bytes = 0

        # Enforce the actual decoded size even without a trustworthy length
        # header (including chunked and compressed responses).
        async for chunk in response.content.iter_chunked(self.CHUNK_SIZE):
            if total_bytes + len(chunk) > cap:
                return oversized()
            chunks.append(chunk)
            total_bytes += len(chunk)

        raw_content = b"".join(chunks)
        content_type = response.content_type or ""
        is_binary = self._looks_binary(raw_content, content_type)

        # Only decode when the payload is actually text. Decoding binary with
        # errors="replace" is lossy and irreversible -- the bytes cannot be
        # recovered from the resulting string.
        text = ""
        if not is_binary:
            try:
                content = raw_content.decode("utf-8", errors="replace")
            except Exception:
                content = raw_content.decode("latin-1", errors="replace")

            # Extract text if HTML
            text = (
                self._extract_text(content, content_type)
                if self.extract_text
                else content
            )

        # Write to file
        try:
            # Re-resolve immediately before the write so an existing symlink
            # cannot redirect the destination outside the configured root.
            destination = self._resolve_download_path(
                str(destination.relative_to(Path(self.working_dir).resolve()))
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, prefix=".amplifier-download-", delete=False
                ) as stream:
                    pending = Path(stream.name)
                    stream.write(raw_content if is_binary else text.encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(pending, destination)
            finally:
                if pending is not None:
                    pending.unlink(missing_ok=True)
        except Exception as e:
            return ToolResult(
                success=False, error={"message": f"Failed to write file: {e}"}
            )

        # Create preview
        if is_binary:
            preview = (
                f"[Binary content ({content_type or 'unknown type'}), "
                f"{total_bytes} bytes saved verbatim to {destination}. "
                f"No text preview available.]"
            )
            saved_bytes = len(raw_content)
        else:
            preview = text[: self.PREVIEW_SIZE]
            if len(text) > self.PREVIEW_SIZE:
                preview += f"\n\n[... {len(text) - self.PREVIEW_SIZE} more characters saved to {destination}]"
            saved_bytes = len(text.encode("utf-8"))

        return ToolResult(
            success=True,
            output={
                "url": url,
                "requested_url": url,
                **_source_metadata(str(response.url)),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "status_code": response.status,
                "content": preview,
                "content_type": content_type,
                "truncated": False,
                "total_bytes": total_bytes,
                "saved_to": str(destination),
                "saved_bytes": saved_bytes,
                "download_limit": cap,
                "content_sha256": sha256(raw_content).hexdigest(),
                "saved_sha256": sha256(
                    raw_content if is_binary else text.encode("utf-8")
                ).hexdigest(),
            },
        )

    def _looks_binary(self, raw: bytes, content_type: str) -> bool:
        """Detect binary payloads before any decode is attempted.

        The primary signal is a NUL byte, which cannot occur in valid UTF-8
        text. Unlike a strict-decode attempt or a U+FFFD scan, it is unaffected
        by the byte-window slicing in _fetch_with_limit, which can cut a
        multibyte character in half and make legitimate text look invalid.

        Content-type is a secondary signal only, for binary formats whose
        sampled window might happen to contain no NUL bytes. Anything not
        positively identified as binary is treated as text, so unusual but
        legitimate text types are never refused.
        """
        if b"\x00" in raw:
            return True

        ct = (content_type or "").lower().split(";")[0].strip()
        if not ct:
            return False
        if ct.startswith(self.BINARY_TYPE_PREFIXES):
            return True
        return ct in self.BINARY_TYPES

    def _resolve_download_path(self, file_path: Any) -> Path:
        """Resolve a caller path strictly beneath the configured working directory."""
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("save_to_file must be a non-empty relative path")
        if not self.working_dir:
            raise ValueError("save_to_file requires a configured working_dir")

        requested = Path(file_path)
        windows_path = PureWindowsPath(file_path)
        if (
            requested.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or file_path.startswith("~")
        ):
            raise ValueError("save_to_file must be relative to working_dir")

        root = Path(self.working_dir).resolve()
        destination = (root / requested).resolve()
        try:
            destination.relative_to(root)
        except ValueError as error:
            raise ValueError("save_to_file must remain within working_dir") from error
        return destination

    @staticmethod
    def _normalize_hostname(host: str) -> str:
        """Canonicalize a URL hostname for exact and subdomain matching."""
        return host.rstrip(".").encode("idna").decode("ascii").lower()

    @classmethod
    def _domain_matches(cls, host: str, pattern: str) -> bool:
        """Match a hostname exactly or at a DNS label boundary."""
        normalized_pattern = cls._normalize_hostname(pattern.lstrip("."))
        return host == normalized_pattern or host.endswith("." + normalized_pattern)

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL for safety."""
        try:
            parsed = urlparse(url)

            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc or not parsed.hostname:
                return False

            # Only allow http/https
            if parsed.scheme not in ["http", "https"]:
                return False

            if parsed.username is not None or parsed.password is not None:
                return False

            # Accessing port validates malformed and out-of-range values.
            _ = parsed.port
            host = self._normalize_hostname(parsed.hostname)

            if not self.allow_private_networks:
                literal = _parse_ip_literal(host)
                if literal is not None and not _address_is_public(str(literal)):
                    logger.warning(f"Blocked non-public address: {host}")
                    return False

            # Check blocked domains
            for blocked in self.blocked_domains:
                if self._domain_matches(host, blocked):
                    logger.warning(f"Blocked domain: {host}")
                    return False

            # Check allowed domains if configured
            if self.allowed_domains:
                allowed = any(
                    self._domain_matches(host, domain)
                    for domain in self.allowed_domains
                )
                if not allowed:
                    logger.warning(f"Domain not in allowlist: {host}")
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
