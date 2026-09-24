# Amplifier Web Tools Module

Web tools for searching and fetching content from the internet.

## Features

### WebSearchTool

- **Real web search** using DDGS (no API key required; its default is automatic backend selection)
- Explicit failures for provider errors, rate limits, and timeouts; no synthetic fallback
- Bounded result counts, titles, and snippets with source URL attribution
- Visibly labeled mock mode available only through explicit development configuration

### WebFetchTool

- Fetch and parse web pages
- Extract text from HTML content
- Domain allowlist/blocklist checks on the original URL and every redirect
- Inline byte windows, bounded complete downloads and timeout protection
- Atomic PDF/binary saves with original-byte and saved-byte SHA-256 evidence
- Requested and final source URLs, with source IDs shared by search and fetch

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

```bash
uv pip install -e .
```

## Usage

### Web Search

```python
from amplifier_module_tool_web import WebSearchTool

# Create tool with config
tool = WebSearchTool({"max_results": 5})

# Execute search
result = await tool.execute({"query": "Python programming"})

# Check result.success before using result.output as evidence.
# Successful output includes query, results, count, provider, backend,
# mock (False in normal use), and retrieved_at (UTC ISO 8601).
# Each result includes title, url, snippet, source_url, source_id, truncated.
```

### Web Fetch

```python
from amplifier_module_tool_web import WebFetchTool

# Create tool with config
tool = WebFetchTool({
    "timeout": 10,
    "extract_text": True
})

# Fetch a webpage
result = await tool.execute({"url": "https://example.com"})

# Result includes extracted text content
```

## Configuration

Both tools mount through module `tool-web`, with tool names `web_search` and
`web_fetch`. Existing input names and successful output fields remain available.

| Setting | Default | Meaning |
| --- | --- | --- |
| `search_engine` | `ddgs` | Real DDGS search using its automatic backend selection. `duckduckgo` selects that specific DDGS backend. `mock` explicitly enables development fixtures. Other values fail configuration. |
| `max_results` | `5` | Maximum search results; integer from 1 through 50. |
| `search_timeout` | `timeout` or `10` | Positive finite search deadline in seconds; also passed to the DDGS HTTP client. |
| `timeout` | `10` | Fetch deadline in seconds, including redirects and body consumption. Also the fallback for `search_timeout`. |
| `default_limit` | `204800` | Default inline fetch byte window. Callers may override it using `limit` and paginate using `offset`. |
| `extract_text` | `true` | Extract text from HTML. |
| `max_download_bytes` | `20971520` (20 MiB) | Maximum decoded complete-download body; integer from1 through268435456. A caller may lower it with `download_limit`, never raise host policy. |
| `allowed_domains` | `[]` | Hostname allowlist; entries match the exact host and its subdomains at DNS label boundaries. Empty allows all otherwise permitted public hosts. |
| `blocked_domains` | local-address patterns | Hostname blocklist, checked before every request including redirect hops. Entries match the exact host and its subdomains at DNS label boundaries. |
| `allow_private_networks` | `false` | Explicitly permit loopback, private, link-local, reserved, and other non-public destinations. Intended only for trusted local development. |
| `working_dir` | session capability | Required sandbox root for `save_to_file`; caller paths must be relative and remain beneath it. |

Search queries must be non-empty strings of at most 4096 characters. Returned
titles and snippets are capped at 512 and 2000 characters; `truncated` indicates
clipping. Source URLs are preserved exactly (up to 8192 characters); malformed
provider results fail instead of receiving invented attribution. Duplicate exact
URLs are returned once. `source_id` is `web-` followed by the first 16 hexadecimal
characters of the SHA-256 of the exact source URL; it identifies that URL, not a
verified claim or an immutable content snapshot.

Search failures return `success: false` with an error containing `code`,
`message`, `provider`, and `retryable`. Codes are `invalid_input`, `search_timeout`,
`search_rate_limited`, and `search_failed`. Raw provider exception text is not
returned. An actual empty result list is a successful result with `count: 0`.
DDGS can instead raise an exception for no results; that remains an explicit
failure because it cannot reliably be distinguished from a provider outage.

Mock mode is for tests/development only:

```python
tool = WebSearchTool({"search_engine": "mock"})
```

The output has `mock: true`, `provider: "mock"`, and a warning; each row has a
`[MOCK]` title and `mock: true`. These are synthetic fixtures, never web evidence.
Previously, `search_engine` was ignored and provider errors silently produced
mock success. Remove any old `search_engine: mock` setting from production
configuration to use real search. `api_key` is not used by the DDGS backend.

Fetch keeps the original `url` field and adds `requested_url`, `source_url` (the
final URL after redirects), `source_id`, `retrieved_at`, and `status_code`.
`offset` must be a non-negative integer and `limit` a positive integer. A
truncated inline fetch stops reading instead of draining the entire response.
`total_bytes` is `null` when the total cannot be known without reading more; a
compressed response's Content-Length is not mistaken for its decoded size.
The byte window applies before optional HTML extraction; the truncation notice
adds a small amount of text. `save_to_file` still requests full content and returns
a bounded preview, preserving binary files byte for byte. Complete downloads,
including PDFs, stop at the configured `max_download_bytes` ceiling (20 MiB by
default). `download_limit` may lower that ceiling for one request. Both declared
length and actual decoded streamed bytes are checked; compressed or chunked
responses cannot bypass the limit. A `download_too_large` failure does not save a
partial file or replace an existing destination. Timeout and cancellation also
leave the destination unchanged. A successful full download is atomically saved;
temporary files are removed after failed writes. Source and saved SHA-256 fields
identify the fetched bytes and any HTML text transformation separately.

Fetch rejects URL userinfo, non-HTTP schemes, non-public IP literals, and DNS
answers containing any non-public address by default. DNS is checked by the
connector at connection time, and the same policy applies to every redirect hop.
Setting `blocked_domains: []` removes only configured hostname blocks; it does
not disable the non-public network restriction. Trusted local development must
set `allow_private_networks: true` explicitly.

`save_to_file` no longer accepts absolute, home-relative, drive-qualified, UNC,
or parent-traversal paths. It requires `working_dir` and resolves the destination
inside that directory before downloading and again immediately before the atomic
replacement.

PDF retrieval preserves the original binary document for an installed PDF reader
or artifact runtime; this tool does not extract page text or perform OCR. Binary
bytes are never decoded into fake text evidence. The maximum configurable ceiling
is256 MiB, and complete bodies are buffered within that finite bound.

Cancellation propagates from both tools. DDGS is synchronous: cancelling an await
does not forcibly terminate an already-running provider thread, which may finish
under its own timeout. It cannot publish a late result. Fetch closes owned HTTP
sessions and leaves shared sessions open for their owner to manage.

The existing domain matching policy is unchanged; redirect checking does not add
DNS/IP resolution enforcement. Callers requiring network-level restrictions must
apply them at the host or network boundary.

## Dependencies

- `ddgs`: DDGS metasearch client (no API key required)
- `aiohttp`: Async HTTP client
- `beautifulsoup4`: HTML parsing and text extraction
- `amplifier-core`: Core amplifier functionality

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
