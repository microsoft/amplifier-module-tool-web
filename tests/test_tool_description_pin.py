"""Byte-for-byte pins on this module's tool descriptions.

Tool descriptions render into the tool-schema block of EVERY request of every
session, whether or not the tool is ever called. They are therefore a
measured cost surface, not prose: the lean wording below was measured and
shipped deliberately, and it must not drift back.

Provenance of the pinned text:
  amplifier-foundation main, PR #372 (4384805),
  docs/lanes/zc6t-lean-head-ship/patches/tool-descriptions/web_fetch.lean.txt

If you intend to change a description, change the pin in the same commit and
say what the new character count is. A silent edit is the failure this
guards against.
"""

from amplifier_module_tool_web import WebFetchTool
from amplifier_module_tool_web import WebSearchTool

# web_fetch: stock 632 chars -> lean 549 chars (-83).
WEB_FETCH_DESCRIPTION_V1 = """Fetch content from a web URL.

Content is limited to 200KB by default; for more, set save_to_file to write the full content to a file (returns metadata + preview), or paginate with offset/limit.

The response includes `truncated` (was content cut off) and `total_bytes` (original size, when available) - use them to decide whether to re-fetch with save_to_file.

Binary content (PDFs, images, archives) cannot be returned inline as text and will be refused; use save_to_file to download it intact - the bytes are written to disk exactly as received."""

# web_search: already byte-identical to the measured v1 text -- a verified
# no-op, deliberately left unedited (zc6t finding F2).
WEB_SEARCH_DESCRIPTION_V1 = "Search the web for information"


def test_web_fetch_description_is_pinned_byte_for_byte():
    assert WebFetchTool.description == WEB_FETCH_DESCRIPTION_V1


def test_web_fetch_description_char_count():
    assert len(WebFetchTool.description) == 549


def test_web_search_description_is_pinned_byte_for_byte():
    assert WebSearchTool.description == WEB_SEARCH_DESCRIPTION_V1


def test_web_search_description_char_count():
    assert len(WebSearchTool.description) == 30


def test_binary_content_guidance_survives():
    """Guard the one clause a fuzzy patch apply silently deleted.

    The shipped web_fetch.patch is malformed -- its final removal line and
    final addition line are concatenated with no separating newline. `git
    apply` rejects it outright, but `patch -p1` accepts it "with fuzz 2" and
    produces a 430-char hybrid that drops this paragraph entirely while
    retaining a stale stock bullet. The description was hand-ported from
    web_fetch.lean.txt instead. This test fails if that loss ever lands.
    """
    assert "Binary content (PDFs, images, archives)" in WebFetchTool.description
    assert "the bytes are written to disk exactly as received" in WebFetchTool.description
