"""
MCP Server for Vibe-Coding Article Crawling Checkpoint Management.

This stdio-based MCP server exposes CRUD tools that allow an AI agent
(e.g., Gemini CLI) to manage a checkpoint database (`checkpoint.json`)
during the article crawling process.

The checkpoint tracks two things:
    1. **Searched keywords/queries** — so the agent doesn't repeat searches.
    2. **Discovered URLs** — deduplicated, auto-classified by source type.

Run this server:
    python server.py

The server communicates via stdio (stdin/stdout) using the MCP protocol.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from urllib.parse import urlparse
from typing import Optional, Union

from mcp.server.fastmcp import FastMCP

# =============================================================================
# Configuration
# =============================================================================

# Resolve paths relative to this file so it works regardless of cwd
_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(_DIR, "checkpoint.json")

# Known corporate blog domains (used for auto-classification)
CORPORATE_BLOG_DOMAINS: set[str] = {
    "cursor.com",
    "github.blog",
    "openai.com",
    "anthropic.com",
    "blog.replit.com",
    "sourcegraph.com",
    "codeium.com",
    "tabnine.com",
    "vercel.com",
    "aws.amazon.com",
    "devblogs.microsoft.com",
    "ai.meta.com",
    "blog.google",
    "huggingface.co",
    "pinecone.io",
    "blog.langchain.dev",
    "databricks.com",
    "supabase.com",
    "blog.cloudflare.com",
    "fly.io",
    "blog.jetbrains.com",
    "blog.postman.com",
    "linear.app",
    "netflixtechblog.com",
    "uber.com",
    "discord.com",
    "engineering.atspotify.com",
}


# =============================================================================
# URL Utilities
# =============================================================================

def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication.

    Strips trailing slashes, fragments (#...), and lowercases everything.
    Preserves query parameters.

    Examples:
        >>> normalize_url("https://Medium.COM/some-article/")
        'https://medium.com/some-article'
        >>> normalize_url("https://example.com/page#section")
        'https://example.com/page'
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized.lower()


def classify_source_type(url: str) -> str:
    """Auto-classify a URL into a source type based on its domain.

    Classification rules (checked in order):
        1. medium.com / towardsdatascience.com → "medium"
        2. substack.com → "substack"
        3. stackoverflow.com / stackexchange.com → "stackoverflow"
        4. Any domain in CORPORATE_BLOG_DOMAINS → "corporate_blog"
        5. Everything else → "personal_blog"

    Args:
        url: The full URL string to classify.

    Returns:
        One of: "medium", "substack", "stackoverflow",
                "corporate_blog", "personal_blog".
    """
    domain = urlparse(url).netloc.lower()

    if "medium.com" in domain or "towardsdatascience.com" in domain:
        return "medium"
    if "substack.com" in domain:
        return "substack"
    if "stackoverflow.com" in domain or "stackexchange.com" in domain:
        return "stackoverflow"

    for corp_domain in CORPORATE_BLOG_DOMAINS:
        if corp_domain in domain:
            return "corporate_blog"

    return "personal_blog"


# =============================================================================
# Checkpoint Persistence
# =============================================================================

def _load_checkpoint() -> dict:
    """Load checkpoint data from disk. Returns empty schema if file missing."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_queries": [], "urls": {}}


def _save_checkpoint(data: dict) -> None:
    """Atomically save checkpoint data to disk."""
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CHECKPOINT_FILE)


# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP(
    "Crawl-Checkpoint",
    instructions=(
        "This server manages a checkpoint database for article crawling. "
        "Use keyword tools to track which searches you've done, "
        "and URL tools to store and deduplicate discovered article links."
    ),
)


# ---------------------------------------------------------------------------
# Keyword / Query Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def add_searched_keyword(keywords: Union[str, list[str]]) -> str:
    """Mark one or more keywords as searched in the checkpoint.

    Use this AFTER you finish searching for a keyword, so you don't
    repeat the same search in future sessions.

    Supports two calling styles:
        • Single:  add_searched_keyword("vibe coding medium")
        • Batch:   add_searched_keyword(["vibe coding", "AI pair programming"])

    Duplicate keywords (already marked as searched) are silently skipped.

    Args:
        keywords: A single keyword string, or a list of keyword strings.

    Returns:
        A summary of how many keywords were added vs. already existed.
    """
    data = _load_checkpoint()

    if isinstance(keywords, str):
        keywords = [keywords]

    existing = set(data["completed_queries"])
    added = []
    skipped = []

    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        if kw in existing:
            skipped.append(kw)
        else:
            existing.add(kw)
            data["completed_queries"].append(kw)
            added.append(kw)

    _save_checkpoint(data)

    parts = [f"Added {len(added)} keyword(s)."]
    if skipped:
        parts.append(f"Skipped {len(skipped)} already-searched keyword(s).")
    parts.append(f"Total searched keywords: {len(data['completed_queries'])}.")
    return " ".join(parts)


@mcp.tool()
def is_keyword_searched(keyword: str) -> str:
    """Check whether a specific keyword has already been searched.

    Args:
        keyword: The exact keyword string to check.

    Returns:
        "yes" if the keyword was already searched, "no" otherwise.
    """
    data = _load_checkpoint()
    found = keyword.strip() in data["completed_queries"]
    return "yes" if found else "no"


@mcp.tool()
def get_all_searched_keywords() -> str:
    """Return the full list of all keywords that have been searched so far.

    Returns:
        A JSON array of keyword strings. Returns "[]" if none exist.
    """
    data = _load_checkpoint()
    return json.dumps(data["completed_queries"], indent=2)


# ---------------------------------------------------------------------------
# URL Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def add_urls(
    urls: Union[str, dict, list[Union[str, dict]]],
    query: Optional[str] = None,
) -> str:
    """Add one or more discovered URLs to the checkpoint.

    URLs are automatically deduplicated (by normalized form) and
    auto-classified into a source type (medium, substack, stackoverflow,
    corporate_blog, or personal_blog).

    Supports multiple calling styles:

        • Single string:
            add_urls("https://medium.com/some-article", query="vibe coding")

        • Single dict (with optional metadata):
            add_urls({"url": "https://example.com/post", "query": "AI coding"})

        • Batch (list of strings and/or dicts):
            add_urls([
                "https://medium.com/article-1",
                {"url": "https://substack.com/post", "query": "cursor IDE"},
            ], query="default query for strings")

    When a dict is provided, it may include:
        - "url" (required): The URL string.
        - "query" (optional): The search query that found this URL.
          Overrides the top-level `query` param for this entry.

    If a URL already exists, the query is appended to its `queries` list
    (no duplicate entry is created).

    Args:
        urls:  A URL string, a URL dict, or a list of URL strings/dicts.
        query: Default query string to associate with URLs provided as
               plain strings. Dicts can override this with their own "query".

    Returns:
        A summary of URLs added, duplicates merged, and totals.
    """
    data = _load_checkpoint()
    now = datetime.now().isoformat()

    # Normalize input to a list of dicts
    if isinstance(urls, str):
        urls = [{"url": urls}]
    elif isinstance(urls, dict):
        urls = [urls]
    else:
        urls = [
            item if isinstance(item, dict) else {"url": item}
            for item in urls
        ]

    added = 0
    merged = 0

    for item in urls:
        raw_url = item.get("url", "").strip()
        if not raw_url:
            continue

        norm = normalize_url(raw_url)
        item_query = item.get("query", query) or ""
        source_type = classify_source_type(raw_url)

        if norm in data["urls"]:
            # Merge: append query if not already tracked
            existing = data["urls"][norm]
            if item_query and item_query not in existing.get("queries", []):
                existing.setdefault("queries", []).append(item_query)
            merged += 1
        else:
            data["urls"][norm] = {
                "original_url": raw_url,
                "source_type": source_type,
                "discovery_method": "gemini_cli",
                "queries": [item_query] if item_query else [],
                "discovered_at": now,
            }
            added += 1

    _save_checkpoint(data)

    parts = [f"Added {added} new URL(s)."]
    if merged:
        parts.append(f"Merged queries for {merged} existing URL(s).")
    parts.append(f"Total unique URLs: {len(data['urls'])}.")
    return " ".join(parts)


@mcp.tool()
def is_url_discovered(url: str) -> str:
    """Check whether a specific URL has already been discovered.

    The check is done against the normalized form of the URL
    (lowercased, trailing slashes and fragments removed).

    Args:
        url: The URL string to check.

    Returns:
        "yes" if the URL exists in the checkpoint, "no" otherwise.
    """
    data = _load_checkpoint()
    norm = normalize_url(url.strip())
    return "yes" if norm in data["urls"] else "no"


@mcp.tool()
def get_all_urls(source_type: Optional[str] = None) -> str:
    """Return all discovered URLs with their metadata.

    Optionally filter by source type.

    Args:
        source_type: If provided, only return URLs matching this type.
                     Valid values: "medium", "substack", "stackoverflow",
                     "corporate_blog", "personal_blog".
                     If omitted or empty, returns all URLs.

    Returns:
        A JSON object mapping normalized URLs to their metadata.
        Returns "{}" if no URLs match.
    """
    data = _load_checkpoint()
    urls = data["urls"]

    if source_type:
        source_type = source_type.strip().lower()
        urls = {
            k: v for k, v in urls.items()
            if v.get("source_type") == source_type
        }

    return json.dumps(urls, indent=2, ensure_ascii=False)


@mcp.tool()
def get_stats() -> str:
    """Return summary statistics about the crawling progress.

    Returns a JSON object with:
        - total_keywords_searched: Number of completed search queries.
        - total_unique_urls: Number of unique URLs discovered.
        - urls_by_source_type: Breakdown of URL counts per source type.

    Returns:
        A JSON string with the statistics.
    """
    data = _load_checkpoint()

    source_counts: dict[str, int] = {}
    for entry in data["urls"].values():
        st = entry.get("source_type", "unknown")
        source_counts[st] = source_counts.get(st, 0) + 1

    stats = {
        "total_keywords_searched": len(data["completed_queries"]),
        "total_unique_urls": len(data["urls"]),
        "urls_by_source_type": dict(
            sorted(source_counts.items(), key=lambda x: -x[1])
        ),
    }
    return json.dumps(stats, indent=2)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
