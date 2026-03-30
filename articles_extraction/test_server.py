"""
Automated test for the MCP Checkpoint CRUD tools.

Tests all 7 tools by calling the underlying functions directly
(bypassing the MCP transport layer). Uses a temporary checkpoint file.

Usage:
    python test_server.py
"""

import json
import os
import sys
import tempfile

# Patch the checkpoint file BEFORE importing server
_tmpfile = tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False, prefix="test_checkpoint_"
)
_tmpfile.write('{"completed_queries": [], "urls": {}}')
_tmpfile.close()

# Now import and patch
import server
server.CHECKPOINT_FILE = _tmpfile.name


def test_add_and_check_keywords():
    """Test keyword add (single + batch) and lookup."""
    # Single add
    result = server.add_searched_keyword("vibe coding medium")
    assert "Added 1" in result, f"Expected 1 added, got: {result}"

    # Check it exists
    assert server.is_keyword_searched("vibe coding medium") == "yes"
    assert server.is_keyword_searched("nonexistent") == "no"

    # Batch add
    result = server.add_searched_keyword([
        "AI pair programming",
        "cursor IDE",
        "vibe coding medium",  # duplicate, should be skipped
    ])
    assert "Added 2" in result, f"Expected 2 added, got: {result}"
    assert "Skipped 1" in result, f"Expected 1 skipped, got: {result}"

    # Get all
    all_kw = json.loads(server.get_all_searched_keywords())
    assert len(all_kw) == 3
    assert "vibe coding medium" in all_kw
    assert "AI pair programming" in all_kw
    assert "cursor IDE" in all_kw

    print("✅ Keyword tools passed")


def test_add_and_check_urls():
    """Test URL add (single string, dict, batch) with dedup and classification."""
    # Single string
    result = server.add_urls(
        "https://medium.com/some-article",
        query="vibe coding"
    )
    assert "Added 1" in result, f"Expected 1 added, got: {result}"

    # Check it exists
    assert server.is_url_discovered("https://medium.com/some-article") == "yes"
    assert server.is_url_discovered("https://example.com/nope") == "no"

    # Duplicate with different query → should merge
    result = server.add_urls(
        "https://MEDIUM.com/some-article/",  # same after normalization
        query="AI coding"
    )
    assert "Merged" in result, f"Expected merge, got: {result}"
    assert "Added 0" in result, f"Expected 0 new, got: {result}"

    # Single dict
    result = server.add_urls({
        "url": "https://substack.com/my-post",
        "query": "substack search"
    })
    assert "Added 1" in result

    # Batch (mixed strings and dicts)
    result = server.add_urls([
        "https://stackoverflow.com/questions/12345",
        {"url": "https://cursor.com/blog/something", "query": "cursor blog"},
        "https://medium.com/some-article",  # duplicate
        "https://example-personal.com/post",
    ], query="batch query")
    assert "Added 3" in result, f"Expected 3 added, got: {result}"
    assert "Merged" in result, f"Expected merge, got: {result}"

    print("✅ URL tools passed")


def test_classification():
    """Test auto-classification of URLs."""
    data = json.loads(server.get_all_urls())
    
    # Check medium classification
    medium_key = "https://medium.com/some-article"
    assert data[medium_key]["source_type"] == "medium"

    # Check substack
    substack_urls = json.loads(server.get_all_urls("substack"))
    assert len(substack_urls) == 1

    # Check stackoverflow
    so_urls = json.loads(server.get_all_urls("stackoverflow"))
    assert len(so_urls) == 1

    # Check corporate blog
    corp_urls = json.loads(server.get_all_urls("corporate_blog"))
    assert len(corp_urls) == 1

    # Check personal blog
    personal_urls = json.loads(server.get_all_urls("personal_blog"))
    assert len(personal_urls) == 1

    print("✅ Classification passed")


def test_deduplication():
    """Test that URL normalization deduplicates correctly."""
    data = json.loads(server.get_all_urls("medium"))
    assert len(data) == 1, f"Expected 1 medium URL, got {len(data)}"

    # The medium URL should have 3 queries merged:
    # "vibe coding" (first add), "AI coding" (second add), "batch query" (batch add)
    medium_entry = list(data.values())[0]
    assert len(medium_entry["queries"]) == 3, (
        f"Expected 3 queries, got: {medium_entry['queries']}"
    )

    print("✅ Deduplication passed")


def test_stats():
    """Test statistics endpoint."""
    stats = json.loads(server.get_stats())
    assert stats["total_keywords_searched"] == 3
    assert stats["total_unique_urls"] == 5
    assert stats["urls_by_source_type"]["medium"] == 1
    assert stats["urls_by_source_type"]["substack"] == 1
    assert stats["urls_by_source_type"]["stackoverflow"] == 1
    assert stats["urls_by_source_type"]["corporate_blog"] == 1
    assert stats["urls_by_source_type"]["personal_blog"] == 1

    print("✅ Stats passed")


if __name__ == "__main__":
    try:
        test_add_and_check_keywords()
        test_add_and_check_urls()
        test_classification()
        test_deduplication()
        test_stats()
        print("\n🎉 All tests passed!")
    finally:
        # Cleanup
        os.unlink(_tmpfile.name)
        if os.path.exists(_tmpfile.name + ".tmp"):
            os.unlink(_tmpfile.name + ".tmp")
