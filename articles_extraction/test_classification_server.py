"""
Automated tests for the Article Classification MCP server.

Tests all 5 tools by calling the underlying ArticleClassifier methods
directly (bypassing MCP transport). Uses a temporary checkpoint file
and a temporary directory with fake article JSONs.

Usage:
    python test_classification_server.py
"""

import json
import os
import sys
import tempfile
import shutil

# ---------------------------------------------------------------------------
# Set up a temp environment BEFORE importing the module
# ---------------------------------------------------------------------------

_tmpdir = tempfile.mkdtemp(prefix="test_articles_")
_tmpcheckpoint = os.path.join(_tmpdir, "relevant_checkpoint.json")

# Create 3 fake articles
FAKE_ARTICLES = {
    "article_aaa": {
        "url": "https://medium.com/vibe-coding-rocks",
        "normalized_url": "https://medium.com/vibe-coding-rocks",
        "source_type": "medium",
        "title": "Vibe Coding Changed My Life",
        "author": "Test Author",
        "date": "2025-06-15",
        "text": "Vibe coding is the best thing since sliced bread. " * 100,
        "excerpt": "Vibe coding is the best thing...",
        "word_count": 900,
        "scraped_at": "2026-02-24T08:00:00+00:00",
        "scrape_method": "trafilatura",
        "queries": ["vibe coding"],
    },
    "article_bbb": {
        "url": "https://substack.com/ai-coding-tutorial",
        "normalized_url": "https://substack.com/ai-coding-tutorial",
        "source_type": "substack",
        "title": "How to Use GitHub Copilot: A Tutorial",
        "author": "Tutorial Writer",
        "date": "2025-07-01",
        "text": "Step 1: Install Copilot. Step 2: Open VS Code. " * 50,
        "excerpt": "Step 1: Install Copilot...",
        "word_count": 500,
        "scraped_at": "2026-02-24T09:00:00+00:00",
        "scrape_method": "trafilatura",
        "queries": ["copilot tutorial"],
    },
    "article_ccc": {
        "url": "https://example.com/unrelated-post",
        "normalized_url": "https://example.com/unrelated-post",
        "source_type": "personal_blog",
        "title": "Best Pizza Recipes 2025",
        "author": "Chef",
        "date": "2025-08-01",
        "text": "Pizza is delicious. Here is how to make pizza. " * 30,
        "excerpt": "Pizza is delicious...",
        "word_count": 300,
        "scraped_at": "2026-02-24T10:00:00+00:00",
        "scrape_method": "trafilatura",
        "queries": ["pizza"],
    },
}

# Write fake articles to temp dir
_articles_dir = os.path.join(_tmpdir, "articles")
os.makedirs(_articles_dir)
for aid, content in FAKE_ARTICLES.items():
    with open(os.path.join(_articles_dir, f"{aid}.json"), "w") as f:
        json.dump(content, f)

# Now import and configure
import classification_server

classifier = classification_server.ArticleClassifier(
    articles_dir=_articles_dir,
    checkpoint_file=_tmpcheckpoint,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_next_article():
    """get_next_article returns an article with expected fields."""
    result = json.loads(classifier.get_next_article())
    assert "article_id" in result, f"Missing article_id: {result}"
    assert "title" in result
    assert "url" in result
    assert "text" in result
    assert "remaining_after_this" in result
    assert result["remaining_after_this"] == 2  # 3 total, serving 1st
    print("✅ get_next_article returns valid article")


def test_submit_classification():
    """submit_classification persists correctly."""
    # Classify the first article
    article = json.loads(classifier.get_next_article())
    aid = article["article_id"]

    result = classifier.submit_classification(aid, "yes", "yes")
    assert "Saved" in result, f"Expected 'Saved', got: {result}"
    assert "1 classified" in result

    # Verify it's persisted
    with open(_tmpcheckpoint, "r") as f:
        data = json.load(f)
    assert aid in data["classified"]
    assert data["classified"][aid]["is_relevant"] == "yes"
    assert data["classified"][aid]["has_opinion"] == "yes"
    print("✅ submit_classification persists correctly")


def test_submit_classification_validation():
    """submit_classification rejects invalid enum values."""
    result = classifier.submit_classification("article_bbb", "maybe", "yes")
    assert "ERROR" in result, f"Expected ERROR for invalid value, got: {result}"

    result = classifier.submit_classification("article_bbb", "yes", "sometimes")
    assert "ERROR" in result, f"Expected ERROR for invalid value, got: {result}"
    print("✅ submit_classification validates enums")


def test_next_skips_classified():
    """get_next_article skips already-classified articles."""
    article = json.loads(classifier.get_next_article())
    # The first article was classified above, so this should be a different one
    assert article["article_id"] != "article_aaa", (
        "Should not return already-classified article"
    )
    print("✅ get_next_article skips classified articles")


def test_skip_article():
    """skip_article marks an article and excludes it from future next calls."""
    article = json.loads(classifier.get_next_article())
    aid = article["article_id"]

    result = classifier.skip_article(aid, "unreadable")
    assert "Skipped" in result

    # Verify persistence
    with open(_tmpcheckpoint, "r") as f:
        data = json.load(f)
    assert aid in data["skipped"]
    assert data["skipped"][aid]["reason"] == "unreadable"
    print("✅ skip_article works correctly")


def test_classify_remaining():
    """Classify the last remaining article."""
    article = json.loads(classifier.get_next_article())
    aid = article["article_id"]
    result = classifier.submit_classification(aid, "no", "no")
    assert "Saved" in result
    print("✅ Classified remaining article")


def test_all_done():
    """get_next_article returns ALL_DONE when nothing is left."""
    result = classifier.get_next_article()
    assert result == "ALL_DONE", f"Expected ALL_DONE, got: {result}"
    print("✅ ALL_DONE when all articles processed")


def test_get_classification_stats():
    """Stats reflect correct counts."""
    stats = json.loads(classifier.get_classification_stats())
    assert stats["total_articles"] == 3
    assert stats["classified"] == 2  # article_aaa + one more
    assert stats["skipped"] == 1
    assert stats["pending"] == 0
    print("✅ get_classification_stats correct")


def test_get_classification_results():
    """Results filtering works."""
    # All results
    all_results = json.loads(classifier.get_classification_results())
    assert len(all_results) == 2

    # Filter by relevant=yes
    relevant = json.loads(classifier.get_classification_results(filter_relevant="yes"))
    assert all(v["is_relevant"] == "yes" for v in relevant.values())

    # Filter by has_opinion=no
    no_opinion = json.loads(classifier.get_classification_results(filter_opinion="no"))
    assert all(v["has_opinion"] == "no" for v in no_opinion.values())

    print("✅ get_classification_results filtering works")


def test_cross_session_persistence():
    """Simulate a new session by creating a fresh classifier instance."""
    new_classifier = classification_server.ArticleClassifier(
        articles_dir=_articles_dir,
        checkpoint_file=_tmpcheckpoint,
    )
    # Should immediately return ALL_DONE since all were processed
    result = new_classifier.get_next_article()
    assert result == "ALL_DONE", "New session should see prior progress"

    # Stats should match
    stats = json.loads(new_classifier.get_classification_stats())
    assert stats["classified"] == 2
    assert stats["skipped"] == 1
    print("✅ Cross-session persistence works")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        test_get_next_article()
        test_submit_classification()
        test_submit_classification_validation()
        test_next_skips_classified()
        test_skip_article()
        test_classify_remaining()
        test_all_done()
        test_get_classification_stats()
        test_get_classification_results()
        test_cross_session_persistence()
        print("\n🎉 All tests passed!")
    finally:
        shutil.rmtree(_tmpdir, ignore_errors=True)
