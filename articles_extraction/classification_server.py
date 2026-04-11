"""
MCP Server for Article Classification (Relevance & Opinion Detection).

This stdio-based MCP server feeds scraped articles one-at-a-time to an AI
agent (e.g., Gemini CLI) and records the agent's classification for each:

    1. Is the article relevant to Vibe-Coding / AI-Assisted Coding?
    2. Does the article express an opinion about Vibe-Coding / AI-Assisted Coding?

All progress is persisted to `relevant_checkpoint.json` so work can
resume across sessions without overlap.

Run this server:
    python classification_server.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from mcp.server.fastmcp import FastMCP

# =============================================================================
# Configuration
# =============================================================================

_DIR = os.path.dirname(os.path.abspath(__file__))
SCRAPED_ARTICLES_DIR = os.path.join(_DIR, "scraped_articles")
CLASSIFICATION_CHECKPOINT_FILE = os.path.join(_DIR, "relevant_checkpoint.json")

# Max words to send to the LLM per article
MAX_WORDS = 2000


# =============================================================================
# ArticleClassifier — manages queue + checkpoint persistence
# =============================================================================

class ArticleClassifier:
    """Manages incremental classification of scraped articles.

    State is persisted to a JSON checkpoint file so that:
      - Work resumes across sessions without overlap.
      - Every classification is saved immediately to disk.
    """

    def __init__(
        self,
        articles_dir: str = SCRAPED_ARTICLES_DIR,
        checkpoint_file: str = CLASSIFICATION_CHECKPOINT_FILE,
    ):
        self.articles_dir = articles_dir
        self.checkpoint_file = checkpoint_file
        self._data: dict | None = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Load checkpoint from disk. Returns empty schema if missing."""
        if self._data is not None:
            return self._data
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {"classified": {}, "skipped": {}}
        return self._data

    def _save(self) -> None:
        """Atomically persist checkpoint to disk."""
        data = self._load()
        os.makedirs(os.path.dirname(self.checkpoint_file) or ".", exist_ok=True)
        tmp = self.checkpoint_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.checkpoint_file)

    def _invalidate_cache(self) -> None:
        """Force re-read from disk on next access."""
        self._data = None

    # ------------------------------------------------------------------
    # Article discovery
    # ------------------------------------------------------------------

    def _all_article_ids(self) -> list[str]:
        """Return sorted list of article IDs from the scraped_articles dir."""
        if not os.path.isdir(self.articles_dir):
            return []
        return sorted(
            f[:-5]  # strip ".json"
            for f in os.listdir(self.articles_dir)
            if f.endswith(".json")
        )

    def _pending_ids(self) -> list[str]:
        """Return article IDs that have not been classified or skipped."""
        data = self._load()
        done = set(data["classified"]) | set(data["skipped"])
        return [aid for aid in self._all_article_ids() if aid not in done]

    def _load_article(self, article_id: str) -> dict | None:
        """Load a single article JSON from disk."""
        path = os.path.join(self.articles_dir, f"{article_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _truncate_text(text: str, max_words: int = MAX_WORDS) -> str:
        """Return the first `max_words` words of text."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + f"\n\n[... truncated at {max_words} words, original had {len(words)} words ...]"

    # ------------------------------------------------------------------
    # Public API (called by MCP tools)
    # ------------------------------------------------------------------

    def get_next_article(self) -> str:
        """Return the next unclassified article as a JSON string.

        Returns a JSON object with keys:
            article_id, title, url, source_type, word_count, text

        Returns the string "ALL_DONE" when no articles remain.
        """
        self._invalidate_cache()  # re-read in case another session wrote
        pending = self._pending_ids()
        if not pending:
            return "ALL_DONE"

        article_id = pending[0]
        article = self._load_article(article_id)
        if article is None:
            # File disappeared — skip it silently and try next
            return self.get_next_article()

        payload = {
            "article_id": article_id,
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "source_type": article.get("source_type", ""),
            "word_count": article.get("word_count", 0),
            "text": self._truncate_text(article.get("text", "")),
        }
        remaining = len(pending) - 1
        payload["remaining_after_this"] = remaining
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def submit_classification(
        self,
        article_id: str,
        is_relevant: str,
        has_opinion: str,
    ) -> str:
        """Record the classification for an article.

        Args:
            article_id:  The article ID returned by get_next_article.
            is_relevant: "yes", "no", or "partially".
            has_opinion: "yes" or "no".

        Returns:
            Confirmation message with updated counts.
        """
        # Validate enums
        is_relevant = is_relevant.strip().lower()
        has_opinion = has_opinion.strip().lower()

        valid_relevance = {"yes", "no", "partially"}
        valid_opinion = {"yes", "no"}

        if is_relevant not in valid_relevance:
            return (
                f"ERROR: is_relevant must be one of {valid_relevance}, "
                f"got '{is_relevant}'"
            )
        if has_opinion not in valid_opinion:
            return (
                f"ERROR: has_opinion must be one of {valid_opinion}, "
                f"got '{has_opinion}'"
            )

        self._invalidate_cache()
        data = self._load()

        # Load article metadata for reference
        article = self._load_article(article_id)
        entry = {
            "url": article.get("url", "") if article else "",
            "title": article.get("title", "") if article else "",
            "source_type": article.get("source_type", "") if article else "",
            "is_relevant": is_relevant,
            "has_opinion": has_opinion,
            "classified_at": datetime.now().isoformat(),
        }
        data["classified"][article_id] = entry
        self._save()

        total_classified = len(data["classified"])
        total_skipped = len(data["skipped"])
        total_pending = len(self._pending_ids())

        return (
            f"Saved classification for '{article_id}'. "
            f"Progress: {total_classified} classified, "
            f"{total_skipped} skipped, {total_pending} pending."
        )

    def skip_article(self, article_id: str, reason: str = "") -> str:
        """Mark an article as skipped.

        Args:
            article_id: The article ID to skip.
            reason:     Why it was skipped (optional).

        Returns:
            Confirmation message.
        """
        self._invalidate_cache()
        data = self._load()
        data["skipped"][article_id] = {
            "reason": reason.strip(),
            "skipped_at": datetime.now().isoformat(),
        }
        self._save()

        total_pending = len(self._pending_ids())
        return (
            f"Skipped article '{article_id}'. "
            f"Reason: {reason or 'none given'}. "
            f"{total_pending} articles remaining."
        )

    def get_classification_stats(self) -> str:
        """Return classification progress and distribution stats."""
        self._invalidate_cache()
        data = self._load()
        all_ids = self._all_article_ids()
        classified = data["classified"]
        skipped = data["skipped"]

        # Relevance breakdown
        relevance_counts = {"yes": 0, "no": 0, "partially": 0}
        opinion_counts = {"yes": 0, "no": 0}
        for entry in classified.values():
            r = entry.get("is_relevant", "")
            o = entry.get("has_opinion", "")
            if r in relevance_counts:
                relevance_counts[r] += 1
            if o in opinion_counts:
                opinion_counts[o] += 1

        stats = {
            "total_articles": len(all_ids),
            "classified": len(classified),
            "skipped": len(skipped),
            "pending": len(all_ids) - len(classified) - len(skipped),
            "relevance_breakdown": relevance_counts,
            "opinion_breakdown": opinion_counts,
        }
        return json.dumps(stats, indent=2)

    def get_classification_results(
        self,
        filter_relevant: str = "",
        filter_opinion: str = "",
    ) -> str:
        """Return classification results, optionally filtered.

        Args:
            filter_relevant: If set, only return entries matching this
                             is_relevant value ("yes", "no", "partially").
            filter_opinion:  If set, only return entries matching this
                             has_opinion value ("yes", "no").

        Returns:
            JSON object of matching classified entries.
        """
        self._invalidate_cache()
        data = self._load()
        results = data["classified"]

        if filter_relevant:
            fr = filter_relevant.strip().lower()
            results = {
                k: v for k, v in results.items()
                if v.get("is_relevant") == fr
            }

        if filter_opinion:
            fo = filter_opinion.strip().lower()
            results = {
                k: v for k, v in results.items()
                if v.get("has_opinion") == fo
            }

        return json.dumps(results, indent=2, ensure_ascii=False)


# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP(
    "Article-Classifier",
    instructions=(
        "This server feeds scraped articles one-at-a-time for classification. "
        "Call get_next_article to receive an article, then call "
        "submit_classification with your answers. Repeat until ALL_DONE."
    ),
)

# Global classifier instance
_classifier = ArticleClassifier()


@mcp.tool()
def get_next_article() -> str:
    """Get the next unclassified article for review.

    Returns a JSON object with:
        - article_id: unique identifier (use this in submit_classification)
        - title: article title
        - url: original URL
        - source_type: medium, substack, stackoverflow, corporate_blog, personal_blog
        - word_count: total word count of the full article
        - text: article text (capped at first 2,000 words)
        - remaining_after_this: how many articles are left after this one

    Returns the string "ALL_DONE" when all articles have been classified.
    """
    return _classifier.get_next_article()


@mcp.tool()
def submit_classification(
    article_id: str,
    is_relevant: str,
    has_opinion: str,
) -> str:
    """Submit your classification for an article.

    Call this after reading the article from get_next_article.

    Args:
        article_id: The article_id from get_next_article.
        is_relevant: Is this article about Vibe-Coding or AI-Assisted Coding?
                     Must be one of: "yes", "no", "partially".
        has_opinion: Does this article express an opinion about Vibe-Coding
                     or AI-Assisted Coding? Must be: "yes" or "no".

    Returns:
        Confirmation with updated progress counts.
    """
    return _classifier.submit_classification(article_id, is_relevant, has_opinion)


@mcp.tool()
def skip_article(article_id: str, reason: str = "") -> str:
    """Skip an article that cannot be meaningfully classified.

    Use this for articles that are empty, unreadable, or clearly garbage
    content that doesn't warrant a classification.

    Args:
        article_id: The article_id from get_next_article.
        reason: Optional reason for skipping.

    Returns:
        Confirmation with remaining article count.
    """
    return _classifier.skip_article(article_id, reason)


@mcp.tool()
def get_classification_stats() -> str:
    """Get progress statistics for the classification task.

    Returns a JSON object with:
        - total_articles: total number of scraped articles
        - classified: how many have been classified
        - skipped: how many have been skipped
        - pending: how many remain
        - relevance_breakdown: {"yes": N, "no": N, "partially": N}
        - opinion_breakdown: {"yes": N, "no": N}
    """
    return _classifier.get_classification_stats()


@mcp.tool()
def get_classification_results(
    filter_relevant: str = "",
    filter_opinion: str = "",
) -> str:
    """Retrieve classification results, optionally filtered.

    Args:
        filter_relevant: Filter by relevance value ("yes", "no", "partially").
                         Leave empty for all.
        filter_opinion:  Filter by opinion value ("yes", "no").
                         Leave empty for all.

    Returns:
        JSON object mapping article_id to classification data.
    """
    return _classifier.get_classification_results(filter_relevant, filter_opinion)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
