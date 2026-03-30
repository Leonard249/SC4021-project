"""
Stage 2: Content Extraction

Reads discovered URLs from Stage 1 and extracts article content using:
- trafilatura (primary) — best for blog/article extraction
- newspaper3k (fallback) — handles some edge cases trafilatura misses

Includes paywall detection for Medium and Substack.

Usage:
    python content_extractor.py

Input:
    discovered_urls/discovered_urls.csv

Output:
    raw_articles/raw_articles.csv
"""

import os
import sys
import csv
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seeds_config import DISCOVERED_URLS_DIR, RAW_ARTICLES_DIR
from utils import (
    setup_logger,
    load_urls_from_csv,
    save_articles_to_csv,
    ensure_dir,
)

logger = setup_logger("content_extractor", os.path.join(RAW_ARTICLES_DIR, "extraction.log"))


# =============================================================================
# Paywall Detection
# =============================================================================
PAYWALL_INDICATORS = [
    # Medium
    "Member-only story",
    "You've read all your free member-only stories",
    "Open in app",  # Medium paywall redirect
    # Substack
    "This post is for paid subscribers",
    "This post is for paying subscribers",
    "Subscribe to continue reading",
]


def is_paywalled(content: str, url: str) -> bool:
    """Check if extracted content suggests a paywall."""
    if not content:
        return True

    # Very short content likely means paywall or extraction failure
    if len(content.strip()) < 200:
        return True

    content_lower = content.lower()
    for indicator in PAYWALL_INDICATORS:
        if indicator.lower() in content_lower:
            return True

    return False


# =============================================================================
# Content Extraction
# =============================================================================
def extract_with_trafilatura(url: str) -> dict | None:
    """Extract article content using trafilatura."""
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        # Extract as JSON for structured metadata
        result = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            output_format="json",
            with_metadata=True,
        )

        if not result:
            return None

        import json
        data = json.loads(result)

        return {
            "title": data.get("title", ""),
            "author": data.get("author", ""),
            "date": data.get("date", ""),
            "content": data.get("text", ""),
            "description": data.get("description", ""),
            "sitename": data.get("sitename", ""),
        }

    except Exception as e:
        logger.debug(f"trafilatura failed for {url}: {e}")
        return None


def extract_with_newspaper(url: str) -> dict | None:
    """Extract article content using newspaper3k (fallback)."""
    try:
        from newspaper import Article

        article = Article(url)
        article.download()
        article.parse()

        if not article.text:
            return None

        return {
            "title": article.title or "",
            "author": ", ".join(article.authors) if article.authors else "",
            "date": article.publish_date.isoformat() if article.publish_date else "",
            "content": article.text,
            "description": article.meta_description or "",
            "sitename": "",
        }

    except Exception as e:
        logger.debug(f"newspaper3k failed for {url}: {e}")
        return None


def extract_article(url: str) -> dict | None:
    """
    Extract article content. Tries trafilatura first, then newspaper3k fallback.
    Returns None if both fail or content is paywalled.
    """
    # Try trafilatura first
    result = extract_with_trafilatura(url)

    # Fall back to newspaper3k if trafilatura fails
    if not result or not result.get("content"):
        result = extract_with_newspaper(url)

    # Check for paywall
    if result and is_paywalled(result.get("content", ""), url):
        logger.info(f"  ⚠ Paywalled or too short, skipping")
        return None

    return result


# =============================================================================
# Progress Tracking
# =============================================================================
def load_completed_urls(progress_file: str) -> set:
    """Load set of already-extracted URLs from progress file."""
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_url_done(progress_file: str, url: str):
    """Append a URL to the progress file."""
    with open(progress_file, "a") as f:
        f.write(url + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Read discovered URLs and extract article content."""
    start_time = time.time()

    # Setup
    ensure_dir(RAW_ARTICLES_DIR)
    input_file = os.path.join(DISCOVERED_URLS_DIR, "discovered_urls.csv")
    output_file = os.path.join(RAW_ARTICLES_DIR, "raw_articles.csv")
    progress_file = os.path.join(RAW_ARTICLES_DIR, "progress.txt")

    # Load discovered URLs
    discovered = load_urls_from_csv(input_file)
    if not discovered:
        logger.error(f"No URLs found in {input_file}. Run seed_discovery.py first.")
        return

    # Load progress (skip already-extracted URLs)
    completed = load_completed_urls(progress_file)
    pending = [u for u in discovered if u["url"] not in completed]

    logger.info("=" * 60)
    logger.info("CONTENT EXTRACTION — Starting")
    logger.info(f"  Total discovered URLs: {len(discovered)}")
    logger.info(f"  Already extracted: {len(completed)}")
    logger.info(f"  Pending: {len(pending)}")
    logger.info("=" * 60)

    # Load existing articles (to append to)
    existing_articles = []
    if os.path.exists(output_file):
        from utils import load_articles_from_csv
        existing_articles = load_articles_from_csv(output_file)

    articles = list(existing_articles)
    success_count = 0
    fail_count = 0
    paywall_count = 0

    for i, url_entry in enumerate(pending):
        url = url_entry["url"]
        source_type = url_entry.get("source_type", "unknown")

        logger.info(f"[{i+1}/{len(pending)}] Extracting: {url[:80]}...")

        result = extract_article(url)

        if result:
            word_count = len(result["content"].split())
            article = {
                "url": url,
                "source_type": source_type,
                "title": result["title"],
                "author": result["author"],
                "date": result["date"],
                "content": result["content"],
                "description": result["description"],
                "word_count": word_count,
                "crawled_at": datetime.now().isoformat(),
            }
            articles.append(article)
            success_count += 1
            logger.info(f"  ✓ {result['title'][:60]} ({word_count} words)")
        else:
            fail_count += 1

        # Mark URL as processed (whether success or fail)
        mark_url_done(progress_file, url)

        # Save periodically (every 50 articles) so we don't lose progress
        if (i + 1) % 50 == 0:
            save_articles_to_csv(articles, output_file)
            logger.info(f"  💾 Saved checkpoint ({len(articles)} articles so far)")

        # Small delay to be polite to servers
        time.sleep(0.5)

    # Final save
    save_articles_to_csv(articles, output_file)

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"  Successfully extracted: {success_count}")
    logger.info(f"  Failed/paywalled: {fail_count}")
    logger.info(f"  Total articles in dataset: {len(articles)}")
    logger.info(f"  Time elapsed: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"  Output saved to: {output_file}")

    # Breakdown by source type
    source_counts = {}
    for a in articles:
        st = a.get("source_type", "unknown")
        source_counts[st] = source_counts.get(st, 0) + 1
    logger.info("\n  Breakdown by source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {source}: {count}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
