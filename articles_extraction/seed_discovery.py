"""
Stage 1: Seed URL Discovery

Discovers candidate URLs about vibe coding from:
1. Google Search (via googlesearch-python) — Medium, Substack, curated blogs, general web
2. DuckDuckGo Search — parallel supplement, no rate limits
3. Stack Exchange API — Stack Overflow questions and answers

Usage:
    python seed_discovery.py

Output:
    discovered_urls/discovered_urls.csv
"""

import os
import sys
import time
import random
import requests
from datetime import datetime
from urllib.parse import quote_plus

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seeds_config import (
    ALL_CURATED_DOMAINS,
    SEARCH_KEYWORDS,
    GOOGLE_RESULTS_PER_QUERY,
    GOOGLE_DELAY_MIN,
    GOOGLE_DELAY_MAX,
    GOOGLE_BACKOFF_SECONDS,
    GOOGLE_MAX_RETRIES,
    STACKOVERFLOW_TAGS,
    STACKOVERFLOW_KEYWORDS,
    DISCOVERED_URLS_DIR,
)
from utils import (
    setup_logger,
    classify_source_type,
    deduplicate_urls,
    save_urls_to_csv,
    Checkpoint,
    random_delay,
    ensure_dir,
)

logger = setup_logger("seed_discovery", os.path.join(DISCOVERED_URLS_DIR, "discovery.log"))


# =============================================================================
# Google Search Discovery
# =============================================================================
def google_search_urls(query: str, num_results: int = 20) -> list[str]:
    """
    Search Google and return a list of URLs.
    Uses googlesearch-python (scraping-based, no API key needed).
    """
    try:
        from googlesearch import search
        results = list(search(query, num_results=num_results, sleep_interval=0))
        return results
    except Exception as e:
        logger.warning(f"Google search failed for '{query}': {e}")
        return []


def discover_google(checkpoint: Checkpoint) -> list[dict]:
    """
    Run all Google searches: Medium, Substack, curated domains, and general web.
    Returns list of URL dicts.
    """
    all_urls = []
    queries = []

    # 1. Medium searches
    for keyword in SEARCH_KEYWORDS:
        queries.append({
            "query": f"site:medium.com {keyword}",
            "id": f"google_medium_{keyword}",
        })

    # 2. Substack searches
    for keyword in SEARCH_KEYWORDS:
        queries.append({
            "query": f"site:substack.com {keyword}",
            "id": f"google_substack_{keyword}",
        })

    # 3. Curated domain searches (1 broad query per domain)
    broad_keyword = '"vibe coding" OR "AI coding" OR "copilot" OR "cursor" OR "AI-assisted"'
    for domain in ALL_CURATED_DOMAINS:
        queries.append({
            "query": f"site:{domain} {broad_keyword}",
            "id": f"google_curated_{domain}",
        })

    # 4. General web searches (no site: filter)
    for keyword in SEARCH_KEYWORDS:
        queries.append({
            "query": keyword,
            "id": f"google_general_{keyword}",
        })

    total = len(queries)
    skipped = 0

    logger.info(f"=== Google Search: {total} queries, {GOOGLE_RESULTS_PER_QUERY} results each ===")

    for i, q in enumerate(queries):
        # Skip if already completed (checkpointing)
        if checkpoint.is_query_done(q["id"]):
            skipped += 1
            continue

        logger.info(f"[{i+1}/{total}] Searching: {q['query'][:80]}...")

        urls = []
        retries = 0
        while retries <= GOOGLE_MAX_RETRIES:
            try:
                raw_urls = google_search_urls(q["query"], GOOGLE_RESULTS_PER_QUERY)
                now = datetime.now().isoformat()
                urls = [
                    {
                        "url": url,
                        "source_type": classify_source_type(url),
                        "discovery_method": "google",
                        "query": q["query"],
                        "discovered_at": now,
                    }
                    for url in raw_urls
                ]
                logger.info(f"  → Found {len(urls)} URLs")
                break
            except Exception as e:
                retries += 1
                if retries <= GOOGLE_MAX_RETRIES:
                    logger.warning(f"  ⚠ Error (retry {retries}/{GOOGLE_MAX_RETRIES}): {e}")
                    logger.info(f"  ⏳ Backing off for {GOOGLE_BACKOFF_SECONDS}s...")
                    time.sleep(GOOGLE_BACKOFF_SECONDS)
                else:
                    logger.error(f"  ✗ Failed after {GOOGLE_MAX_RETRIES} retries: {e}")

        # Save checkpoint
        checkpoint.mark_query_done(q["id"], urls)
        all_urls.extend(urls)

        # Random delay between queries
        delay = random_delay(GOOGLE_DELAY_MIN, GOOGLE_DELAY_MAX)
        logger.info(f"  💤 Waiting {delay:.1f}s before next query...")

    if skipped > 0:
        logger.info(f"  ⏭ Skipped {skipped} already-completed queries (checkpoint)")

    return all_urls


# =============================================================================
# DuckDuckGo Search Discovery
# =============================================================================
def discover_duckduckgo(checkpoint: Checkpoint) -> list[dict]:
    """
    Run searches via DuckDuckGo (no rate limits).
    Supplements Google results with different ranking.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed. Skipping DuckDuckGo discovery.")
        return []

    all_urls = []
    queries = []

    # Same keyword queries, but DuckDuckGo doesn't support site: reliably,
    # so we use general searches + platform-specific terms
    for keyword in SEARCH_KEYWORDS:
        queries.append({
            "query": keyword,
            "id": f"ddg_general_{keyword}",
        })

    # Add a few targeted ones
    for keyword in SEARCH_KEYWORDS[:4]:  # Top keywords only to limit volume
        queries.append({
            "query": f"{keyword} medium.com OR substack.com blog",
            "id": f"ddg_platforms_{keyword}",
        })

    total = len(queries)
    skipped = 0

    logger.info(f"\n=== DuckDuckGo Search: {total} queries ===")

    for i, q in enumerate(queries):
        if checkpoint.is_query_done(q["id"]):
            skipped += 1
            continue

        logger.info(f"[{i+1}/{total}] Searching: {q['query'][:80]}...")

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(q["query"], max_results=30))
                now = datetime.now().isoformat()
                urls = [
                    {
                        "url": r["href"],
                        "source_type": classify_source_type(r["href"]),
                        "discovery_method": "duckduckgo",
                        "query": q["query"],
                        "discovered_at": now,
                    }
                    for r in results
                    if "href" in r
                ]
                logger.info(f"  → Found {len(urls)} URLs")
                checkpoint.mark_query_done(q["id"], urls)
                all_urls.extend(urls)
        except Exception as e:
            logger.warning(f"  ⚠ DuckDuckGo error: {e}")
            checkpoint.mark_query_done(q["id"], [])

        # Small delay to be polite (DDG is lenient but let's not abuse it)
        time.sleep(1)

    if skipped > 0:
        logger.info(f"  ⏭ Skipped {skipped} already-completed queries (checkpoint)")

    return all_urls


# =============================================================================
# Stack Exchange API Discovery
# =============================================================================
def discover_stackoverflow(checkpoint: Checkpoint) -> list[dict]:
    """
    Search Stack Overflow using the Stack Exchange API.
    Fetches questions and their top answers.
    """
    all_urls = []
    base_url = "https://api.stackexchange.com/2.3"

    logger.info(f"\n=== Stack Overflow API Search ===")

    # Search by tags
    for tag in STACKOVERFLOW_TAGS:
        query_id = f"so_tag_{tag}"
        if checkpoint.is_query_done(query_id):
            continue

        logger.info(f"Searching tag: [{tag}]...")
        try:
            params = {
                "order": "desc",
                "sort": "relevance",
                "tagged": tag,
                "site": "stackoverflow",
                "pagesize": 100,
                "filter": "default",
            }
            response = requests.get(f"{base_url}/questions", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            now = datetime.now().isoformat()
            urls = [
                {
                    "url": item["link"],
                    "source_type": "stackoverflow",
                    "discovery_method": "stackexchange_api",
                    "query": f"tag:{tag}",
                    "discovered_at": now,
                }
                for item in data.get("items", [])
            ]
            logger.info(f"  → Found {len(urls)} questions")
            checkpoint.mark_query_done(query_id, urls)
            all_urls.extend(urls)

        except Exception as e:
            logger.warning(f"  ⚠ Stack Exchange API error for tag [{tag}]: {e}")
            checkpoint.mark_query_done(query_id, [])

    # Search by keywords
    for keyword in STACKOVERFLOW_KEYWORDS:
        query_id = f"so_keyword_{keyword}"
        if checkpoint.is_query_done(query_id):
            continue

        logger.info(f"Searching keyword: '{keyword}'...")
        try:
            params = {
                "order": "desc",
                "sort": "relevance",
                "intitle": keyword,
                "site": "stackoverflow",
                "pagesize": 100,
                "filter": "default",
            }
            response = requests.get(f"{base_url}/search", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            now = datetime.now().isoformat()
            urls = [
                {
                    "url": item["link"],
                    "source_type": "stackoverflow",
                    "discovery_method": "stackexchange_api",
                    "query": f"keyword:{keyword}",
                    "discovered_at": now,
                }
                for item in data.get("items", [])
            ]
            logger.info(f"  → Found {len(urls)} questions")
            checkpoint.mark_query_done(query_id, urls)
            all_urls.extend(urls)

        except Exception as e:
            logger.warning(f"  ⚠ Stack Exchange API error for '{keyword}': {e}")
            checkpoint.mark_query_done(query_id, [])

        time.sleep(0.5)  # Polite delay for API

    return all_urls


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Run all discovery methods and produce discovered_urls.csv."""
    start_time = time.time()

    # Setup
    ensure_dir(DISCOVERED_URLS_DIR)
    checkpoint = Checkpoint(os.path.join(DISCOVERED_URLS_DIR, "checkpoint.json"))

    logger.info("=" * 60)
    logger.info("SEED URL DISCOVERY — Starting")
    logger.info(f"Checkpoint: {checkpoint.get_completed_count()} queries already done")
    logger.info("=" * 60)

    # Run all discovery methods
    google_urls = discover_google(checkpoint)
    ddg_urls = discover_duckduckgo(checkpoint)
    so_urls = discover_stackoverflow(checkpoint)

    # Combine all discovered URLs (including previously checkpointed ones)
    all_urls = checkpoint.get_all_urls()

    # Deduplicate
    unique_urls = deduplicate_urls(all_urls)

    # Save final output
    output_file = os.path.join(DISCOVERED_URLS_DIR, "discovered_urls.csv")
    save_urls_to_csv(unique_urls, output_file)

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("DISCOVERY COMPLETE")
    logger.info(f"  Total URLs found: {len(all_urls)}")
    logger.info(f"  Unique URLs after dedup: {len(unique_urls)}")
    logger.info(f"  Time elapsed: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"  Output saved to: {output_file}")

    # Breakdown by source type
    source_counts = {}
    for u in unique_urls:
        st = u.get("source_type", "unknown")
        source_counts[st] = source_counts.get(st, 0) + 1
    logger.info("\n  Breakdown by source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {source}: {count}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
