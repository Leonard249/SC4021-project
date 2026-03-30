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
# Direct Web Search Discovery (Bypassing Blocked APIs)
# =============================================================================
def search_serper_api(query: str, max_results: int = 30) -> list[str]:
    """
    Search Google via Serper.dev API.
    This bypasses all bot checks and returns highly accurate Google JSON results.
    """
    from seeds_config import SERPER_API_KEY
    import requests
    import json
    
    if not SERPER_API_KEY:
        logger.error("  ❌ SERPER_API_KEY is missing! Run: export SERPER_API_KEY='your-key'")
        return []
        
    url = "https://google.serper.dev/search"
    
    # max_results for Serper is typically controlled by 'num' parameter (max 100)
    # We ask for the exact number we want to save credits
    payload = json.dumps({
      "q": query,
      "num": max_results
    })
    headers = {
      'X-API-KEY': SERPER_API_KEY,
      'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Serper returns "organic" for standard web results
        organic_results = data.get("organic", [])
        return [r["link"] for r in organic_results if "link" in r]
        
    except Exception as e:
        logger.warning(f"  ⚠ Serper request failed for '{query}': {e}")
        return []


def discover_web(checkpoint: Checkpoint) -> list[dict]:
    """
    Runs search queries via Serper API (Google Search) for high reliability.
    """
    from seeds_config import SERPER_API_KEY
    
    if not SERPER_API_KEY:
        logger.error("FATAL: SERPER_API_KEY environment variable is not set.")
        logger.error("Please sign up at https://serper.dev, get a free key, and export it:")
        logger.error("export SERPER_API_KEY='your_api_key_here'")
        # We return an empty list here to gracefully skip the web discovery
        # but the error is logged very clearly.
        return []

    all_urls = []
    queries = []

    # 1. Medium searches
    for keyword in SEARCH_KEYWORDS:
        queries.append({
            "query": f"medium.com {keyword}".replace('"', ''),
            "id": f"web_medium_{keyword}",
        })

    # 2. Substack searches
    for keyword in SEARCH_KEYWORDS:
        queries.append({
            "query": f"substack.com {keyword}".replace('"', ''),
            "id": f"web_substack_{keyword}",
        })

    # 3. Curated domain searches
    broad_keyword = '"vibe coding" OR "AI" OR "copilot" OR "cursor"'
    for domain in ALL_CURATED_DOMAINS:
        # Avoid using 'site:' operator on Serper free tier
        queries.append({
            "query": f"{domain} {broad_keyword}".replace('"', ''),
            "id": f"web_curated_{domain}",
        })

    # 4. General web searches
    for keyword in SEARCH_KEYWORDS:
        queries.append({
            "query": keyword.replace('"', ''),
            "id": f"web_general_{keyword}",
        })

    total = len(queries)
    skipped = 0

    logger.info(f"=== Web Search (Serper API): {total} queries ===")

    for i, q in enumerate(queries):
        if checkpoint.is_query_done(q["id"]):
            skipped += 1
            continue

        logger.info(f"[{i+1}/{total}] Searching: {q['query'][:80]}...")

        raw_urls = search_serper_api(q["query"])
        now = datetime.now().isoformat()
        
        urls = [
            {
                "url": url,
                "source_type": classify_source_type(url),
                "discovery_method": "serper_api",
                "query": q["query"],
                "discovered_at": now,
            }
            for url in raw_urls
            if url.startswith("http")
        ]
        
        logger.info(f"  → Found {len(urls)} URLs")
        checkpoint.mark_query_done(q["id"], urls)
        all_urls.extend(urls)

        # Serper API is fast and doesn't explicitly require human-level delays between calls on paid plans,
        # but to be conservative with rate limits on a free tier, we add a very short delay
        delay = random_delay(0.5, 1.5)
        logger.info(f"  💤 Waiting {delay:.1f}s...")

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
    web_urls = discover_web(checkpoint)
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
