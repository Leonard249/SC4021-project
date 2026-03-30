#!/usr/bin/env python3
"""
Article Content Scraper for the AI-Assisted Coding Dataset.

Reads URLs from merged_checkpoint.json and scrapes article content
using trafilatura (primary) with newspaper3k as fallback.

Features:
  - Per-domain concurrency locks: no two workers hit the same domain at once
  - Resumable: skips already-scraped articles on re-run
  - Tracks permanent failures in scrape_failures.json
  - Configurable concurrency, rate limiting, and timeouts
  - Rich progress reporting via tqdm

Usage:
    conda activate sc4021
    python scrape_articles.py                          # scrape all
    python scrape_articles.py --limit 10 --workers 3   # test run
    python scrape_articles.py --source-type medium      # only medium articles
"""

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura
import httpx
from curl_cffi import requests as cffi_requests
from tqdm import tqdm

# Optional newspaper3k fallback
try:
    from newspaper import Article as NewspaperArticle
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = _DIR / "merged_checkpoint.json"
DEFAULT_OUTPUT_DIR = _DIR / "scraped_articles"
FAILURES_FILE = "scrape_failures.json"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress noisy retry/SSL/download warnings from trafilatura and urllib3
for _noisy in ("urllib3", "trafilatura", "trafilatura.downloads",
               "trafilatura.utils", "trafilatura.core", "trafilatura.meta",
               "trafilatura.readability_lxml", "charset_normalizer"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# Graceful shutdown flag
_shutdown = threading.Event()


# ============================================================================
# Per-Domain Rate Limiter
# ============================================================================
class DomainRateLimiter:
    """
    Ensures:
      1. Only ONE request per domain at a time (per-domain lock).
      2. A minimum delay between consecutive requests to the same domain.
    
    This prevents hammering a single domain even with multiple workers.
    """

    def __init__(self, delay: float = 1.0):
        self._delay = delay
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._last_request: dict[str, float] = {}
        self._meta_lock = threading.Lock()  # protects _locks and _last_request

    def _get_lock(self, domain: str) -> threading.Lock:
        with self._meta_lock:
            return self._locks[domain]

    def acquire(self, domain: str):
        """Acquire exclusive access for a domain, respecting delay."""
        lock = self._get_lock(domain)
        lock.acquire()
        # Enforce minimum delay since last request to this domain
        with self._meta_lock:
            last = self._last_request.get(domain, 0)
        elapsed = time.time() - last
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)

    def release(self, domain: str):
        """Release domain lock and record timestamp."""
        with self._meta_lock:
            self._last_request[domain] = time.time()
        lock = self._get_lock(domain)
        lock.release()


# ============================================================================
# URL Utilities
# ============================================================================
def url_to_filename(url: str) -> str:
    """Generate a stable, unique filename from a URL using its SHA-256 hash."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{url_hash}.json"


def extract_domain(url: str) -> str:
    """Extract the effective domain from a URL for rate-limiting purposes."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    # Group subdomains: e.g. foo.medium.com → medium.com
    parts = hostname.split(".")
    if len(parts) > 2:
        # Keep last 2 parts for most domains
        # Special case for co.uk, com.au, etc. — not critical for rate limiting
        hostname = ".".join(parts[-2:])
    return hostname


# ============================================================================
# Scraping Functions
# ============================================================================
def fetch_html(url: str, timeout: int = 30) -> tuple[str | None, int, str]:
    """
    Fetch raw HTML using a multi-strategy approach:
      1. curl_cffi with Chrome TLS impersonation (bypasses Cloudflare)
      2. trafilatura's built-in downloader (good bot-evasion)
      3. httpx with HTTP/2 (fallback)
    
    Returns (html, status_code, error_message).
    """
    # Strategy 1: curl_cffi — impersonates real Chrome TLS fingerprint
    # This is the most effective against Cloudflare bot detection
    for verify in (True, False):
        try:
            resp = cffi_requests.get(
                url,
                impersonate="chrome131",
                timeout=timeout,
                allow_redirects=True,
                verify=verify,
            )
            if resp.status_code < 400:
                text = resp.text
                if text and len(text) > 200:
                    return text, resp.status_code, ""
            elif resp.status_code >= 400:
                # For 404s, don't try other strategies — the page genuinely doesn't exist
                if resp.status_code == 404:
                    return None, 404, "HTTP 404"
                # For 403, fall through to try other strategies
                if resp.status_code != 403:
                    return None, resp.status_code, f"HTTP {resp.status_code}"
            break  # Don't retry with verify=False if we got a valid HTTP response
        except Exception as e:
            err_str = str(e).lower()
            if "ssl" in err_str or "certificate" in err_str:
                if verify:
                    continue  # Retry with verify=False
            log.debug(f"curl_cffi failed for {url}: {e}")
            break

    # Strategy 2: trafilatura's downloader (handles many anti-bot measures)
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded and len(downloaded) > 200:
            return downloaded, 200, ""
    except Exception:
        pass

    # Strategy 3: httpx with HTTP/2
    try:
        with httpx.Client(
            http2=True,
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
        ) as client:
            resp = client.get(url)
            if resp.status_code < 400:
                return resp.text, resp.status_code, ""
            return None, resp.status_code, f"HTTP {resp.status_code}"
    except Exception as e:
        log.debug(f"httpx failed for {url}: {e}")
        return None, 0, str(e)[:100]


def scrape_with_trafilatura(html: str, url: str) -> dict | None:
    """
    Extract article content using trafilatura.
    Returns a dict with title, author, date, text, etc. or None on failure.
    """
    try:
        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            output_format="json",
            with_metadata=True,
        )
        if not result:
            return None
        data = json.loads(result)
        text = data.get("text", "").strip()
        if not text or len(text) < 50:
            return None  # Too short to be a real article
        return {
            "title": data.get("title", ""),
            "author": data.get("author", ""),
            "date": data.get("date", ""),
            "text": text,
            "scrape_method": "trafilatura",
        }
    except Exception as e:
        log.debug(f"Trafilatura error for {url}: {e}")
        return None


def scrape_with_newspaper(html: str, url: str) -> dict | None:
    """
    Fallback extraction using newspaper3k.
    Returns a dict with title, author, date, text, etc. or None on failure.
    """
    if not HAS_NEWSPAPER:
        return None
    try:
        article = NewspaperArticle(url)
        article.set_html(html)
        article.parse()
        text = (article.text or "").strip()
        if not text or len(text) < 50:
            return None
        return {
            "title": article.title or "",
            "author": ", ".join(article.authors) if article.authors else "",
            "date": (
                article.publish_date.strftime("%Y-%m-%d")
                if article.publish_date
                else ""
            ),
            "text": text,
            "scrape_method": "newspaper3k",
        }
    except Exception as e:
        log.debug(f"Newspaper error for {url}: {e}")
        return None


def scrape_single_article(
    url: str,
    metadata: dict,
    output_dir: Path,
    rate_limiter: DomainRateLimiter,
    timeout: int = 30,
    max_retries: int = 2,
) -> dict:
    """
    Scrape a single article. Returns a result dict with status info.
    
    Uses per-domain locking to prevent concurrent requests to the same domain.
    """
    domain = extract_domain(url)
    original_url = metadata.get("original_url", url)
    out_file = output_dir / url_to_filename(url)

    # Early exit if shutting down
    if _shutdown.is_set():
        return {"url": url, "status": "skipped", "reason": "shutdown"}

    # Skip if already scraped
    if out_file.exists():
        return {"url": url, "status": "skipped", "reason": "already_scraped"}

    last_error = ""
    for attempt in range(max_retries + 1):
        if attempt > 0:
            backoff = 2 ** attempt
            time.sleep(backoff)

        # Acquire per-domain lock (blocks if another worker is using this domain)
        rate_limiter.acquire(domain)
        try:
            html, status_code, fetch_err = fetch_html(original_url, timeout=timeout)
        finally:
            rate_limiter.release(domain)

        if html is None:
            last_error = fetch_err or "fetch_failed"
            # Don't retry on permanent HTTP errors (4xx = page doesn't exist)
            if 400 <= status_code < 500:
                log.debug(f"Permanent failure ({status_code}) for {url}")
                break
            continue

        # Try trafilatura first, then newspaper3k
        result = scrape_with_trafilatura(html, original_url)
        if result is None:
            result = scrape_with_newspaper(html, original_url)

        if result is not None:
            text = result.get("text") or ""
            if not text or len(text) < 50:
                last_error = "extraction_failed"
                continue
            title = result.get("title") or ""
            # Build output record
            article = {
                "url": original_url,
                "normalized_url": url,
                "source_type": metadata.get("source_type", ""),
                "title": title,
                "author": result.get("author") or "",
                "date": result.get("date") or "",
                "text": text,
                "excerpt": text[:500],
                "word_count": len(text.split()),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "scrape_method": result.get("scrape_method", "unknown"),
                "queries": metadata.get("queries", []),
            }
            # Write atomically
            tmp_file = out_file.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
            tmp_file.rename(out_file)
            return {"url": url, "status": "success", "title": title[:80]}

        last_error = "extraction_failed"

    return {"url": url, "status": "failed", "reason": last_error}


# ============================================================================
# Failure Tracking
# ============================================================================
def load_failures(output_dir: Path) -> dict:
    """Load the set of permanently failed URLs."""
    failures_path = output_dir / FAILURES_FILE
    if failures_path.exists():
        with open(failures_path, "r") as f:
            return json.load(f)
    return {}


def save_failures(output_dir: Path, failures: dict):
    """Save the failure records to disk."""
    failures_path = output_dir / FAILURES_FILE
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)


# ============================================================================
# Main Orchestrator
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Scrape article content from URLs in merged_checkpoint.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to merged checkpoint JSON (default: merged_checkpoint.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for scraped articles (default: scraped_articles/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers (default: 5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Minimum seconds between requests to the same domain (default: 1.0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max URLs to process (0 = all, useful for testing)",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="",
        help="Filter by source type (e.g. medium, substack, personal_blog)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries per URL on failure (default: 2)",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Skip URLs that previously failed (from scrape_failures.json)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load checkpoint ──
    log.info(f"Loading checkpoint from {args.checkpoint}")
    with open(args.checkpoint, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    urls_dict = checkpoint.get("urls", {})
    if not isinstance(urls_dict, dict):
        log.error("Expected urls to be a dict (merged_checkpoint format).")
        sys.exit(1)

    # ── Filter by source type if requested ──
    if args.source_type:
        urls_dict = {
            k: v for k, v in urls_dict.items()
            if v.get("source_type") == args.source_type
        }
        log.info(f"Filtered to {len(urls_dict)} URLs of type '{args.source_type}'")

    # ── Create output directory ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load previous failures if skip-failures enabled ──
    failures = load_failures(output_dir) if args.skip_failures else {}
    if failures:
        before = len(urls_dict)
        urls_dict = {k: v for k, v in urls_dict.items() if k not in failures}
        log.info(f"Skipping {before - len(urls_dict)} previously failed URLs")

    # ── Prepare URL list ──
    url_items = list(urls_dict.items())
    if args.limit > 0:
        url_items = url_items[:args.limit]

    # ── Skip already-scraped (quick file check) ──
    pending_items = []
    already_done = 0
    for norm_url, meta in url_items:
        out_file = output_dir / url_to_filename(norm_url)
        if out_file.exists():
            already_done += 1
        else:
            pending_items.append((norm_url, meta))

    total = len(url_items)
    log.info(
        f"Total: {total} URLs | Already scraped: {already_done} | "
        f"Pending: {len(pending_items)}"
    )

    if not pending_items:
        log.info("Nothing to scrape. All done!")
        return

    # ── Shuffle for domain diversity across workers ──
    # Sort by domain so we interleave different domains
    pending_items.sort(key=lambda x: extract_domain(x[0]))
    # Then interleave: pick one from each domain in round-robin
    by_domain: dict[str, list] = defaultdict(list)
    for item in pending_items:
        domain = extract_domain(item[0])
        by_domain[domain].append(item)
    
    interleaved = []
    domain_lists = list(by_domain.values())
    max_len = max(len(lst) for lst in domain_lists) if domain_lists else 0
    for i in range(max_len):
        for lst in domain_lists:
            if i < len(lst):
                interleaved.append(lst[i])
    pending_items = interleaved

    # ── Set up rate limiter and scrape ──
    rate_limiter = DomainRateLimiter(delay=args.delay)

    # Stats
    stats = {"success": 0, "failed": 0, "skipped": 0}
    new_failures = {}

    log.info(
        f"Starting scrape with {args.workers} workers, "
        f"{args.delay}s delay/domain, {args.timeout}s timeout"
    )

    # Show domain distribution
    domain_counts = defaultdict(int)
    for norm_url, _ in pending_items:
        domain_counts[extract_domain(norm_url)] += 1
    top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:10]
    log.info(f"Top domains: {', '.join(f'{d}({c})' for d, c in top_domains)}")

    # ── Signal handler for graceful Ctrl+C ──
    def _signal_handler(sig, frame):
        log.info("\n⏱  Ctrl+C received — finishing current articles, saving progress...")
        _shutdown.set()
    signal.signal(signal.SIGINT, _signal_handler)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for norm_url, meta in pending_items:
            if _shutdown.is_set():
                break
            future = executor.submit(
                scrape_single_article,
                norm_url,
                meta,
                output_dir,
                rate_limiter,
                args.timeout,
                args.max_retries,
            )
            futures[future] = norm_url

        with tqdm(total=len(pending_items), desc="Scraping", unit="article") as pbar:
            try:
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)
                    except Exception as e:
                        result = {"url": futures[future], "status": "failed", "reason": str(e)[:100]}
                    status = result["status"]
                    stats[status] = stats.get(status, 0) + 1

                    if status == "failed":
                        new_failures[result["url"]] = {
                            "reason": result.get("reason", "unknown"),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        }

                    pbar.set_postfix(
                        ok=stats["success"],
                        fail=stats["failed"],
                        skip=stats["skipped"],
                    )
                    pbar.update(1)

                    if _shutdown.is_set():
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
            except KeyboardInterrupt:
                log.info("Interrupted — saving progress...")
                _shutdown.set()
                for f in futures:
                    f.cancel()

    # ── Save failures ──
    if new_failures:
        all_failures = load_failures(output_dir)
        all_failures.update(new_failures)
        save_failures(output_dir, all_failures)

    # ── Final report ──
    print("\n" + "=" * 60)
    print("  SCRAPING COMPLETE")
    print("=" * 60)
    print(f"  Success:  {stats['success']}")
    print(f"  Failed:   {stats['failed']}")
    print(f"  Skipped:  {stats['skipped'] + already_done}")
    print(f"  Total:    {stats['success'] + stats['failed'] + stats['skipped'] + already_done}")
    print(f"\n  Output:   {output_dir}/")
    if new_failures:
        print(f"  Failures: {output_dir / FAILURES_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
