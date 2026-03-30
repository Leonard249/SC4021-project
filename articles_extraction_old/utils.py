"""
Shared utilities for the blog crawling pipeline.
Provides logging, deduplication, checkpointing, and retry helpers.
"""

import csv
import hashlib
import json
import logging
import os
import time
import random
from datetime import datetime
from urllib.parse import urlparse


# =============================================================================
# Logging Setup
# =============================================================================
def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Create a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# URL Utilities
# =============================================================================
def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication (strip trailing slashes, fragments, etc.)."""
    parsed = urlparse(url)
    # Remove fragment and trailing slash
    path = parsed.path.rstrip("/")
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized.lower()


def url_hash(url: str) -> str:
    """Generate a short hash for a URL (for dedup tracking)."""
    return hashlib.md5(normalize_url(url).encode()).hexdigest()[:12]


def classify_source_type(url: str) -> str:
    """Classify a URL into a source type based on its domain."""
    domain = urlparse(url).netloc.lower()

    if "medium.com" in domain or "towardsdatascience.com" in domain:
        return "medium"
    elif "substack.com" in domain:
        return "substack"
    elif "stackoverflow.com" in domain or "stackexchange.com" in domain:
        return "stackoverflow"
    else:
        # Check against curated lists
        from seeds_config import CORPORATE_BLOG_SEEDS, PERSONAL_BLOG_SEEDS

        for seed in CORPORATE_BLOG_SEEDS:
            if seed["domain"].replace("/blog", "").replace("/blogs", "") in domain:
                return "corporate_blog"
        for seed in PERSONAL_BLOG_SEEDS:
            if seed["domain"] in domain:
                return "personal_blog"

        return "other_blog"


# =============================================================================
# CSV Helpers
# =============================================================================
def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_urls_to_csv(urls: list[dict], filepath: str):
    """Save a list of URL dicts to CSV. Creates parent dirs if needed."""
    ensure_dir(os.path.dirname(filepath))

    fieldnames = ["url", "source_type", "discovery_method", "query", "discovered_at"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(urls)


def load_urls_from_csv(filepath: str) -> list[dict]:
    """Load URL dicts from a CSV file."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_articles_to_csv(articles: list[dict], filepath: str):
    """Save extracted articles to CSV."""
    ensure_dir(os.path.dirname(filepath))
    if not articles:
        return

    fieldnames = list(articles[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)


def load_articles_from_csv(filepath: str) -> list[dict]:
    """Load articles from CSV."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =============================================================================
# Checkpointing
# =============================================================================
class Checkpoint:
    """Simple checkpoint system that saves progress after each query."""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return {"completed_queries": [], "urls": []}

    def save(self):
        ensure_dir(os.path.dirname(self.checkpoint_file))
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_query_done(self, query_id: str) -> bool:
        return query_id in self.data["completed_queries"]

    def mark_query_done(self, query_id: str, urls: list[dict]):
        self.data["completed_queries"].append(query_id)
        self.data["urls"].extend(urls)
        self.save()

    def get_all_urls(self) -> list[dict]:
        return self.data["urls"]

    def get_completed_count(self) -> int:
        return len(self.data["completed_queries"])


# =============================================================================
# Deduplication
# =============================================================================
def deduplicate_urls(url_list: list[dict]) -> list[dict]:
    """Remove duplicate URLs from a list of URL dicts."""
    seen = set()
    unique = []
    for item in url_list:
        h = url_hash(item["url"])
        if h not in seen:
            seen.add(h)
            unique.append(item)
    return unique


# =============================================================================
# Retry with Backoff
# =============================================================================
def random_delay(min_sec: float, max_sec: float):
    """Sleep for a random duration between min_sec and max_sec."""
    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)
    return delay
