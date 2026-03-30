#!/usr/bin/env python3
"""
Merge two checkpoint.json files into one, with deduplication.

File 1 (new format): articles_extraction/checkpoint.json
  - completed_queries: list of strings
  - urls: dict keyed by normalized URL, each value has:
      original_url, source_type, discovery_method, queries (list), discovered_at

File 2 (old format): articles_extraction_old/discovered_urls/checkpoint.json
  - completed_queries: list of strings
  - urls: list of dicts, each with:
      url, source_type, discovery_method, query (single string), discovered_at

Output: merged_checkpoint.json (new format — dict-based urls)
  Deduplication is based on normalized URL (lowercase, strip trailing slash and fragment).
"""

import json
import sys
from urllib.parse import urldefrag
from pathlib import Path
from datetime import datetime


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication: lowercase, strip fragment, strip trailing slash."""
    url = url.strip()
    url, _ = urldefrag(url)       # remove #fragment
    url = url.lower()
    url = url.rstrip("/")
    return url


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_checkpoints(new_path: str, old_path: str, output_path: str):
    new_data = load_json(new_path)
    old_data = load_json(old_path)

    # ── 1. Merge completed_queries (deduplicated, order-preserved) ──
    seen_queries = set()
    merged_queries = []
    for q in new_data.get("completed_queries", []) + old_data.get("completed_queries", []):
        q_lower = q.strip().lower()
        if q_lower not in seen_queries:
            seen_queries.add(q_lower)
            merged_queries.append(q)

    # ── 2. Merge URLs ──
    # Start from the new-format dict (already keyed by normalized URL)
    merged_urls: dict = {}

    # --- Add new-format URLs first ---
    new_urls = new_data.get("urls", {})
    if isinstance(new_urls, dict):
        for norm_key, entry in new_urls.items():
            # Re-normalize just to be safe
            norm = normalize_url(norm_key)
            if norm in merged_urls:
                # Merge queries
                existing = merged_urls[norm]
                existing_queries = set(existing.get("queries", []))
                for q in entry.get("queries", []):
                    if q not in existing_queries:
                        existing["queries"].append(q)
                        existing_queries.add(q)
                # Keep earliest discovered_at
                if entry.get("discovered_at", "") < existing.get("discovered_at", ""):
                    existing["discovered_at"] = entry["discovered_at"]
            else:
                merged_urls[norm] = {
                    "original_url": entry.get("original_url", norm_key),
                    "source_type": entry.get("source_type", "personal_blog"),
                    "discovery_method": entry.get("discovery_method", "unknown"),
                    "queries": list(entry.get("queries", [])),
                    "discovered_at": entry.get("discovered_at", ""),
                }
    elif isinstance(new_urls, list):
        # In case the new file also uses list format
        for entry in new_urls:
            raw_url = entry.get("url", entry.get("original_url", ""))
            norm = normalize_url(raw_url)
            query = entry.get("query", "")
            queries = entry.get("queries", [query] if query else [])
            if norm in merged_urls:
                existing = merged_urls[norm]
                existing_queries = set(existing.get("queries", []))
                for q in queries:
                    if q and q not in existing_queries:
                        existing["queries"].append(q)
                        existing_queries.add(q)
                if entry.get("discovered_at", "") < existing.get("discovered_at", ""):
                    existing["discovered_at"] = entry["discovered_at"]
            else:
                merged_urls[norm] = {
                    "original_url": raw_url,
                    "source_type": entry.get("source_type", "personal_blog"),
                    "discovery_method": entry.get("discovery_method", "unknown"),
                    "queries": [q for q in queries if q],
                    "discovered_at": entry.get("discovered_at", ""),
                }

    # --- Add old-format URLs (list of dicts) ---
    old_urls = old_data.get("urls", [])
    if isinstance(old_urls, list):
        for entry in old_urls:
            raw_url = entry.get("url", entry.get("original_url", ""))
            norm = normalize_url(raw_url)
            query = entry.get("query", "")
            queries = entry.get("queries", [query] if query else [])
            if norm in merged_urls:
                existing = merged_urls[norm]
                existing_queries = set(existing.get("queries", []))
                for q in queries:
                    if q and q not in existing_queries:
                        existing["queries"].append(q)
                        existing_queries.add(q)
                if entry.get("discovered_at", "") < existing.get("discovered_at", ""):
                    existing["discovered_at"] = entry["discovered_at"]
            else:
                merged_urls[norm] = {
                    "original_url": raw_url,
                    "source_type": entry.get("source_type", "personal_blog"),
                    "discovery_method": entry.get("discovery_method", "unknown"),
                    "queries": [q for q in queries if q],
                    "discovered_at": entry.get("discovered_at", ""),
                }
    elif isinstance(old_urls, dict):
        for norm_key, entry in old_urls.items():
            norm = normalize_url(norm_key)
            if norm in merged_urls:
                existing = merged_urls[norm]
                existing_queries = set(existing.get("queries", []))
                for q in entry.get("queries", []):
                    if q not in existing_queries:
                        existing["queries"].append(q)
                        existing_queries.add(q)
                if entry.get("discovered_at", "") < existing.get("discovered_at", ""):
                    existing["discovered_at"] = entry["discovered_at"]
            else:
                merged_urls[norm] = {
                    "original_url": entry.get("original_url", norm_key),
                    "source_type": entry.get("source_type", "personal_blog"),
                    "discovery_method": entry.get("discovery_method", "unknown"),
                    "queries": list(entry.get("queries", [])),
                    "discovered_at": entry.get("discovered_at", ""),
                }

    # ── 3. Build output ──
    merged = {
        "completed_queries": merged_queries,
        "urls": merged_urls,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # ── 4. Print stats ──
    new_url_count = len(new_urls) if isinstance(new_urls, dict) else len(new_urls)
    old_url_count = len(old_urls) if isinstance(old_urls, list) else len(old_urls)
    merged_url_count = len(merged_urls)
    duplicates_removed = (new_url_count + old_url_count) - merged_url_count

    # Count by source type
    source_counts = {}
    for entry in merged_urls.values():
        st = entry.get("source_type", "unknown")
        source_counts[st] = source_counts.get(st, 0) + 1

    print("=" * 60)
    print("  CHECKPOINT MERGE COMPLETE")
    print("=" * 60)
    print(f"  New file URLs:       {new_url_count}")
    print(f"  Old file URLs:       {old_url_count}")
    print(f"  Merged URLs:         {merged_url_count}")
    print(f"  Duplicates removed:  {duplicates_removed}")
    print()
    print(f"  New file queries:    {len(new_data.get('completed_queries', []))}")
    print(f"  Old file queries:    {len(old_data.get('completed_queries', []))}")
    print(f"  Merged queries:      {len(merged_queries)}")
    print()
    print("  URLs by source type:")
    for st in sorted(source_counts.keys()):
        print(f"    {st:20s}  {source_counts[st]}")
    print()
    print(f"  Output written to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    base = Path(__file__).parent

    new_path = base / "articles_extraction" / "checkpoint.json"
    old_path = base / "articles_extraction_old" / "discovered_urls" / "checkpoint.json"
    output_path = base / "merged_checkpoint.json"

    # Allow overriding output path via CLI arg
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])

    print(f"Merging:\n  NEW: {new_path}\n  OLD: {old_path}\n")
    merge_checkpoints(str(new_path), str(old_path), str(output_path))
