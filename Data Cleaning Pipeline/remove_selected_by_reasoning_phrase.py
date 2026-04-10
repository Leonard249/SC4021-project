#!/usr/bin/env python3
"""Remove selected_data entries whose Classification_Reasoning contains a phrase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
DEFAULT_PATH = BASE / "selected_data.json"
BUCKETS = ("Irrelevant", "Neutral", "Positive", "Negative")


def load_selected(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return data


def reasoning_matches(reasoning: object, phrase: str, ignore_case: bool) -> bool:
    text = str(reasoning or "")
    if ignore_case:
        return phrase.lower() in text.lower()
    return phrase in text


def remove_matching_entries(data: dict, phrase: str, ignore_case: bool):
    updated = {}
    removed = []

    for bucket, items in data.items():
        if not isinstance(items, list):
            updated[bucket] = items
            continue

        kept_items = []
        for item in items:
            if not isinstance(item, dict):
                kept_items.append(item)
                continue

            reasoning = item.get("Classification_Reasoning", "")
            if reasoning_matches(reasoning, phrase, ignore_case):
                removed.append({
                    "bucket": bucket,
                    "ID": item.get("ID"),
                    "Classification_Reasoning": reasoning,
                })
                continue

            kept_items.append(item)

        updated[bucket] = kept_items

    for bucket in BUCKETS:
        updated.setdefault(bucket, [])

    return updated, removed


def save_selected(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=str(DEFAULT_PATH),
        help="Path to selected_data.json.",
    )
    parser.add_argument(
        "--phrase",
        default="the comments",
        help="Phrase to match in Classification_Reasoning.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive matching.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without writing changes.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="How many removed entries to print as samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path).expanduser().resolve()
    data = load_selected(path)
    updated, removed = remove_matching_entries(
        data,
        phrase=args.phrase,
        ignore_case=not args.case_sensitive,
    )

    if not args.dry_run:
        save_selected(path, updated)

    print(f"path={path}")
    print(f"phrase={args.phrase!r}")
    print(f"removed_count={len(removed)}")
    print(f"mode={'dry_run' if args.dry_run else 'applied'}")

    for entry in removed[: max(args.show, 0)]:
        print(
            json.dumps(
                entry,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
