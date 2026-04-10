#!/usr/bin/env python3
"""
Build reviewer-specific queue databases from the copied selected data file.

The source file remains untouched. Each output DB stores:
- the original LLM prediction under `llm`
- the human review state under `review`

Usage:
  python scripts/build_reviewer_dbs.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SOURCE_PATH = ROOT / "selected_data.json"
LEGACY_SOURCE_PATH = ROOT / "selected_data_check.json"
SOURCE_PATH = PRIMARY_SOURCE_PATH if PRIMARY_SOURCE_PATH.exists() else LEGACY_SOURCE_PATH

PROFILE_BUCKETS = [
    ("bryan", "Bryan", "Irrelevant"),
    ("ananya", "Ananya", "Neutral"),
    ("ryan", "Ryan", "Positive"),
    ("leonard", "Leonard", "Negative"),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text != "" else None


def load_existing_db(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def existing_item_map(db_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = db_payload.get("items")
    if not isinstance(items, list):
        return {}
    mapped = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("item_id")
        if isinstance(item_id, str) and item_id:
            mapped[item_id] = item
    return mapped


def existing_comment_map(item_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    comments = item_payload.get("comments")
    if not isinstance(comments, list):
        return {}
    mapped = {}
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        comment_id = comment.get("comment_id")
        if isinstance(comment_id, str) and comment_id:
            mapped[comment_id] = comment
    return mapped


def merge_review(default_review: dict[str, Any], existing_review: Any) -> dict[str, Any]:
    if not isinstance(existing_review, dict):
        return default_review
    merged = dict(default_review)
    for key in default_review:
        if key in existing_review:
            merged[key] = existing_review[key]
    return merged


def build_comment(
    comment: dict[str, Any],
    reviewer_id: str,
    existing_comment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    llm_label = comment.get("Classification")
    default_review = {
        "status": "pending",
        "decision": None,
        "final_label": None,
        "reviewed_by": reviewer_id,
        "reviewed_at": None,
        "notes": None,
    }

    return {
        "comment_id": comment.get("comment_id"),
        "parent_id": comment.get("parent_id"),
        "author": comment.get("Author"),
        "source": comment.get("Source"),
        "text": clean_text(comment.get("Text")),
        "score": comment.get("Score"),
        "date": comment.get("Date"),
        "word_count": comment.get("Word_Count"),
        "llm": {
            "label": llm_label,
            "confidence": comment.get("Classification_Confidence"),
            "reasoning": clean_text(comment.get("Classification_Reasoning")),
            "evidence": comment.get("Classification_Evidence"),
            "classified_at": comment.get("Classified_At"),
        },
        "review": merge_review(default_review, (existing_comment or {}).get("review")),
    }


def build_item(
    item: dict[str, Any],
    reviewer_id: str,
    reviewer_name: str,
    assigned_bucket: str,
    position: int,
    existing_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comments = item.get("Comments")
    comment_rows = []
    prior_comments = existing_comment_map(existing_item or {})
    if isinstance(comments, list):
        comment_rows = [
            build_comment(comment, reviewer_id, prior_comments.get(comment.get("comment_id")))
            for comment in comments
        ]

    default_review = {
        "status": "pending",
        "decision": None,
        "final_label": None,
        "reviewed_by": reviewer_id,
        "reviewed_at": None,
        "notes": None,
    }

    return {
        "item_id": item.get("ID"),
        "position": position,
        "assigned_bucket": assigned_bucket,
        "assigned_reviewer": {
            "id": reviewer_id,
            "name": reviewer_name,
        },
        "post": {
            "id": item.get("ID"),
            "source": item.get("Source"),
            "type": item.get("Type"),
            "author": item.get("Author"),
            "title": clean_text(item.get("Title")),
            "text": clean_text(item.get("Text")),
            "score": item.get("Score"),
            "date": item.get("Date"),
            "word_count": item.get("Word_Count"),
        },
        "llm": {
            "label": item.get("Classification"),
            "confidence": item.get("Classification_Confidence"),
            "reasoning": clean_text(item.get("Classification_Reasoning")),
            "evidence": item.get("Classification_Evidence"),
            "classified_at": item.get("Classified_At"),
        },
        "review": merge_review(default_review, (existing_item or {}).get("review")),
        "comments": comment_rows,
    }


def build_profile_db(
    reviewer_id: str,
    reviewer_name: str,
    assigned_bucket: str,
    items: list[dict[str, Any]],
    existing_db: dict[str, Any],
) -> dict[str, Any]:
    prior_items = existing_item_map(existing_db)
    return {
        "profile_id": reviewer_id,
        "display_name": reviewer_name,
        "assigned_bucket": assigned_bucket,
        "source_file": SOURCE_PATH.name,
        "item_count": len(items),
        "items": [
            build_item(
                item,
                reviewer_id,
                reviewer_name,
                assigned_bucket,
                position,
                prior_items.get(item.get("ID")),
            )
            for position, item in enumerate(items, start=1)
        ],
    }


def main() -> None:
    selected = load_json(SOURCE_PATH)

    profiles_payload = []
    for reviewer_id, reviewer_name, assigned_bucket in PROFILE_BUCKETS:
        items = selected.get(assigned_bucket, [])
        if not isinstance(items, list):
            raise ValueError(f"Bucket {assigned_bucket!r} is not a list.")

        output_path = ROOT / f"db_{reviewer_id}.json"
        existing_db = load_existing_db(output_path)
        db_payload = build_profile_db(
            reviewer_id,
            reviewer_name,
            assigned_bucket,
            items,
            existing_db,
        )
        save_json(output_path, db_payload)

        profiles_payload.append({
            "id": reviewer_id,
            "name": reviewer_name,
            "assigned_bucket": assigned_bucket,
            "db_file": output_path.name,
            "item_count": len(items),
        })

    save_json(ROOT / "profiles.json", {"profiles": profiles_payload})


if __name__ == "__main__":
    main()
