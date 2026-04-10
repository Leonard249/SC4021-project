"""
finalize_labels.py
------------------
Loads all db_*.json files, resolves each post (and its nested comments)
final label based on human review:

  - decision == "accept"  →  use llm.label  (human agreed with AI)
  - decision == "reject"  →  use review.final_label  (human override)

Output schema matches data/processed/db_labelled.json exactly:
  Post:    ID, Source, Type, Author, Title, Text, Score, Date, Word_Count,
           Comments, is_labeled, label
  Comment: comment_id, parent_id, Source, Author, Text, Score, Date,
           Word_Count, is_labeled, label

Posts are grouped into four output files in data/processed/:
  data_irrelevant.json
  data_positive.json
  data_negative.json
  data_neutral.json

Comments remain nested within their parent post. The "llm" and "review"
sections are removed from the output.
"""

import json
import glob
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def resolve_label(review: dict, llm: dict) -> str:
    """Return the authoritative label after human review (lowercased)."""
    if review.get("decision") == "reject":
        return review.get("final_label", "").strip().lower()
    return llm.get("label", "").strip().lower()


def build_comment(comment: dict) -> dict:
    """
    Convert a raw comment (with llm + review) into the clean output schema.
    Mirrors the comment shape in db_labelled.json.
    """
    llm    = comment.get("llm", {})
    review = comment.get("review", {})
    label  = resolve_label(review, llm)

    return {
        "comment_id": comment.get("comment_id"),
        "parent_id":  comment.get("parent_id"),
        "Source":     comment.get("source"),
        "Author":     comment.get("author"),
        "Text":       comment.get("text") or "",
        "Score":      comment.get("score") if comment.get("score") is not None else 0,
        "Date":       comment.get("date") or "",
        "Word_Count": comment.get("word_count") or 0,
        "is_labeled": True,
        "label":      label,
    }


def build_post(item: dict) -> dict:
    """
    Convert a raw item (post + nested comments, with llm + review) into the
    clean output schema. Mirrors the post shape in db_labelled.json.
    """
    post   = item["post"]
    llm    = item["llm"]
    review = item["review"]
    label  = resolve_label(review, llm)

    # Process each nested comment
    clean_comments = []
    for comment in item.get("comments", []):
        c_review = comment.get("review", {})
        if c_review.get("status") != "reviewed":
            print(f"  ⚠  Skipping unreviewed comment: {comment.get('comment_id')}")
            continue
        clean_comments.append(build_comment(comment))

    # Follow REF convention: null when no comments, list when there are some
    comments_field = clean_comments if clean_comments else None

    return {
        "ID":         post.get("id"),
        "Source":     post.get("source"),
        "Type":       post.get("type"),
        "Author":     post.get("author"),
        "Title":      post.get("title"),
        "Text":       post.get("text") or "",
        "Score":      post.get("score") if post.get("score") is not None else 0,
        "Date":       post.get("date") or "",
        "Word_Count": post.get("word_count") or 0,
        "Comments":   comments_field,
        "is_labeled": True,
        "label":      label,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Label (lowercased) → output filename
    LABEL_FILES = {
        "irrelevant": "data_irrelevant.json",
        "positive":   "data_positive.json",
        "negative":   "data_negative.json",
        "neutral":    "data_neutral.json",
    }

    buckets: dict[str, list] = {k: [] for k in LABEL_FILES}

    db_files = sorted(DATA_DIR.glob("db_*.json"))
    if not db_files:
        print("No db_*.json files found in", DATA_DIR)
        return

    total_posts = 0
    rejected_posts = 0
    rejected_comments = 0

    for db_path in db_files:
        print(f"Processing {db_path.name} …")
        with open(db_path, encoding="utf-8") as f:
            data = json.load(f)

        for item in data.get("items", []):
            review = item.get("review", {})

            # Safety: skip anything not fully reviewed
            if review.get("status") != "reviewed":
                print(f"  ⚠  Skipping unreviewed post: {item.get('item_id')}")
                continue

            post_record = build_post(item)
            label = post_record["label"]

            if label not in buckets:
                print(f"  ⚠  Unknown label '{label}' for post {post_record['ID']} – skipping")
                continue

            buckets[label].append(post_record)
            total_posts += 1

            if review.get("decision") == "reject":
                rejected_posts += 1

            # Count rejected comments for stats
            for comment in item.get("comments", []):
                if comment.get("review", {}).get("decision") == "reject":
                    rejected_comments += 1

    # ── Write output files ────────────────────────────────────────────────────
    print("\n── Output summary ──────────────────────────────────────────────")
    for label, filename in LABEL_FILES.items():
        records = buckets[label]
        out_path = OUTPUT_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        total_comments_in_bucket = sum(len(r["Comments"]) for r in records if r["Comments"])
        print(f"  {filename:<25}  {len(records):>4} posts,  {total_comments_in_bucket:>4} comments")

    total_comments_all = sum(
        len(r["Comments"]) for bucket in buckets.values() for r in bucket if r["Comments"]
    )
    print(f"\n  Total posts     : {total_posts}")
    print(f"  Total comments  : {total_comments_all}")
    print(f"  Posts rejected  : {rejected_posts}  ({rejected_posts/total_posts*100:.1f}%)")
    print(f"  Comments rejected: {rejected_comments}")
    print(f"\n✅ Output written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
