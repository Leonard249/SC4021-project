"""
compare_schema.py
-----------------
Compares db_labelled.json (reference) and db_labeled_new.json (new output)
to verify their schemas are identical.

Checks:
  1. Top-level structure (list of posts)
  2. Post-level keys and value types
  3. Comment-level keys and value types
  4. Valid label values (across all posts and comments)
  5. Field-level type consistency (samples every record)
  6. Summary stats (post count, comment count, label distribution)
"""

import json
from pathlib import Path
from collections import Counter

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
REF_PATH    = BASE / "data" / "processed" / "db_labelled.json"
NEW_PATH    = Path(__file__).parent / "data" / "processed" / "db_labeled_new.json"

VALID_LABELS = {"irrelevant", "positive", "negative", "neutral"}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def type_name(value) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def infer_schema(record: dict) -> dict:
    """Return {field: type_name} for a dict."""
    return {k: type_name(v) for k, v in record.items()}


def schema_union(records: list[dict]) -> dict[str, set]:
    """
    Collect all observed types per field across a list of records.
    Returns {field: {type1, type2, ...}}.
    """
    union: dict[str, set] = {}
    for r in records:
        for k, v in r.items():
            union.setdefault(k, set()).add(type_name(v))
    return union


def compare_schema_union(name_a: str, schema_a: dict[str, set],
                          name_b: str, schema_b: dict[str, set]) -> list[str]:
    """Return list of mismatch descriptions between two schema unions."""
    issues = []
    all_keys = set(schema_a) | set(schema_b)
    for key in sorted(all_keys):
        in_a = key in schema_a
        in_b = key in schema_b
        if not in_a:
            issues.append(f"  ✗  Field '{key}' present in {name_b} but MISSING in {name_a}")
        elif not in_b:
            issues.append(f"  ✗  Field '{key}' present in {name_a} but MISSING in {name_b}")
        elif schema_a[key] != schema_b[key]:
            issues.append(
                f"  ✗  Field '{key}' type mismatch: "
                f"{name_a}={schema_a[key]}  vs  {name_b}={schema_b[key]}"
            )
    return issues


def load(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Reference : {REF_PATH}")
    print(f"New output: {NEW_PATH}")
    print()

    ref = load(REF_PATH)
    new = load(NEW_PATH)

    all_ok = True

    # ── 1. Top-level type ─────────────────────────────────────────────────────
    print("── [1] Top-level structure ─────────────────────────────────────")
    for label, data in [("REF", ref), ("NEW", new)]:
        t = type(data).__name__
        mark = "✓" if t == "list" else "✗"
        print(f"  {mark}  {label} is a {t}  (len={len(data)})")
        if t != "list":
            all_ok = False
    print()

    # ── 2. Post-level schema ──────────────────────────────────────────────────
    print("── [2] Post-level keys & types ─────────────────────────────────")
    ref_post_schema = schema_union(ref)
    new_post_schema = schema_union(new)

    # Show all fields with types from both
    all_post_keys = sorted(set(ref_post_schema) | set(new_post_schema))
    header = f"  {'Field':<15}  {'REF types':<20}  {'NEW types':<20}  Match"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in all_post_keys:
        ra = str(ref_post_schema.get(key, {"MISSING"}))
        nb = str(new_post_schema.get(key, {"MISSING"}))
        match = "✓" if ref_post_schema.get(key) == new_post_schema.get(key) else "✗"
        if match == "✗":
            all_ok = False
        print(f"  {match}  {key:<15}  {ra:<20}  {nb:<20}")
    print()

    # ── 3. Comment-level schema ───────────────────────────────────────────────
    print("── [3] Comment-level keys & types ──────────────────────────────")
    ref_comments = [c for post in ref for c in (post.get("Comments") or [])]
    new_comments = [c for post in new for c in (post.get("Comments") or [])]

    if not ref_comments and not new_comments:
        print("  (no comments in either file)")
    else:
        ref_cmt_schema = schema_union(ref_comments)
        new_cmt_schema = schema_union(new_comments)
        all_cmt_keys = sorted(set(ref_cmt_schema) | set(new_cmt_schema))
        header = f"  {'Field':<15}  {'REF types':<20}  {'NEW types':<20}  Match"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in all_cmt_keys:
            ra = str(ref_cmt_schema.get(key, {"MISSING"}))
            nb = str(new_cmt_schema.get(key, {"MISSING"}))
            match = "✓" if ref_cmt_schema.get(key) == new_cmt_schema.get(key) else "✗"
            if match == "✗":
                all_ok = False
            print(f"  {match}  {key:<15}  {ra:<20}  {nb:<20}")
    print()

    # ── 4. Label validation ───────────────────────────────────────────────────
    print("── [4] Label values ────────────────────────────────────────────")
    for label, data, comments in [
        ("REF", ref, ref_comments),
        ("NEW", new, new_comments),
    ]:
        post_labels   = {p.get("label") for p in data}
        cmt_labels    = {c.get("label") for c in comments}
        unknown_post  = post_labels - VALID_LABELS
        unknown_cmt   = cmt_labels  - VALID_LABELS
        print(f"  {label}  post labels : {sorted(post_labels)}")
        print(f"  {label}  comment labels: {sorted(cmt_labels)}")
        if unknown_post:
            print(f"  ✗  Unknown post labels: {unknown_post}")
            all_ok = False
        if unknown_cmt:
            print(f"  ✗  Unknown comment labels: {unknown_cmt}")
            all_ok = False
    print()

    # ── 5. Summary stats ──────────────────────────────────────────────────────
    print("── [5] Summary statistics ──────────────────────────────────────")
    for label, data, comments in [
        ("REF", ref, ref_comments),
        ("NEW", new, new_comments),
    ]:
        label_dist = Counter(p["label"] for p in data)
        print(f"  {label}  posts={len(data)},  comments={len(comments)},  "
              f"label dist={dict(sorted(label_dist.items()))}")
    print()

    # ── Result ────────────────────────────────────────────────────────────────
    if all_ok:
        print("✅ PASS — schemas are identical.")
    else:
        print("❌ FAIL — schema mismatches detected (see ✗ above).")


if __name__ == "__main__":
    main()
