"""
Tool: submit_classification
Submits the LLM's classification for the current item.

Usage: python submit_classification.py --id <ID> --classification <Irrelevant|Neutral|Positive|Negative>
Output: JSON result with status (accepted/discarded) and current bucket counts.

State files read/written:
  in_review.json      - source of full item data (written by get_next_data)
  selected_data.json  - updated if item is accepted
"""
import json, argparse, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED_PATH  = os.path.join(BASE, "selected_data.json")
IN_REVIEW_PATH = os.path.join(BASE, "in_review.json")

BUCKET_LIMITS = {
    "Irrelevant": 300,
    "Neutral": 300,
    "Positive": 300,
    "Negative": 300,
}
BUCKETS = list(BUCKET_LIMITS.keys())


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_selected(data):
    if not isinstance(data, dict):
        data = {}
    normalized = dict(data)
    for bucket in BUCKETS:
        value = normalized.get(bucket, [])
        normalized[bucket] = value if isinstance(value, list) else []
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="ID of the item to classify")
    parser.add_argument(
        "--classification",
        required=True,
        choices=BUCKETS,
        help="Classification label",
    )
    args = parser.parse_args()

    in_review = load_json(IN_REVIEW_PATH, {})
    if args.id not in in_review:
        print(json.dumps({
            "status": "error",
            "reason": f"ID '{args.id}' not found in in_review.json. Call get_next_data first."
        }))
        sys.exit(1)

    item   = in_review[args.id]
    bucket = args.classification
    limit  = BUCKET_LIMITS[bucket]

    selected = normalize_selected(load_json(SELECTED_PATH, {}))

    if len(selected.get(bucket, [])) >= limit:
        # Bucket full — discard but still remove from in_review
        del in_review[args.id]
        save_json(IN_REVIEW_PATH, in_review)

        print(json.dumps({
            "status":  "discarded",
            "reason":  f"{bucket} bucket is full ({limit}/{limit})",
            "counts":  {b: len(selected.get(b, [])) for b in BUCKETS},
            "total":   sum(len(selected.get(b, [])) for b in BUCKETS),
        }))
        return

    # Accept: add to bucket
    selected.setdefault(bucket, []).append(item)
    save_json(SELECTED_PATH, selected)

    # Remove from in_review
    del in_review[args.id]
    save_json(IN_REVIEW_PATH, in_review)

    counts = {b: len(selected.get(b, [])) for b in BUCKETS}
    print(json.dumps({
        "status":       "accepted",
        "bucket":       bucket,
        "bucket_count": counts[bucket],
        "bucket_limit": limit,
        "counts":       counts,
        "total":        sum(counts.values()),
    }))


main()
