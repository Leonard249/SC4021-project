"""
Tool: get_next_data
Returns the next item for the LLM to classify, or null if done.

Usage: python get_next_data.py
Output: JSON item (with ID, Source, Type, Title, Text, ...) or null.

State files read/written:
  index.json          - pool of unprocessed IDs per source (shrinks over time)
  selected_data.json  - already selected items (used to compute source deficits)
  processed_item.json - list of all processed IDs
  in_review.json      - item currently being reviewed (for submit_classification to read)
"""
import json, random, os

BASE           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH       = os.path.join(BASE, "raw_data.json")
INDEX_PATH     = os.path.join(BASE, "index.json")
SELECTED_PATH  = os.path.join(BASE, "selected_data.json")
PROCESSED_PATH = os.path.join(BASE, "processed_item.json")
IN_REVIEW_PATH = os.path.join(BASE, "in_review.json")

BUCKET_LIMITS = {
    "Irrelevant": 300,
    "Neutral": 300,
    "Positive": 300,
    "Negative": 300,
}
BUCKETS = list(BUCKET_LIMITS.keys())
TARGET_TOTAL = sum(BUCKET_LIMITS.values())


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


def iter_selected_items(selected):
    for value in selected.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                yield item


def main():
    selected = normalize_selected(load_json(SELECTED_PATH, {}))
    index    = load_json(INDEX_PATH, {})

    # Stop if target reached
    total_selected = sum(len(selected.get(b, [])) for b in BUCKETS)
    if total_selected >= TARGET_TOTAL:
        print(json.dumps(None))
        return

    # Count selected items per source to compute deficits
    selected_per_source = {}
    for item in iter_selected_items(selected):
        src = item.get("Source")
        if src:
            selected_per_source[src] = selected_per_source.get(src, 0) + 1

    all_sources = list(index.keys())
    if not all_sources:
        print(json.dumps(None))
        return

    target_per_source = TARGET_TOTAL / len(all_sources)
    deficits = {
        src: target_per_source - selected_per_source.get(src, 0)
        for src in all_sources
    }
    sorted_sources = sorted(all_sources, key=lambda s: deficits[s], reverse=True)

    # Pick a random ID from the most underrepresented non-empty source
    chosen_id     = None
    chosen_source = None
    for source in sorted_sources:
        pool = index.get(source, [])
        if pool:
            idx        = random.randint(0, len(pool) - 1)
            chosen_id  = pool[idx]
            chosen_source = source
            pool.pop(idx)
            if not pool:
                del index[source]
            break

    if chosen_id is None:
        print(json.dumps(None))
        return

    # Persist updated index
    save_json(INDEX_PATH, index)

    # Record as processed
    processed = load_json(PROCESSED_PATH, [])
    processed.append(chosen_id)
    save_json(PROCESSED_PATH, processed)

    # Look up full item from raw_data
    with open(RAW_PATH) as f:
        raw = json.load(f)
    lookup = {item["ID"]: item for item in raw}
    chosen_item = lookup[chosen_id]

    # Store in in_review so submit_classification can access it
    in_review = load_json(IN_REVIEW_PATH, {})
    in_review[chosen_id] = chosen_item
    save_json(IN_REVIEW_PATH, in_review)

    print(json.dumps(chosen_item, ensure_ascii=False))


main()
