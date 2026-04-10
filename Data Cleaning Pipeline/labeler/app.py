"""
Labeling interface for building sample.json.
Target: 10 Irrelevant, 10 Neutral, 10 Positive, 10 Negative.
Posts and their comments can each be labeled independently.

Run: conda run -n sc4021 python app.py
Then open http://localhost:8080
"""
import json, os, random
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(os.path.dirname(BASE), "data", "processed", "db_labelled.json")
SAMPLE_PATH = os.path.join(BASE, "sample.json")
STATE_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")

BUCKETS = ["Irrelevant", "Neutral", "Positive", "Negative"]
LIMITS  = {"Irrelevant": 6, "Neutral": 5, "Positive": 5, "Negative": 5}


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_db():
    return load_json(DB_PATH, [])


def load_sample():
    return load_json(SAMPLE_PATH, {b: [] for b in BUCKETS})


def load_state():
    return load_json(STATE_PATH, {"processed_ids": [], "rotation_index": 0})


def save_state(state):
    save_json(STATE_PATH, state)


def save_sample(sample):
    save_json(SAMPLE_PATH, sample)


def get_counts(sample):
    """Only count posts (not comments) toward bucket limits."""
    return {
        b: sum(1 for item in sample.get(b, []) if item.get("Type") != "Comment")
        for b in BUCKETS
    }


def next_target_bucket(counts, rotation_index):
    for i in range(len(BUCKETS)):
        idx = (rotation_index + i) % len(BUCKETS)
        bucket = BUCKETS[idx]
        if counts[bucket] < LIMITS[bucket]:
            return bucket, idx
    return None, rotation_index


def pick_item(db, processed_ids, target_bucket, sample):
    TARGET_TOTAL = sum(LIMITS.values())

    processed_set = set(processed_ids)
    target_label  = target_bucket.lower()

    selected_per_source = {}
    for bucket in BUCKETS:
        for item in sample.get(bucket, []):
            src = item["Source"]
            selected_per_source[src] = selected_per_source.get(src, 0) + 1

    all_sources      = list({item["Source"] for item in db})
    target_per_source = TARGET_TOTAL / len(all_sources)

    def source_deficit(src):
        return target_per_source - selected_per_source.get(src, 0)

    matching = [
        item for item in db
        if item["ID"] not in processed_set
        and item.get("label", "").lower() == target_label
    ]
    if matching:
        matching.sort(key=lambda x: source_deficit(x["Source"]), reverse=True)
        top_deficit = source_deficit(matching[0]["Source"])
        top_pool    = [x for x in matching if source_deficit(x["Source"]) == top_deficit]
        return random.choice(top_pool)

    unprocessed = [item for item in db if item["ID"] not in processed_set]
    if unprocessed:
        unprocessed.sort(key=lambda x: source_deficit(x["Source"]), reverse=True)
        top_deficit = source_deficit(unprocessed[0]["Source"])
        top_pool    = [x for x in unprocessed if source_deficit(x["Source"]) == top_deficit]
        return random.choice(top_pool)

    return None


def clean_item(item):
    """Strip original is_labeled/label fields."""
    entry = {k: v for k, v in item.items() if k not in ("is_labeled", "label")}
    if entry.get("Comments"):
        entry["Comments"] = [
            {k: v for k, v in c.items() if k not in ("is_labeled", "label")}
            for c in entry["Comments"]
        ]
    return entry


def build_comment_lookup(db):
    """{ comment_id -> (comment_dict, parent_post) }"""
    lookup = {}
    for post in db:
        for c in (post.get("Comments") or []):
            lookup[c["comment_id"]] = (c, post)
    return lookup


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    sample = load_sample()
    counts = get_counts(sample)
    return jsonify({"counts": counts, "limits": LIMITS,
                    "total": sum(counts.values()),
                    "done":  all(counts[b] >= LIMITS[b] for b in BUCKETS)})


@app.route("/next")
def next_item():
    db     = load_db()
    sample = load_sample()
    state  = load_state()
    counts = get_counts(sample)

    target_bucket, new_rotation = next_target_bucket(counts, state["rotation_index"])
    if target_bucket is None:
        return jsonify({"done": True})

    item = pick_item(db, state["processed_ids"], target_bucket, sample)
    if item is None:
        return jsonify({"done": True, "reason": "No more unprocessed items"})

    state["rotation_index"] = (new_rotation + 1) % len(BUCKETS)
    save_state(state)

    return jsonify({
        "done":      False,
        "item":      item,
        "suggested": target_bucket,
        "counts":    counts,
        "limits":    LIMITS,
    })


@app.route("/label", methods=["POST"])
def label_item():
    """
    Body: {
      post_id: str,                          # always required (to mark processed)
      entries: [                             # list of things to label
        { id, is_comment, label, reasoning }
      ]
    }
    """
    body    = request.get_json()
    post_id = body.get("post_id")
    entries = body.get("entries", [])

    db     = load_db()
    sample = load_sample()
    state  = load_state()

    # Mark post as processed so it never appears again
    if post_id and post_id not in state["processed_ids"]:
        state["processed_ids"].append(post_id)
        save_state(state)

    if not entries:
        return jsonify({"status": "skipped", "counts": get_counts(sample)})

    post_map    = {p["ID"]: p for p in db}
    comment_lkp = build_comment_lookup(db)
    results     = []

    for entry in entries:
        item_id    = entry.get("id")
        label      = entry.get("label")
        reasoning  = entry.get("reasoning", "").strip()
        is_comment = entry.get("is_comment", False)

        if label not in BUCKETS:
            results.append({"id": item_id, "status": "error", "reason": "Invalid label"})
            continue

        if not is_comment:
            counts = get_counts(sample)
            if counts[label] >= LIMITS[label]:
                results.append({"id": item_id, "status": "discarded",
                                "reason": f"{label} bucket is full"})
                continue

        if is_comment:
            if item_id not in comment_lkp:
                results.append({"id": item_id, "status": "error", "reason": "Comment not found"})
                continue
            c, parent = comment_lkp[item_id]
            record = {
                "ID":            c["comment_id"],
                "Source":        c.get("Source", parent["Source"]),
                "Type":          "Comment",
                "Author":        c.get("Author"),
                "Text":          c.get("Text"),
                "Score":         c.get("Score"),
                "Date":          c.get("Date"),
                "Word_Count":    c.get("Word_Count"),
                "parent_post_id": parent["ID"],
            }
        else:
            if item_id not in post_map:
                results.append({"id": item_id, "status": "error", "reason": "Post not found"})
                continue
            record = clean_item(post_map[item_id])

        if reasoning:
            record["reasoning"] = reasoning

        sample.setdefault(label, []).append(record)
        results.append({"id": item_id, "status": "accepted", "bucket": label})

    save_sample(sample)
    counts = get_counts(sample)
    return jsonify({"results": results, "counts": counts, "total": sum(counts.values())})


@app.route("/bucket/<name>")
def get_bucket(name):
    if name not in BUCKETS:
        return jsonify({"error": "Unknown bucket"}), 404
    sample = load_sample()
    return jsonify({"bucket": name, "items": sample.get(name, [])})


@app.route("/relabel", methods=["POST"])
def relabel_item():
    """
    Body: { id, from_bucket, action: "move"|"remove", to_bucket?, reasoning? }
    Moves or removes an item from a bucket. Also updates reasoning if provided.
    """
    body        = request.get_json()
    item_id     = body.get("id")
    from_bucket = body.get("from_bucket")
    action      = body.get("action")       # "move" | "remove"
    to_bucket   = body.get("to_bucket")
    reasoning   = body.get("reasoning")   # None = don't change, "" = clear it

    if from_bucket not in BUCKETS:
        return jsonify({"status": "error", "reason": "Invalid from_bucket"}), 400

    sample = load_sample()
    items  = sample.get(from_bucket, [])
    idx    = next((i for i, x in enumerate(items) if x["ID"] == item_id), None)

    if idx is None:
        return jsonify({"status": "error", "reason": "Item not found"}), 404

    item = items.pop(idx)
    sample[from_bucket] = items

    # Update reasoning if caller sent it
    if reasoning is not None:
        if reasoning.strip():
            item["reasoning"] = reasoning.strip()
        else:
            item.pop("reasoning", None)

    if action == "move" and to_bucket in BUCKETS:
        sample.setdefault(to_bucket, []).append(item)

    save_sample(sample)
    return jsonify({"status": "ok", "counts": get_counts(sample)})


@app.route("/labeled-comments")
def labeled_comments():
    """Returns { comment_id: { bucket, reasoning } } for all labeled comments."""
    sample = load_sample()
    result = {}
    for bucket in BUCKETS:
        for item in sample.get(bucket, []):
            if item.get("Type") == "Comment":
                result[item["ID"]] = {
                    "bucket":    bucket,
                    "reasoning": item.get("reasoning", ""),
                }
    return jsonify(result)


@app.route("/save-comment", methods=["POST"])
def save_comment():
    """
    Upserts a comment label. Removes from old bucket if already labeled.
    Body: { comment_id, parent_post_id, label, reasoning }
    """
    body          = request.get_json()
    comment_id    = body.get("comment_id")
    parent_post_id = body.get("parent_post_id")
    label         = body.get("label")
    reasoning     = body.get("reasoning", "").strip()

    if label not in BUCKETS:
        return jsonify({"status": "error", "reason": "Invalid label"}), 400

    db     = load_db()
    sample = load_sample()

    # Remove from whichever bucket it currently lives in
    for bucket in BUCKETS:
        sample[bucket] = [x for x in sample.get(bucket, []) if x["ID"] != comment_id]

    # Build comment record from db
    comment_lkp = build_comment_lookup(db)
    if comment_id not in comment_lkp:
        return jsonify({"status": "error", "reason": "Comment not found in db"}), 404

    c, parent = comment_lkp[comment_id]
    record = {
        "ID":             c["comment_id"],
        "Source":         c.get("Source", parent["Source"]),
        "Type":           "Comment",
        "Author":         c.get("Author"),
        "Text":           c.get("Text"),
        "Score":          c.get("Score"),
        "Date":           c.get("Date"),
        "Word_Count":     c.get("Word_Count"),
        "parent_post_id": parent["ID"],
    }
    if reasoning:
        record["reasoning"] = reasoning

    sample.setdefault(label, []).append(record)
    save_sample(sample)
    return jsonify({"status": "ok", "bucket": label, "counts": get_counts(sample)})


if __name__ == "__main__":
    app.run(debug=True, port=8080)
