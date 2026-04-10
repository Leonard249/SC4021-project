"""
Monitoring dashboard for the headless data-cleaning pipeline.

Run:
  python dashboard/app.py

Then open:
  http://localhost:8090
"""
import json
import os
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))

BASE = os.path.dirname(HERE)
SELECTED_PATH = os.path.join(BASE, "selected_data.json")
PROCESSED_PATH = os.path.join(BASE, "processed_item.json")
INDEX_PATH = os.path.join(BASE, "index.json")
FAILED_PATH = os.path.join(BASE, "failed_item.json")
TIMING_LOGS_PATH = os.path.join(BASE, "timing_logs.json")

BUCKET_LIMITS = {
    "Irrelevant": 300,
    "Neutral": 300,
    "Positive": 300,
    "Negative": 300,
}
BUCKETS = list(BUCKET_LIMITS.keys())
GMT_PLUS_8 = timezone(timedelta(hours=8))


def load_json(path, default):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def ensure_selected_shape(data):
    if not isinstance(data, dict):
        data = {}
    normalized = dict(data)
    for bucket in BUCKETS:
        value = normalized.get(bucket, [])
        normalized[bucket] = value if isinstance(value, list) else []
    return normalized


def ensure_list(value):
    return value if isinstance(value, list) else []


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def make_preview(item, limit=240):
    text = safe_text(item.get("Text"))
    title = safe_text(item.get("Title"))
    basis = text or title
    if not basis and item.get("Comments"):
        first_comment = item["Comments"][0]
        basis = safe_text(first_comment.get("Text"))
    if len(basis) <= limit:
        return basis
    return basis[:limit].rstrip() + "..."


def count_index_remaining(index):
    if not isinstance(index, dict):
        return 0
    total = 0
    for pool in index.values():
        if isinstance(pool, list):
            total += len(pool)
    return total


def source_histogram(selected):
    counts = {}
    for bucket in BUCKETS:
        for item in selected.get(bucket, []):
            source = item.get("Source")
            if source:
                counts[source] = counts.get(source, 0) + 1
    return sorted(
        [{"source": source, "count": count} for source, count in counts.items()],
        key=lambda item: (-item["count"], item["source"]),
    )


def flatten_selected(selected):
    rows = []
    sequence = 0
    for bucket in BUCKETS:
        for item in selected.get(bucket, []):
            sequence += 1
            rows.append({
                "sequence": sequence,
                "bucket": bucket,
                "id": item.get("ID"),
                "source": item.get("Source"),
                "type": item.get("Type"),
                "title": safe_text(item.get("Title")),
                "preview": make_preview(item),
                "date": item.get("Date"),
                "classified_at": item.get("Classified_At"),
                "confidence": item.get("Classification_Confidence"),
                "reasoning": safe_text(item.get("Classification_Reasoning") or item.get("reasoning")),
            })
    rows.sort(
        key=lambda item: (
            safe_text(item.get("classified_at")),
            safe_text(item.get("date")),
            item["sequence"],
        ),
        reverse=True,
    )
    return rows


def flatten_failures(failures):
    rows = []
    for index, item in enumerate(failures, start=1):
        rows.append({
            "sequence": index,
            "id": item.get("ID"),
            "source": item.get("Source"),
            "type": item.get("Type"),
            "error": safe_text(item.get("Error")),
            "failed_at": item.get("Failed_At"),
        })
    rows.sort(
        key=lambda item: (safe_text(item.get("failed_at")), item["sequence"]),
        reverse=True,
    )
    return rows


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return round(values[low] * (1 - fraction) + values[high] * fraction, 3)


def average(values):
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def timing_metric(entry, name):
    timings = entry.get("timings_ms") or {}
    value = timings.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def llm_total_ms(entry):
    attempts = (entry.get("timings_ms") or {}).get("attempts") or []
    total = 0.0
    seen = False
    for attempt in attempts:
        value = attempt.get("llm_call_ms")
        if not isinstance(value, (int, float)):
            value = attempt.get("gemini_call_ms")
        if isinstance(value, (int, float)):
            total += float(value)
            seen = True
    return round(total, 3) if seen else None


def summarize_timings(logs):
    item_entries = [entry for entry in logs if entry.get("item_id")]
    run_summaries = [entry for entry in logs if entry.get("type") == "run_summary"]

    iteration_values = [
        value for value in (timing_metric(entry, "iteration_total_ms") for entry in item_entries)
        if value is not None
    ]
    classify_values = [
        value for value in (timing_metric(entry, "classify_total_ms") for entry in item_entries)
        if value is not None
    ]
    llm_values = [
        value for value in (llm_total_ms(entry) for entry in item_entries)
        if value is not None
    ]

    recent_entries = []
    for entry in reversed(item_entries[-30:]):
        recent_entries.append({
            "item_id": entry.get("item_id"),
            "source": entry.get("source"),
            "type": entry.get("type"),
            "status": entry.get("status"),
            "classification": entry.get("classification"),
            "accepted": entry.get("accepted"),
            "finished_at": entry.get("finished_at"),
            "iteration_total_ms": timing_metric(entry, "iteration_total_ms"),
            "classify_total_ms": timing_metric(entry, "classify_total_ms"),
            "choose_candidate_ms": timing_metric(entry, "choose_candidate_ms"),
            "prompt_build_ms": timing_metric(entry, "prompt_build_ms"),
            "llm_total_ms": llm_total_ms(entry),
            "persist_ms": timing_metric(entry, "persist_ms"),
            "attempt_count": len(((entry.get("timings_ms") or {}).get("attempts") or [])),
            "error": entry.get("error"),
        })

    status_counts = {}
    for entry in item_entries:
        status = entry.get("status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "entry_count": len(item_entries),
        "status_counts": status_counts,
        "avg_iteration_ms": average(iteration_values),
        "p50_iteration_ms": percentile(iteration_values, 0.50),
        "p90_iteration_ms": percentile(iteration_values, 0.90),
        "avg_classify_ms": average(classify_values),
        "avg_llm_ms": average(llm_values),
        "recent_entries": recent_entries,
        "recent_runs": list(reversed(run_summaries[-10:])),
    }


def estimate_completion(summary, timing_logs):
    remaining_selected = summary["selected_remaining_total"]
    if remaining_selected <= 0:
        now_gmt8 = datetime.now(GMT_PLUS_8)
        return {
            "available": True,
            "eta_iso": now_gmt8.isoformat(),
            "eta_display": now_gmt8.strftime("%d %b %Y, %I:%M:%S %p GMT+8"),
            "duration_display": "Complete",
            "basis": "All target buckets are already full.",
        }

    item_entries = [entry for entry in timing_logs if entry.get("item_id")]
    consumed_entries = [
        entry for entry in item_entries
        if entry.get("status") in {"accepted", "discarded", "failed_parse_error", "failed"}
    ]
    sample = consumed_entries[-200:]
    if not sample:
        return {
            "available": False,
            "eta_display": "Unavailable",
            "duration_display": "Need runtime data",
            "basis": "ETA appears after enough items have been processed.",
        }

    avg_iteration_ms = average([
        timing_metric(entry, "iteration_total_ms")
        for entry in sample
        if timing_metric(entry, "iteration_total_ms") is not None
    ])
    if avg_iteration_ms is None:
        return {
            "available": False,
            "eta_display": "Unavailable",
            "duration_display": "Need timing data",
            "basis": "Iteration timings have not been recorded yet.",
        }

    accepted_count = sum(1 for entry in sample if entry.get("accepted"))
    acceptance_rate = accepted_count / len(sample) if sample else 0.0
    if acceptance_rate <= 0:
        return {
            "available": False,
            "eta_display": "Unavailable",
            "duration_display": "No accepted items recently",
            "basis": "ETA needs at least some accepted items in the recent sample.",
        }

    estimated_processed_needed = remaining_selected / acceptance_rate
    estimated_ms = estimated_processed_needed * avg_iteration_ms
    eta_dt = datetime.now(GMT_PLUS_8) + timedelta(milliseconds=estimated_ms)
    eta_hours = estimated_ms / 1000 / 3600

    if eta_hours >= 24:
        duration_display = f"~{eta_hours / 24:.1f} days"
    elif eta_hours >= 1:
        duration_display = f"~{eta_hours:.1f} hours"
    else:
        duration_display = f"~{estimated_ms / 1000 / 60:.1f} minutes"

    return {
        "available": True,
        "eta_iso": eta_dt.isoformat(),
        "eta_display": eta_dt.strftime("%d %b %Y, %I:%M:%S %p GMT+8"),
        "duration_display": duration_display,
        "basis": (
            f"Based on the last {len(sample)} consumed items: "
            f"{acceptance_rate * 100:.1f}% accepted, "
            f"{avg_iteration_ms:.1f} ms average iteration."
        ),
    }


def build_dashboard_state():
    selected = ensure_selected_shape(load_json(SELECTED_PATH, {}))
    processed_ids = ensure_list(load_json(PROCESSED_PATH, []))
    index = load_json(INDEX_PATH, {})
    failures = ensure_list(load_json(FAILED_PATH, []))
    timing_logs = ensure_list(load_json(TIMING_LOGS_PATH, []))

    counts = {bucket: len(selected.get(bucket, [])) for bucket in BUCKETS}
    remaining = {
        bucket: max(BUCKET_LIMITS[bucket] - counts[bucket], 0)
        for bucket in BUCKETS
    }
    selected_total = sum(counts.values())
    selected_target_total = sum(BUCKET_LIMITS.values())
    selected_rows = flatten_selected(selected)
    failure_rows = flatten_failures(failures)
    summary = {
        "counts": counts,
        "limits": BUCKET_LIMITS,
        "remaining": remaining,
        "selected_total": selected_total,
        "selected_target_total": selected_target_total,
        "selected_remaining_total": max(selected_target_total - selected_total, 0),
        "processed_total": len(processed_ids),
        "failed_total": len(failure_rows),
        "index_remaining_total": count_index_remaining(index),
        "last_selected_at": selected_rows[0]["classified_at"] if selected_rows else None,
    }
    summary["eta_completion"] = estimate_completion(summary, timing_logs)

    return {
        "summary": summary,
        "source_distribution": source_histogram(selected)[:12],
        "recent_selected": selected_rows[:20],
        "recent_failures": failure_rows[:20],
        "timing": summarize_timings(timing_logs),
        "updated_at": datetime.now(GMT_PLUS_8).isoformat(),
    }


@app.route("/")
def index():
    return render_template("index.html", buckets=BUCKETS)


@app.route("/api/dashboard")
def dashboard_data():
    return jsonify(build_dashboard_state())


@app.route("/api/selected/<bucket>")
def selected_bucket(bucket):
    if bucket not in BUCKETS:
        return jsonify({"error": "Unknown bucket"}), 404

    page = request.args.get("page", default=1, type=int)
    page_size = request.args.get("page_size", default=5, type=int)
    page = max(page, 1)
    page_size = min(max(page_size, 1), 50)

    selected = ensure_selected_shape(load_json(SELECTED_PATH, {}))
    rows = [row for row in flatten_selected(selected) if row["bucket"] == bucket]
    total_items = len(rows)
    total_pages = max((total_items + page_size - 1) // page_size, 1)
    page = min(page, total_pages)
    start = (page - 1) * page_size
    end = start + page_size
    return jsonify({
        "bucket": bucket,
        "count": total_items,
        "limit": BUCKET_LIMITS[bucket],
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "items": rows[start:end],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8090, debug=False)
