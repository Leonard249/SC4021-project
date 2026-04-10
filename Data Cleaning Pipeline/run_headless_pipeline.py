#!/usr/bin/env python3
"""
Run the data-cleaning classification loop with a fresh stateless LLM request per item.

Default bucket targets:
  Irrelevant: 300
  Neutral:    300
  Positive:   300
  Negative:   300

Usage examples:
  python run_headless_pipeline.py
  python run_headless_pipeline.py --max-items 5
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

from openai_headless import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT_PATH,
    create_stateless_completion,
)


BASE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE, "raw_data.json")
INDEX_PATH = os.path.join(BASE, "index.json")
SELECTED_PATH = os.path.join(BASE, "selected_data.json")
PROCESSED_PATH = os.path.join(BASE, "processed_item.json")
FAILED_PATH = os.path.join(BASE, "failed_item.json")
TIMING_LOGS_PATH = os.path.join(BASE, "timing_logs.json")

BUCKET_LIMITS = {
    "Irrelevant": 300,
    "Neutral": 300,
    "Positive": 300,
    "Negative": 300,
}
BUCKETS = list(BUCKET_LIMITS.keys())


def retry_backoff_seconds(attempt):
    return min(5 * attempt, 30)


def load_json(path, default):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


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


class PipelineStopError(RuntimeError):
    """Raised when the pipeline should stop immediately without consuming the item."""

    def __init__(self, message, timing_info=None):
        super().__init__(message)
        self.timing_info = timing_info or {}


class ParseOutputError(RuntimeError):
    """Raised when the model returned output that does not match the JSON contract."""

    def __init__(self, message, timing_info=None):
        super().__init__(message)
        self.timing_info = timing_info or {}


class SkipItemError(RuntimeError):
    """Raised when the current item should be skipped but the pipeline should continue."""

    def __init__(self, message, timing_info=None):
        super().__init__(message)
        self.timing_info = timing_info or {}


def selected_counts(selected):
    return {bucket: len(selected.get(bucket, [])) for bucket in BUCKETS}


def iter_selected_items(selected):
    for value in selected.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                yield item


def processed_set(processed_ids, selected):
    ids = {item_id for item_id in processed_ids if isinstance(item_id, str)}
    for item in iter_selected_items(selected):
        item_id = item.get("ID")
        if isinstance(item_id, str):
            ids.add(item_id)
    return ids


def selected_per_source(selected):
    counts = {}
    for item in iter_selected_items(selected):
        source = item.get("Source")
        if source:
            counts[source] = counts.get(source, 0) + 1
    return counts


def prune_index(index, seen_ids, raw_lookup):
    if not isinstance(index, dict):
        return {}, False

    pruned = {}
    changed = False
    for source, pool in index.items():
        if not isinstance(pool, list):
            changed = True
            continue
        new_pool = []
        for item_id in pool:
            if item_id in seen_ids or item_id not in raw_lookup:
                changed = True
                continue
            new_pool.append(item_id)
        if new_pool:
            pruned[source] = new_pool
        elif pool:
            changed = True
    return pruned, changed


def choose_next_candidate(index, selected):
    all_sources = [source for source, pool in index.items() if pool]
    if not all_sources:
        return None

    target_total = sum(BUCKET_LIMITS.values())
    target_per_source = target_total / len(all_sources)
    source_counts = selected_per_source(selected)
    deficits = {
        source: target_per_source - source_counts.get(source, 0)
        for source in all_sources
    }
    random.shuffle(all_sources)
    ranked_sources = sorted(all_sources, key=lambda source: deficits[source], reverse=True)

    for source in ranked_sources:
        pool = index[source]
        if not pool:
            continue
        pool_index = random.randrange(len(pool))
        return {
            "id": pool[pool_index],
            "source": source,
            "pool_index": pool_index,
        }
    return None


def prepare_item_for_prompt(item):
    return {
        "ID": item.get("ID"),
        "Source": item.get("Source"),
        "Type": item.get("Type"),
        "Author": item.get("Author"),
        "Title": item.get("Title"),
        "Text": item.get("Text"),
        "Score": item.get("Score"),
        "Date": item.get("Date"),
        "Word_Count": item.get("Word_Count"),
    }


def build_prompt(item):
    payload = prepare_item_for_prompt(item)
    return (
        "Classify the following item using the instructions from GEMINI.md.\n\n"
        "Return only a raw JSON object with this exact schema:\n"
        "{\n"
        '  "classification": "Irrelevant" | "Neutral" | "Positive" | "Negative",\n'
        '  "confidence": 0.0,\n'
        '  "reasoning": "short explanation",\n'
        '  "evidence": ["short supporting snippet or paraphrase"]\n'
        "}\n\n"
        "Rules:\n"
        "- No markdown fences.\n"
        "- No text outside the JSON object.\n"
        "- Use exactly one classification.\n"
        "- Base the classification on the item's Title and Text only.\n"
        "- Do not infer relevance or sentiment from comments or discussion around the item.\n"
        "- Reject non-English or mostly non-English items by classifying them as Irrelevant.\n"
        "- Do not translate non-English text to infer relevance or sentiment.\n"
        "- If the item is not meaningfully about vibe-coding or AI-assisted coding as a practice, classify it as Irrelevant.\n\n"
        "ITEM:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def extract_json_block(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Could not find a JSON object in model output.")


def parse_model_response(output_text):
    value = output_text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return extract_json_block(value)


def normalize_result(result):
    classification = result.get("classification")
    if classification not in BUCKETS:
        raise ValueError(f"Invalid classification: {classification!r}")

    confidence = result.get("confidence")
    if isinstance(confidence, str):
        confidence = float(confidence)
    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number.")
    confidence = max(0.0, min(1.0, float(confidence)))

    reasoning = str(result.get("reasoning", "")).strip()
    evidence = result.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list or string.")
    evidence = [str(item).strip() for item in evidence if str(item).strip()]

    return {
        "classification": classification,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence": evidence,
    }


def build_classification_timing(prompt_build_ms, attempts):
    return {
        "prompt_build_ms": round(prompt_build_ms, 3),
        "classify_total_ms": round(
            prompt_build_ms + sum(entry["attempt_total_ms"] for entry in attempts), 3
        ),
        "attempts": attempts,
    }


def run_llm(prompt, timeout_seconds, model, base_url, api_key, on_stream_event=None):
    return create_stateless_completion(
        prompt,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        system_prompt_path=DEFAULT_SYSTEM_PROMPT_PATH,
        on_stream_event=on_stream_event,
    )


def classify_item(item, max_retries, timeout_seconds, model, base_url, api_key):
    prompt_start = time.perf_counter()
    prompt = build_prompt(item)
    prompt_build_ms = (time.perf_counter() - prompt_start) * 1000
    attempts = []

    for attempt in range(1, max_retries + 1):
        attempt_started = time.perf_counter()
        try:
            llm_started = time.perf_counter()
            stream_state = {
                "started": False,
                "last_report": None,
            }

            def on_stream_event(event):
                event_type = event.get("type")
                now = time.perf_counter()
                if event_type == "stream_started":
                    stream_state["started"] = True
                    stream_state["last_report"] = now
                    print(f"  attempt {attempt}/{max_retries}: stream started", flush=True)
                    return
                if event_type == "delta":
                    last_report = stream_state["last_report"]
                    if last_report is None or now - last_report >= 10:
                        print(
                            f"  streaming... received_chars={event.get('total_chars', 0)}",
                            flush=True,
                        )
                        stream_state["last_report"] = now
                    return
                if event_type == "stream_finished" and stream_state["started"]:
                    print(
                        f"  stream finished. received_chars={event.get('total_chars', 0)}",
                        flush=True,
                    )

            try:
                print(f"  attempt {attempt}/{max_retries}: waiting for response stream...", flush=True)
                response_text = run_llm(
                    prompt,
                    timeout_seconds,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    on_stream_event=on_stream_event,
                )
            except TimeoutError as exc:
                attempts.append({
                    "attempt": attempt,
                    "status": "timeout",
                    "error": str(exc),
                    "attempt_total_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                if attempt == max_retries:
                    if stream_state["started"]:
                        raise SkipItemError(
                            f"LLM stream stalled for {timeout_seconds}s.",
                            timing_info=build_classification_timing(prompt_build_ms, attempts),
                        ) from None
                    raise SkipItemError(
                        f"LLM did not start streaming within {timeout_seconds}s.",
                        timing_info=build_classification_timing(prompt_build_ms, attempts),
                    ) from None
                time.sleep(retry_backoff_seconds(attempt))
                continue
            except Exception as exc:  # noqa: BLE001
                error_message = f"{type(exc).__name__}: {exc}"
                attempts.append({
                    "attempt": attempt,
                    "status": "llm_error",
                    "error": error_message,
                    "attempt_total_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                if attempt == max_retries:
                    raise PipelineStopError(
                        f"LLM request failed: {error_message}",
                        timing_info=build_classification_timing(prompt_build_ms, attempts),
                    ) from None
                time.sleep(retry_backoff_seconds(attempt))
                continue
            llm_call_ms = (time.perf_counter() - llm_started) * 1000

            try:
                response_started = time.perf_counter()
                parsed = parse_model_response(response_text)
                result = normalize_result(parsed)
                response_parse_ms = (time.perf_counter() - response_started) * 1000
            except Exception as exc:  # noqa: BLE001
                attempts.append({
                    "attempt": attempt,
                    "status": "parse_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "llm_call_ms": round(llm_call_ms, 3),
                    "gemini_call_ms": round(llm_call_ms, 3),
                    "attempt_total_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                if attempt == max_retries:
                    raise ParseOutputError(
                        f"Failed to parse model output for item {item.get('ID')} after {max_retries} attempts. "
                        f"Last error: {type(exc).__name__}: {exc}",
                        timing_info=build_classification_timing(prompt_build_ms, attempts),
                    ) from None
                continue

            result["raw_response"] = response_text
            result["attempt"] = attempt
            attempts.append({
                "attempt": attempt,
                "status": "ok",
                "llm_call_ms": round(llm_call_ms, 3),
                "gemini_call_ms": round(llm_call_ms, 3),
                "response_parse_ms": round(response_parse_ms, 3),
                "attempt_total_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
            })
            return result, build_classification_timing(prompt_build_ms, attempts)
        except PipelineStopError:
            raise

    raise ParseOutputError(
        f"Failed to parse model output for item {item.get('ID')} after {max_retries} attempts.",
        timing_info=build_classification_timing(prompt_build_ms, attempts),
    ) from None


def remove_candidate_from_index(index, candidate):
    source = candidate["source"]
    pool = index.get(source, [])
    pool_index = candidate["pool_index"]
    if 0 <= pool_index < len(pool) and pool[pool_index] == candidate["id"]:
        pool.pop(pool_index)
    else:
        try:
            pool.remove(candidate["id"])
        except ValueError:
            pass
    if not pool and source in index:
        del index[source]


def append_processed(processed_ids, item_id):
    if item_id not in processed_ids:
        processed_ids.append(item_id)


def build_saved_record(item, result):
    record = dict(item)
    record["Classification"] = result["classification"]
    record["Classification_Confidence"] = result["confidence"]
    record["Classification_Reasoning"] = result["reasoning"]
    record["Classification_Evidence"] = result["evidence"]
    record["Classified_At"] = datetime.now(timezone.utc).isoformat()
    return record


def save_failure(failures, item, error_message):
    failures.append({
        "ID": item.get("ID"),
        "Source": item.get("Source"),
        "Type": item.get("Type"),
        "Error": error_message,
        "Failed_At": datetime.now(timezone.utc).isoformat(),
    })


def print_progress(prefix, counts, processed_now, accepted=False, bucket=None):
    parts = [
        prefix,
        f"processed_this_run={processed_now}",
        f"counts={counts}",
    ]
    if bucket:
        parts.append(f"bucket={bucket}")
    if accepted:
        parts.append("status=accepted")
    print(" | ".join(parts), flush=True)


def append_timing_log(entry):
    logs = load_json(TIMING_LOGS_PATH, [])
    logs = ensure_list(logs)
    logs.append(entry)
    save_json(TIMING_LOGS_PATH, logs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Stop after processing this many items in the current run.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="How many total attempts to make for request or parse failures.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="Seconds to wait for stream activity before skipping the item.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model name for the OpenAI-compatible endpoint, for example "
            "'google/gemma-4-31b-it'."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="API key for the OpenAI-compatible endpoint.",
    )
    return parser.parse_args()


def main():
    run_started = time.perf_counter()
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")

    if not os.path.exists(DEFAULT_SYSTEM_PROMPT_PATH):
        print(f"Error: {DEFAULT_SYSTEM_PROMPT_PATH.name} was not found in {BASE}.", file=sys.stderr)
        sys.exit(1)
    if not args.model:
        print("Error: model is required. Pass --model or set OPENAI_MODEL.", file=sys.stderr)
        sys.exit(1)

    raw = load_json(RAW_PATH, [])
    if not isinstance(raw, list) or not raw:
        print("Error: raw_data.json is missing or invalid.", file=sys.stderr)
        sys.exit(1)
    raw_lookup = {
        item["ID"]: item
        for item in raw
        if isinstance(item, dict) and isinstance(item.get("ID"), str)
    }

    selected = ensure_selected_shape(load_json(SELECTED_PATH, {}))
    processed_ids = load_json(PROCESSED_PATH, [])
    if not isinstance(processed_ids, list):
        processed_ids = []
    failures = ensure_list(load_json(FAILED_PATH, []))
    if not os.path.exists(TIMING_LOGS_PATH) or os.path.getsize(TIMING_LOGS_PATH) == 0:
        save_json(TIMING_LOGS_PATH, [])

    index = load_json(INDEX_PATH, {})
    seen_ids = processed_set(processed_ids, selected)
    index, changed = prune_index(index, seen_ids, raw_lookup)
    if changed:
        save_json(INDEX_PATH, index)

    processed_this_run = 0
    target_total = sum(BUCKET_LIMITS.values())
    run_status = "finished"

    while True:
        counts = selected_counts(selected)
        if sum(counts.values()) >= target_total:
            print(f"Stopping: all target buckets are full. Final counts: {counts}")
            break

        if args.max_items is not None and processed_this_run >= args.max_items:
            print(f"Stopping: reached --max-items={args.max_items}. Current counts: {counts}")
            break

        iteration_started = time.perf_counter()
        iteration_started_at = datetime.now(timezone.utc).isoformat()
        selection_started = time.perf_counter()
        candidate = choose_next_candidate(index, selected)
        choose_candidate_ms = (time.perf_counter() - selection_started) * 1000
        if candidate is None:
            print(f"Stopping: no more unprocessed items in index.json. Final counts: {counts}")
            break

        lookup_started = time.perf_counter()
        item_id = candidate["id"]
        item = raw_lookup.get(item_id)
        lookup_item_ms = (time.perf_counter() - lookup_started) * 1000
        if item is None:
            remove_candidate_from_index(index, candidate)
            save_json(INDEX_PATH, index)
            continue

        print(
            f"Classifying {item_id} | source={item.get('Source')} | type={item.get('Type')}",
            flush=True,
        )

        status_label = "failed"
        timing_details = {
            "choose_candidate_ms": round(choose_candidate_ms, 3),
            "lookup_item_ms": round(lookup_item_ms, 3),
        }
        bucket = None
        accepted = False
        error_message = None
        counts_before = dict(counts)

        try:
            result, classify_timing = classify_item(
                item,
                args.max_retries,
                args.timeout_seconds,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
            )
            timing_details.update(classify_timing)
            bucket = result["classification"]
            accepted = counts[bucket] < BUCKET_LIMITS[bucket]

            persist_started = time.perf_counter()
            if accepted:
                selected.setdefault(bucket, []).append(build_saved_record(item, result))
                save_json(SELECTED_PATH, selected)

            remove_candidate_from_index(index, candidate)
            append_processed(processed_ids, item_id)
            save_json(INDEX_PATH, index)
            save_json(PROCESSED_PATH, processed_ids)
            timing_details["persist_ms"] = round((time.perf_counter() - persist_started) * 1000, 3)

            processed_this_run += 1
            counts = selected_counts(selected)
            status_label = "accepted" if accepted else "discarded"

            if accepted:
                print_progress("Accepted", counts, processed_this_run, accepted=True, bucket=bucket)
            else:
                print_progress("Discarded (bucket full)", counts, processed_this_run, bucket=bucket)
        except ParseOutputError as exc:
            error_message = str(exc)
            timing_details.update(exc.timing_info)
            persist_started = time.perf_counter()
            remove_candidate_from_index(index, candidate)
            append_processed(processed_ids, item_id)
            save_failure(failures, item, error_message)
            save_json(INDEX_PATH, index)
            save_json(PROCESSED_PATH, processed_ids)
            save_json(FAILED_PATH, failures)
            timing_details["persist_ms"] = round((time.perf_counter() - persist_started) * 1000, 3)
            processed_this_run += 1
            status_label = "failed_parse_error"
            print_progress("Failed", selected_counts(selected), processed_this_run)
            print(f"  reason: {exc}", flush=True)
            counts = selected_counts(selected)
        except SkipItemError as exc:
            error_message = str(exc)
            timing_details.update(exc.timing_info)
            persist_started = time.perf_counter()
            remove_candidate_from_index(index, candidate)
            append_processed(processed_ids, item_id)
            save_failure(failures, item, error_message)
            save_json(INDEX_PATH, index)
            save_json(PROCESSED_PATH, processed_ids)
            save_json(FAILED_PATH, failures)
            timing_details["persist_ms"] = round((time.perf_counter() - persist_started) * 1000, 3)
            processed_this_run += 1
            status_label = "skipped_timeout"
            print_progress("Skipped", selected_counts(selected), processed_this_run)
            print(f"  reason: {exc}", flush=True)
            counts = selected_counts(selected)
        except PipelineStopError as exc:
            run_status = "paused_error"
            error_message = str(exc)
            timing_details.update(exc.timing_info)
            timing_details["iteration_total_ms"] = round((time.perf_counter() - iteration_started) * 1000, 3)
            timing_entry = {
                "run_id": run_id,
                "sequence": processed_this_run + 1,
                "item_id": item_id,
                "source": item.get("Source"),
                "type": item.get("Type"),
                "status": "stopped_error",
                "classification": None,
                "accepted": False,
                "started_at": iteration_started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "counts_before": counts_before,
                "counts_after": dict(counts_before),
                "timings_ms": timing_details,
                "error": error_message,
                "model": args.model,
            }
            append_timing_log(timing_entry)
            print(f"Stopping immediately due to pipeline error: {exc}", flush=True)
            break
        except Exception as exc:  # noqa: BLE001
            run_status = "paused_error"
            error_message = str(exc)
            timing_details["iteration_total_ms"] = round((time.perf_counter() - iteration_started) * 1000, 3)
            timing_entry = {
                "run_id": run_id,
                "sequence": processed_this_run + 1,
                "item_id": item_id,
                "source": item.get("Source"),
                "type": item.get("Type"),
                "status": "stopped_unexpected_error",
                "classification": None,
                "accepted": False,
                "started_at": iteration_started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "counts_before": counts_before,
                "counts_after": dict(counts_before),
                "timings_ms": timing_details,
                "error": error_message,
                "model": args.model,
            }
            append_timing_log(timing_entry)
            print(f"Stopping immediately due to unexpected error: {exc}", flush=True)
            break

        timing_details["iteration_total_ms"] = round((time.perf_counter() - iteration_started) * 1000, 3)
        counts_after = dict(counts)
        timing_entry = {
            "run_id": run_id,
            "sequence": processed_this_run,
            "item_id": item_id,
            "source": item.get("Source"),
            "type": item.get("Type"),
            "status": status_label,
            "classification": bucket,
            "accepted": accepted,
            "started_at": iteration_started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "counts_before": counts_before,
            "counts_after": counts_after,
            "timings_ms": timing_details,
            "model": args.model,
        }
        if error_message:
            timing_entry["error"] = error_message
        append_timing_log(timing_entry)

    append_timing_log({
        "run_id": run_id,
        "type": "run_summary",
        "status": run_status,
        "processed_this_run": processed_this_run,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runtime_ms": round((time.perf_counter() - run_started) * 1000, 3),
        "final_counts": selected_counts(selected),
        "model": args.model,
    })


if __name__ == "__main__":
    main()
