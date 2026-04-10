#!/usr/bin/env python3
"""
Run stateless comment classification over comments nested inside selected_data.json.

Each comment is classified independently with a fresh LLM request. The prompt
contains the target comment plus its available parent chain up to the post.

Usage examples:
  python run_comment_headless_pipeline.py
  python run_comment_headless_pipeline.py --max-comments 10
  python run_comment_headless_pipeline.py --retry-failed
"""
import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

from openai_headless import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    create_stateless_completion,
)


BASE = os.path.dirname(os.path.abspath(__file__))
SELECTED_PATH = os.path.join(BASE, "selected_data.json")
FAILED_PATH = os.path.join(BASE, "comment_failed_item.json")
TIMING_LOGS_PATH = os.path.join(BASE, "comment_timing_logs.json")
DEFAULT_SYSTEM_PROMPT_PATH = os.path.join(BASE, "GEMINI_comments.md")

BUCKETS = ["Irrelevant", "Neutral", "Positive", "Negative"]


def retry_backoff_seconds(attempt):
    return min(5 * attempt, 30)


def load_json(path, default):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


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


def build_indexes(selected):
    posts_by_id = {}
    comments_by_id = {}
    work_items = []

    for bucket in BUCKETS:
        for post_index, post in enumerate(selected.get(bucket, [])):
            if not isinstance(post, dict):
                raise ValueError(f"Bucket {bucket!r} contains a non-object post entry.")

            post_id = post.get("ID")
            if not isinstance(post_id, str) or not post_id:
                raise ValueError(f"Bucket {bucket!r} contains a post with no valid ID.")
            if post_id in posts_by_id:
                raise ValueError(f"Duplicate post ID found in selected_data.json: {post_id}")

            posts_by_id[post_id] = {
                "bucket": bucket,
                "post_index": post_index,
                "post": post,
            }

            comments = post.get("Comments")
            if comments is None:
                continue
            if not isinstance(comments, list):
                raise ValueError(f"Post {post_id} has a non-list Comments field.")

            for comment_index, comment in enumerate(comments):
                if not isinstance(comment, dict):
                    raise ValueError(f"Post {post_id} has a non-object comment entry.")

                comment_id = comment.get("comment_id")
                if not isinstance(comment_id, str) or not comment_id:
                    raise ValueError(f"Post {post_id} has a comment with no valid comment_id.")
                if comment_id in comments_by_id:
                    raise ValueError(f"Duplicate comment_id found in selected_data.json: {comment_id}")

                entry = {
                    "bucket": bucket,
                    "post_id": post_id,
                    "post_index": post_index,
                    "comment_index": comment_index,
                    "comment_id": comment_id,
                    "comment": comment,
                }
                comments_by_id[comment_id] = entry
                work_items.append(entry)

    return posts_by_id, comments_by_id, work_items


def is_comment_classified(comment):
    return comment.get("Classification") in BUCKETS


def comment_counts(work_items):
    by_label = {bucket: 0 for bucket in BUCKETS}
    classified = 0
    for item in work_items:
        label = item["comment"].get("Classification")
        if label in BUCKETS:
            by_label[label] += 1
            classified += 1
    total = len(work_items)
    return {
        "total": total,
        "classified": classified,
        "remaining": total - classified,
        "by_label": by_label,
    }


def normalize_failures(failures, comments_by_id):
    latest_by_comment = {}
    for entry in ensure_list(failures):
        if not isinstance(entry, dict):
            continue
        comment_id = entry.get("comment_id")
        if not isinstance(comment_id, str) or comment_id not in comments_by_id:
            continue
        if is_comment_classified(comments_by_id[comment_id]["comment"]):
            continue
        latest_by_comment[comment_id] = entry
    return list(latest_by_comment.values())


def failed_comment_ids(failures):
    return {
        entry["comment_id"]
        for entry in failures
        if isinstance(entry, dict) and isinstance(entry.get("comment_id"), str)
    }


def pending_work_items(work_items, failures, retry_failed=False):
    failed_ids = failed_comment_ids(failures)
    pending = []
    skipped_already_classified = 0
    skipped_failed = 0

    for item in work_items:
        if is_comment_classified(item["comment"]):
            skipped_already_classified += 1
            continue
        if not retry_failed and item["comment_id"] in failed_ids:
            skipped_failed += 1
            continue
        pending.append(item)

    return pending, {
        "already_classified": skipped_already_classified,
        "failed_skipped": skipped_failed,
    }


def upsert_failure(failures, work_item, error_message):
    comment_id = work_item["comment_id"]
    filtered = [
        entry
        for entry in ensure_list(failures)
        if not (isinstance(entry, dict) and entry.get("comment_id") == comment_id)
    ]
    comment = work_item["comment"]
    filtered.append({
        "comment_id": comment_id,
        "post_id": work_item["post_id"],
        "bucket": work_item["bucket"],
        "Source": comment.get("Source"),
        "Author": comment.get("Author"),
        "Error": error_message,
        "Failed_At": datetime.now(timezone.utc).isoformat(),
    })
    return filtered


def clear_failure(failures, comment_id):
    return [
        entry
        for entry in ensure_list(failures)
        if not (isinstance(entry, dict) and entry.get("comment_id") == comment_id)
    ]


def prepare_post_for_prompt(post):
    return {
        "ID": post.get("ID"),
        "Source": post.get("Source"),
        "Type": post.get("Type"),
        "Author": post.get("Author"),
        "Title": post.get("Title"),
        "Text": post.get("Text"),
        "Score": post.get("Score"),
        "Date": post.get("Date"),
        "Word_Count": post.get("Word_Count"),
    }


def prepare_comment_for_prompt(comment):
    return {
        "comment_id": comment.get("comment_id"),
        "parent_id": comment.get("parent_id"),
        "Source": comment.get("Source"),
        "Author": comment.get("Author"),
        "Text": comment.get("Text"),
        "Score": comment.get("Score"),
        "Date": comment.get("Date"),
        "Word_Count": comment.get("Word_Count"),
    }


def build_context_payload(work_item, posts_by_id, comments_by_id):
    target_comment = work_item["comment"]
    enclosing_post = posts_by_id[work_item["post_id"]]["post"]

    ancestor_comments = []
    parent_id = target_comment.get("parent_id")
    visited = {work_item["comment_id"]}
    context_resolution = "resolved_to_post"
    root_post = None

    while isinstance(parent_id, str) and parent_id:
        if parent_id in visited:
            context_resolution = "cycle_detected_fallback_to_enclosing_post"
            break
        visited.add(parent_id)

        if parent_id in comments_by_id:
            parent_entry = comments_by_id[parent_id]
            ancestor_comments.append(prepare_comment_for_prompt(parent_entry["comment"]))
            parent_id = parent_entry["comment"].get("parent_id")
            continue

        if parent_id in posts_by_id:
            root_post = posts_by_id[parent_id]["post"]
            break

        context_resolution = "missing_parent_fallback_to_enclosing_post"
        break

    if root_post is None:
        root_post = enclosing_post
        if context_resolution == "resolved_to_post":
            context_resolution = "used_enclosing_post"

    ancestor_comments.reverse()

    return {
        "target_comment": prepare_comment_for_prompt(target_comment),
        "ancestor_comments": ancestor_comments,
        "root_post": prepare_post_for_prompt(root_post),
        "enclosing_post_id": work_item["post_id"],
        "context_resolution": context_resolution,
        "ancestor_depth": len(ancestor_comments),
    }


def build_prompt(payload):
    return (
        "Classify the TARGET_COMMENT using the instructions from GEMINI_comments.md.\n\n"
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
        "- Classify the TARGET_COMMENT only.\n"
        "- Use ANCESTOR_COMMENTS and ROOT_POST only as context to resolve references or implied stance.\n"
        "- Do not classify the parent comments or post instead of the target comment.\n"
        "- If the target comment is still too fragmentary or ambiguous even with the supplied context, classify it as Irrelevant.\n"
        "- Reject non-English or mostly non-English target comments by classifying them as Irrelevant.\n"
        "- Do not translate non-English text to infer relevance or sentiment.\n"
        "- If the target comment is not meaningfully about vibe-coding or AI-assisted coding as a practice, classify it as Irrelevant.\n\n"
        "PAYLOAD:\n"
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


def run_llm(prompt, timeout_seconds, model, base_url, api_key, system_prompt_path, on_stream_event=None):
    return create_stateless_completion(
        prompt,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        system_prompt_path=system_prompt_path,
        on_stream_event=on_stream_event,
    )


def classify_comment(
    comment_id,
    payload,
    max_retries,
    timeout_seconds,
    model,
    base_url,
    api_key,
    system_prompt_path,
):
    prompt_start = time.perf_counter()
    prompt = build_prompt(payload)
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
                    system_prompt_path=system_prompt_path,
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
                    "attempt_total_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
                })
                if attempt == max_retries:
                    raise ParseOutputError(
                        f"Failed to parse model output for comment {comment_id} after {max_retries} attempts. "
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
                "response_parse_ms": round(response_parse_ms, 3),
                "attempt_total_ms": round((time.perf_counter() - attempt_started) * 1000, 3),
            })
            return result, build_classification_timing(prompt_build_ms, attempts)
        except PipelineStopError:
            raise

    raise ParseOutputError(
        f"Failed to parse model output for comment {comment_id} after {max_retries} attempts.",
        timing_info=build_classification_timing(prompt_build_ms, attempts),
    ) from None


def apply_result_to_comment(comment, result):
    comment["Classification"] = result["classification"]
    comment["Classification_Confidence"] = result["confidence"]
    comment["Classification_Reasoning"] = result["reasoning"]
    comment["Classification_Evidence"] = result["evidence"]
    comment["Classified_At"] = datetime.now(timezone.utc).isoformat()


def print_progress(prefix, counts, processed_now, classification=None):
    parts = [
        prefix,
        f"processed_this_run={processed_now}",
        f"classified={counts['classified']}/{counts['total']}",
        f"remaining={counts['remaining']}",
        f"by_label={counts['by_label']}",
    ]
    if classification:
        parts.append(f"classification={classification}")
    print(" | ".join(parts), flush=True)


def append_timing_log(entry):
    logs = load_json(TIMING_LOGS_PATH, [])
    logs = ensure_list(logs)
    logs.append(entry)
    save_json(TIMING_LOGS_PATH, logs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-comments",
        type=int,
        default=None,
        help="Stop after processing this many comments in the current run.",
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
        help="Seconds to wait for stream activity before skipping the comment.",
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
    parser.add_argument(
        "--system-prompt-path",
        default=DEFAULT_SYSTEM_PROMPT_PATH,
        help="Path to the comment classification system prompt.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry comments already recorded in comment_failed_item.json.",
    )
    return parser.parse_args()


def main():
    run_started = time.perf_counter()
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("comment_run_%Y%m%dT%H%M%S_%fZ")

    if not os.path.exists(args.system_prompt_path):
        print(f"Error: system prompt was not found at {args.system_prompt_path}.", file=sys.stderr)
        sys.exit(1)
    if not args.model:
        print("Error: model is required. Pass --model or set OPENAI_MODEL.", file=sys.stderr)
        sys.exit(1)

    selected = ensure_selected_shape(load_json(SELECTED_PATH, {}))
    if not any(selected.get(bucket) for bucket in BUCKETS):
        print("Error: selected_data.json is missing or invalid.", file=sys.stderr)
        sys.exit(1)

    try:
        posts_by_id, comments_by_id, work_items = build_indexes(selected)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    failures = normalize_failures(load_json(FAILED_PATH, []), comments_by_id)
    save_json(FAILED_PATH, failures)
    if not os.path.exists(TIMING_LOGS_PATH) or os.path.getsize(TIMING_LOGS_PATH) == 0:
        save_json(TIMING_LOGS_PATH, [])

    pending, pending_meta = pending_work_items(work_items, failures, retry_failed=args.retry_failed)
    print(
        "Comments discovered: "
        f"total={len(work_items)} | "
        f"already_classified={pending_meta['already_classified']} | "
        f"failed_skipped={pending_meta['failed_skipped']} | "
        f"pending={len(pending)}",
        flush=True,
    )

    processed_this_run = 0
    run_status = "finished"

    for work_item in pending:
        counts_before = comment_counts(work_items)
        if counts_before["remaining"] == 0:
            print("Stopping: all comments are already classified.", flush=True)
            break

        if args.max_comments is not None and processed_this_run >= args.max_comments:
            print(
                f"Stopping: reached --max-comments={args.max_comments}. "
                f"Current progress: {comment_counts(work_items)}",
                flush=True,
            )
            break

        iteration_started = time.perf_counter()
        iteration_started_at = datetime.now(timezone.utc).isoformat()
        payload = build_context_payload(work_item, posts_by_id, comments_by_id)
        comment = work_item["comment"]
        comment_id = work_item["comment_id"]

        print(
            "Classifying comment "
            f"{comment_id} | source={comment.get('Source')} | post_id={work_item['post_id']} | "
            f"ancestor_depth={payload['ancestor_depth']} | context_resolution={payload['context_resolution']}",
            flush=True,
        )

        status_label = "failed"
        error_message = None
        classification = None
        timing_details = {
            "ancestor_depth": payload["ancestor_depth"],
            "context_resolution": payload["context_resolution"],
        }

        try:
            result, classify_timing = classify_comment(
                comment_id,
                payload,
                args.max_retries,
                args.timeout_seconds,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                system_prompt_path=args.system_prompt_path,
            )
            timing_details.update(classify_timing)
            classification = result["classification"]

            persist_started = time.perf_counter()
            apply_result_to_comment(comment, result)
            save_json(SELECTED_PATH, selected)
            failures = clear_failure(failures, comment_id)
            save_json(FAILED_PATH, failures)
            timing_details["persist_ms"] = round((time.perf_counter() - persist_started) * 1000, 3)

            processed_this_run += 1
            status_label = "classified"
            print_progress("Classified", comment_counts(work_items), processed_this_run, classification)
        except ParseOutputError as exc:
            error_message = str(exc)
            timing_details.update(exc.timing_info)
            persist_started = time.perf_counter()
            failures = upsert_failure(failures, work_item, error_message)
            save_json(FAILED_PATH, failures)
            timing_details["persist_ms"] = round((time.perf_counter() - persist_started) * 1000, 3)
            processed_this_run += 1
            status_label = "failed_parse_error"
            print_progress("Failed", comment_counts(work_items), processed_this_run)
            print(f"  reason: {exc}", flush=True)
        except SkipItemError as exc:
            error_message = str(exc)
            timing_details.update(exc.timing_info)
            persist_started = time.perf_counter()
            failures = upsert_failure(failures, work_item, error_message)
            save_json(FAILED_PATH, failures)
            timing_details["persist_ms"] = round((time.perf_counter() - persist_started) * 1000, 3)
            processed_this_run += 1
            status_label = "skipped_timeout"
            print_progress("Skipped", comment_counts(work_items), processed_this_run)
            print(f"  reason: {exc}", flush=True)
        except PipelineStopError as exc:
            run_status = "paused_error"
            error_message = str(exc)
            timing_details.update(exc.timing_info)
            timing_details["iteration_total_ms"] = round((time.perf_counter() - iteration_started) * 1000, 3)
            append_timing_log({
                "run_id": run_id,
                "sequence": processed_this_run + 1,
                "comment_id": comment_id,
                "post_id": work_item["post_id"],
                "source": comment.get("Source"),
                "status": "stopped_error",
                "classification": None,
                "started_at": iteration_started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "counts_before": counts_before,
                "counts_after": comment_counts(work_items),
                "timings_ms": timing_details,
                "error": error_message,
                "model": args.model,
            })
            print(f"Stopping immediately due to pipeline error: {exc}", flush=True)
            break
        except Exception as exc:  # noqa: BLE001
            run_status = "paused_error"
            error_message = str(exc)
            timing_details["iteration_total_ms"] = round((time.perf_counter() - iteration_started) * 1000, 3)
            append_timing_log({
                "run_id": run_id,
                "sequence": processed_this_run + 1,
                "comment_id": comment_id,
                "post_id": work_item["post_id"],
                "source": comment.get("Source"),
                "status": "stopped_unexpected_error",
                "classification": None,
                "started_at": iteration_started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "counts_before": counts_before,
                "counts_after": comment_counts(work_items),
                "timings_ms": timing_details,
                "error": error_message,
                "model": args.model,
            })
            print(f"Stopping immediately due to unexpected error: {exc}", flush=True)
            break

        timing_details["iteration_total_ms"] = round((time.perf_counter() - iteration_started) * 1000, 3)
        entry = {
            "run_id": run_id,
            "sequence": processed_this_run,
            "comment_id": comment_id,
            "post_id": work_item["post_id"],
            "source": comment.get("Source"),
            "status": status_label,
            "classification": classification,
            "started_at": iteration_started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "counts_before": counts_before,
            "counts_after": comment_counts(work_items),
            "timings_ms": timing_details,
            "model": args.model,
        }
        if error_message:
            entry["error"] = error_message
        append_timing_log(entry)

    append_timing_log({
        "run_id": run_id,
        "type": "run_summary",
        "status": run_status,
        "processed_this_run": processed_this_run,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "runtime_ms": round((time.perf_counter() - run_started) * 1000, 3),
        "final_counts": comment_counts(work_items),
        "model": args.model,
    })


if __name__ == "__main__":
    main()
