"""
run_pipeline.py
SC4021 Information Retrieval 2026 — Full Corpus Pipeline Runner

Runs the complete NLP pipeline on all records in raw_data.json.
Designed for 20,000–30,000 records with full comment processing.

Directory layout (relative to this script at backend/test/):
    backend/
    ├── test/
    │   └── final_pipeline.py          ← this script
    │   └── output/                  ← created automatically
    ├── nlp/
    │   ├── syntactics/
    │   │   ├── microtextnorm.py
    │   │   ├── sbd.py
    │   │   └── pos_tagger.py
    │   └── semantics/
    │       ├── ner_tagger.py
    │       └── subjectivity_detector.py
    └── utils/
        └── spacy_utils.py
    ../data/
    └── raw_data.json

Pipeline stages (run in batch mode for performance):
    1. MicrotextNormalizer  — text cleaning, emoji/slang handling
    2. SBD                  — sentence boundary disambiguation
    3. POSTagger             — sentence-aligned POS tagging
    4. NERTagger             — named entity recognition
    5. SubjectivityDetector — hybrid lexicon + transformer subjectivity

Checkpointing:
    Progress is saved every CHECKPOINT_EVERY records to a .jsonl checkpoint
    file. If the run is interrupted, re-running the script will automatically
    skip already-processed records and continue from where it left off.

    Checkpoint file : backend/test/output/checkpoint.jsonl
    Final output    : backend/test/output/pipeline_output.json
    Summary report  : backend/test/output/pipeline_summary.txt
    Error log       : backend/test/output/pipeline_errors.jsonl

Usage:
    python backend/test/final_pipeline.py
    python backend/test/final_pipeline.py --chunk-size 200 --no-transformer
    python backend/test/final_pipeline.py --fresh          # ignore existing checkpoint, restart
    python backend/test/final_pipeline.py --limit 500      # process only first 500 records (dev/test)
"""

import sys
import os
import json
import argparse
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent          # backend/test/
BACKEND_DIR  = SCRIPT_DIR.parent                        # backend/
CORPUS_PATH  = BACKEND_DIR.parent / "data" / "raw_data.json"
OUTPUT_DIR   = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.jsonl"
FINAL_OUTPUT    = OUTPUT_DIR / "pipeline_output.json"
SUMMARY_PATH    = OUTPUT_DIR / "pipeline_summary.txt"
ERROR_LOG_PATH  = OUTPUT_DIR / "pipeline_errors.jsonl"

# Add all module directories to sys.path
for subdir in [
    BACKEND_DIR / "nlp" / "syntactics",
    BACKEND_DIR / "nlp" / "semantics",
    BACKEND_DIR / "utils",
]:
    sys.path.insert(0, str(subdir))

# ---------------------------------------------------------------------------
# Logging — write to both console and a log file
# ---------------------------------------------------------------------------

LOG_PATH = OUTPUT_DIR / "pipeline_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SC4021 full corpus pipeline runner."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help=(
            "Number of records to process per batch before checkpointing. "
            "Larger = faster (better spaCy batching), smaller = more frequent saves. "
            "Default: 200"
        ),
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Disable transformer in subjectivity detection. Much faster, less accurate.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and restart from the beginning.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records. Useful for development runs.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help=f"Path to corpus JSON (default: {CORPUS_PATH})",
    )
    parser.add_argument(
        "--emoticons",
        type=Path,
        default=BACKEND_DIR.parent / "data" / "lexicons" / "emoticon_dict.json",
        help="Path to emoticon dictionary JSON.",
    )
    parser.add_argument(
        "--slang",
        type=Path,
        default=BACKEND_DIR.parent / "data" / "lexicons" / "slang_dict.json",
        help="Path to slang/acronym dictionary JSON.",
    )
    parser.add_argument(
        "--mpqa",
        type=Path,
        default=BACKEND_DIR.parent / "data" / "lexicons" / "mpqa_subjclues.tff",
        help="Path to MPQA subjectivity clues file.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Module imports (after sys.path is set up)
# ---------------------------------------------------------------------------

def import_pipeline_modules():
    """Import all pipeline modules with clear error messages."""
    modules = {}

    imports = [
        ("MicrotextNormalizer", "microtextnorm",        "syntactics"),
        ("SentenceBoundaryDisambiguator", "sbd",        "syntactics"),
        ("POSTagger",           "pos_tagger",           "syntactics"),
        ("NERTagger",           "ner_tagger",           "semantics"),
        ("SubjectivityDetector","subjectivity_detector","semantics"),
    ]

    for class_name, module_name, folder in imports:
        try:
            mod = __import__(module_name)
            modules[class_name] = getattr(mod, class_name)
            logger.info(f"  ✓ {class_name} imported from nlp/{folder}/{module_name}.py")
        except ImportError as e:
            logger.error(
                f"  ✗ Could not import {class_name} from nlp/{folder}/{module_name}.py\n"
                f"    Error: {e}\n"
                f"    sys.path includes: {[p for p in sys.path if 'backend' in p]}"
            )
            sys.exit(1)
        except AttributeError:
            logger.error(
                f"  ✗ Module nlp/{folder}/{module_name}.py found but class "
                f"'{class_name}' not defined inside it."
            )
            sys.exit(1)

    return modules

# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path) -> tuple[list[dict], set[str]]:
    """
    Load already-processed records from checkpoint JSONL.
    Returns (processed_records, set_of_processed_ids).
    """
    if not checkpoint_path.exists():
        return [], set()

    processed = []
    processed_ids = set()
    corrupt_lines = 0

    with open(checkpoint_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rid = record.get("ID") or record.get("id") or f"line_{lineno}"
                processed.append(record)
                processed_ids.add(str(rid))
            except json.JSONDecodeError:
                corrupt_lines += 1

    if corrupt_lines:
        logger.warning(
            f"Checkpoint: {corrupt_lines} corrupt lines skipped "
            "(likely from interrupted write — safe to ignore)."
        )

    logger.info(
        f"Checkpoint loaded: {len(processed)} records already processed."
    )
    return processed, processed_ids


def append_to_checkpoint(records: list[dict], checkpoint_path: Path) -> None:
    """Append a batch of processed records to the checkpoint JSONL file."""
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_error(record: dict, error: Exception, error_log_path: Path) -> None:
    """Append a structured error entry to the error log."""
    entry = {
        "id":        record.get("ID", "unknown"),
        "source":    record.get("Source", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "error":     str(error),
        "traceback": traceback.format_exc(),
    }
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Pipeline runner — chunk-based with batch processing
# ---------------------------------------------------------------------------

def run_pipeline_on_chunk(
    chunk: list[dict],
    normalizer,
    sbd,
    pos_tagger,
    ner_tagger,
    detector,
    apply_spellcheck: bool,
    error_log_path: Path,
) -> tuple[list[dict], dict]:
    """
    Run all pipeline stages on a chunk of records using batch methods.
    Returns (successfully_processed, per_stage_timings).
    """
    timings = {
        "normalization": 0.0,
        "sbd":           0.0,
        "pos_tagging":   0.0,
        "ner_tagging":   0.0,
        "subjectivity":  0.0,
    }

    # --- Stage 1: Microtext Normalization (record-by-record, fast) ---
    t0 = time.perf_counter()
    failed_ids = set()
    for record in chunk:
        try:
            normalizer.normalize_record(record, apply_spellcheck=apply_spellcheck)
        except Exception as e:
            logger.warning(f"  Normalization failed for {record.get('ID','?')}: {e}")
            log_error(record, e, error_log_path)
            record["pipeline_error"] = f"normalization: {e}"
            failed_ids.add(record.get("ID"))
    timings["normalization"] += time.perf_counter() - t0

    clean_chunk = [r for r in chunk if r.get("ID") not in failed_ids]

    # --- Stage 2: SBD (record-by-record, very fast) ---
    t0 = time.perf_counter()
    for record in clean_chunk:
        try:
            sbd.tag_record(record)
        except Exception as e:
            logger.warning(f"  SBD failed for {record.get('ID','?')}: {e}")
            log_error(record, e, error_log_path)
            record["pipeline_error"] = f"sbd: {e}"
            failed_ids.add(record.get("ID"))
    timings["sbd"] += time.perf_counter() - t0

    clean_chunk = [r for r in chunk if r.get("ID") not in failed_ids]

    # --- Stage 3: POS Tagging (batch — most efficient here) ---
    t0 = time.perf_counter()
    try:
        pos_tagger.tag_corpus(clean_chunk)
    except Exception as e:
        logger.error(f"  POS batch failed on chunk: {e} — falling back to per-record")
        for record in clean_chunk:
            try:
                pos_tagger.tag_record(record)
            except Exception as re:
                log_error(record, re, error_log_path)
                record["pipeline_error"] = f"pos: {re}"
                failed_ids.add(record.get("ID"))
    timings["pos_tagging"] += time.perf_counter() - t0

    clean_chunk = [r for r in chunk if r.get("ID") not in failed_ids]

    # --- Stage 4: NER Tagging (batch) ---
    t0 = time.perf_counter()
    try:
        ner_tagger.tag_corpus(clean_chunk)
    except Exception as e:
        logger.error(f"  NER batch failed on chunk: {e} — falling back to per-record")
        for record in clean_chunk:
            try:
                ner_tagger.tag_record(record)
            except Exception as re:
                log_error(record, re, error_log_path)
                record["pipeline_error"] = f"ner: {re}"
                failed_ids.add(record.get("ID"))
    timings["ner_tagging"] += time.perf_counter() - t0

    clean_chunk = [r for r in chunk if r.get("ID") not in failed_ids]

    # --- Stage 5: Subjectivity Detection ---
    t0 = time.perf_counter()
    try:
        detector.detect_corpus(clean_chunk)
    except Exception as e:
        logger.error(f"  Subjectivity batch failed on chunk: {e} — falling back to per-record")
        for record in clean_chunk:
            try:
                detector.detect_record(record)
            except Exception as re:
                log_error(record, re, error_log_path)
                record["pipeline_error"] = f"subjectivity: {re}"
    timings["subjectivity"] += time.perf_counter() - t0

    # Return all records in the chunk (including ones with errors — they
    # still have their raw fields and any partial pipeline fields)
    return chunk, timings

# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_final_json(all_records: list[dict], output_path: Path) -> None:
    logger.info(f"Writing final JSON output ({len(all_records)} records)...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Final JSON written: {output_path} ({size_mb:.1f} MB)")


def write_summary(
    all_records: list[dict],
    total_timings: dict,
    total_time: float,
    args: argparse.Namespace,
    summary_path: Path,
) -> None:
    n = len(all_records)
    if n == 0:
        return

    lines = []
    div = "=" * 70
    lines.append(div)
    lines.append("SC4021 FULL CORPUS PIPELINE — RUN SUMMARY")
    lines.append(f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Corpus     : {args.corpus}")
    lines.append(f"Records    : {n}")
    lines.append(f"Chunk size : {args.chunk_size}")
    lines.append(f"Transformer: {'disabled' if args.no_transformer else 'enabled'}")
    lines.append(div)

    # Timings
    lines.append("\nSTAGE TIMINGS")
    lines.append("-" * 50)
    for stage, t in total_timings.items():
        ms_per = (t / n * 1000) if n else 0
        lines.append(f"  {stage:<22} {t:>8.1f}s   {ms_per:>7.1f} ms/record")
    lines.append(f"  {'TOTAL':<22} {total_time:>8.1f}s   {total_time/n*1000:>7.1f} ms/record")
    eta_str = str(timedelta(seconds=int(total_time)))
    lines.append(f"  Wall clock             {eta_str}")

    # Error stats
    errors = [r for r in all_records if "pipeline_error" in r]
    lines.append(f"\nERRORS: {len(errors)} records failed ({len(errors)/n*100:.1f}%)")

    # Subjectivity distribution
    subj_labels = [r.get("Subjectivity", "unknown") for r in all_records]
    subj_n  = subj_labels.count("subjective")
    obj_n   = subj_labels.count("objective")
    unk_n   = subj_labels.count("unknown")
    lines.append(f"\nSUBJECTIVITY DISTRIBUTION (post-level)")
    lines.append("-" * 50)
    lines.append(f"  Subjective : {subj_n:>6} ({subj_n/n*100:.1f}%)")
    lines.append(f"  Objective  : {obj_n:>6} ({obj_n/n*100:.1f}%)")
    if unk_n:
        lines.append(f"  Unknown    : {unk_n:>6} ({unk_n/n*100:.1f}%)")

    # Comment-level subjectivity
    all_comments = [c for r in all_records for c in (r.get("Comments") or [])]
    if all_comments:
        c_labels = [c.get("Subjectivity", "unknown") for c in all_comments]
        c_subj = c_labels.count("subjective")
        c_obj  = c_labels.count("objective")
        c_n    = len(all_comments)
        lines.append(f"\nSUBJECTIVITY DISTRIBUTION (comment-level, {c_n} comments)")
        lines.append("-" * 50)
        lines.append(f"  Subjective : {c_subj:>6} ({c_subj/c_n*100:.1f}%)")
        lines.append(f"  Objective  : {c_obj:>6} ({c_obj/c_n*100:.1f}%)")

    # Source distribution
    by_source: dict[str, int] = {}
    for r in all_records:
        src = r.get("Source", "Unknown")
        by_source[src] = by_source.get(src, 0) + 1
    lines.append(f"\nSOURCE DISTRIBUTION")
    lines.append("-" * 50)
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        lines.append(f"  {src:<20} {count:>6} ({count/n*100:.1f}%)")

    # NER entity summary
    entity_counts: dict[str, int] = {}
    for r in all_records:
        for ent_text, ent_label, *_ in (r.get("NER_Tags") or []):
            entity_counts[ent_label] = entity_counts.get(ent_label, 0) + 1
    if entity_counts:
        lines.append(f"\nNER ENTITY LABEL COUNTS (post-level)")
        lines.append("-" * 50)
        for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {label:<15} {count:>8}")

    lines.append("\n" + div)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary written: {summary_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("SC4021 Full Corpus Pipeline")
    logger.info(f"  Corpus     : {args.corpus}")
    logger.info(f"  Chunk size : {args.chunk_size}")
    logger.info(f"  Transformer: {'disabled' if args.no_transformer else 'enabled'}")
    logger.info(f"  Fresh run  : {args.fresh}")
    logger.info(f"  Limit      : {args.limit or 'none (full corpus)'}")
    logger.info(f"  Output dir : {OUTPUT_DIR}")
    logger.info("=" * 60)

    # --- Validate paths ---
    for label, path in [
        ("Corpus",        args.corpus),
        ("Emoticon dict", args.emoticons),
        ("Slang dict",    args.slang),
    ]:
        if not path.exists():
            sys.exit(f"[ERROR] {label} not found: {path}")

    # --- Import modules ---
    logger.info("Importing pipeline modules...")
    mods = import_pipeline_modules()

    # --- Load corpus ---
    logger.info(f"Loading corpus from {args.corpus}...")
    with open(args.corpus, encoding="utf-8") as f:
        corpus = json.load(f)

    if not isinstance(corpus, list):
        sys.exit("[ERROR] Corpus JSON must be a top-level array.")

    if args.limit:
        corpus = corpus[:args.limit]
        logger.info(f"Limit applied: using first {args.limit} records.")

    total_records = len(corpus)
    logger.info(f"Corpus loaded: {total_records} records.")

    # --- Checkpoint / resume ---
    if args.fresh and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Fresh run: existing checkpoint deleted.")

    processed_records, processed_ids = load_checkpoint(CHECKPOINT_PATH)

    # Filter out already-processed records
    pending = [
        r for r in corpus
        if str(r.get("ID", r.get("id", ""))) not in processed_ids
    ]

    logger.info(
        f"Records to process: {len(pending)} "
        f"({len(processed_records)} already done from checkpoint)."
    )

    if not pending:
        logger.info("All records already processed — converting checkpoint to final JSON.")
        write_final_json(processed_records, FINAL_OUTPUT)
        write_summary(processed_records, {k: 0.0 for k in
            ["normalization","sbd","pos_tagging","ner_tagging","subjectivity"]},
            0.0, args, SUMMARY_PATH)
        return

    # --- Initialise pipeline ---
    logger.info("Initialising pipeline components...")
    normalizer = mods["MicrotextNormalizer"](
        emoticons_path=args.emoticons,
        slang_path=args.slang,
    )
    sbd        = mods["SentenceBoundaryDisambiguator"]()
    pos_tagger = mods["POSTagger"](model="en_core_web_sm")
    ner_tagger = mods["NERTagger"](model="en_core_web_sm")
    detector   = mods["SubjectivityDetector"](
        use_transformer=not args.no_transformer,
        mpqa_path=args.mpqa

    )
    logger.info("All components ready.")

    # --- Chunk and process ---
    total_timings = {
        "normalization": 0.0,
        "sbd":           0.0,
        "pos_tagging":   0.0,
        "ner_tagging":   0.0,
        "subjectivity":  0.0,
    }

    chunks = [
        pending[i : i + args.chunk_size]
        for i in range(0, len(pending), args.chunk_size)
    ]
    n_chunks    = len(chunks)
    run_start   = time.perf_counter()
    total_done  = len(processed_records)

    logger.info(
        f"Processing {len(pending)} records in {n_chunks} chunks "
        f"of {args.chunk_size}..."
    )

    for chunk_idx, chunk in enumerate(chunks, 1):
        chunk_start = time.perf_counter()

        logger.info(
            f"[Chunk {chunk_idx:>4}/{n_chunks}] "
            f"records {total_done + 1}–{total_done + len(chunk)} "
            f"of {total_records}..."
        )

        processed_chunk, chunk_timings = run_pipeline_on_chunk(
            chunk=chunk,
            normalizer=normalizer,
            sbd=sbd,
            pos_tagger=pos_tagger,
            ner_tagger=ner_tagger,
            detector=detector,
            apply_spellcheck=False,
            error_log_path=ERROR_LOG_PATH,
        )

        # Accumulate timings
        for stage in total_timings:
            total_timings[stage] += chunk_timings.get(stage, 0.0)

        # Checkpoint after every chunk
        append_to_checkpoint(processed_chunk, CHECKPOINT_PATH)
        processed_records.extend(processed_chunk)
        total_done += len(processed_chunk)

        # Progress log
        chunk_elapsed  = time.perf_counter() - chunk_start
        run_elapsed    = time.perf_counter() - run_start
        records_done   = total_done - (total_records - len(pending))
        pct            = records_done / len(pending) * 100
        rate           = records_done / run_elapsed if run_elapsed > 0 else 0
        eta_secs       = (len(pending) - records_done) / rate if rate > 0 else 0

        logger.info(
            f"  ✓ Chunk done in {chunk_elapsed:.1f}s | "
            f"Total: {total_done}/{total_records} ({pct:.1f}%) | "
            f"Rate: {rate:.1f} rec/s | "
            f"ETA: {str(timedelta(seconds=int(eta_secs)))}"
        )

    # --- Finalise ---
    total_elapsed = time.perf_counter() - run_start
    logger.info(
        f"Pipeline complete. {total_done} records processed in "
        f"{str(timedelta(seconds=int(total_elapsed)))}."
    )

    write_final_json(processed_records, FINAL_OUTPUT)
    write_summary(
        processed_records, total_timings, total_elapsed, args, SUMMARY_PATH
    )

    # Print console summary
    n = len(processed_records)
    subj = sum(1 for r in processed_records if r.get("Subjectivity") == "subjective")
    obj  = sum(1 for r in processed_records if r.get("Subjectivity") == "objective")
    errs = sum(1 for r in processed_records if "pipeline_error" in r)

    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"  Records processed : {n}")
    print(f"  Subjective        : {subj} ({subj/n*100:.1f}%)")
    print(f"  Objective         : {obj}  ({obj/n*100:.1f}%)")
    print(f"  Errors            : {errs} ({errs/n*100:.1f}%)")
    print(f"  Total time        : {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"  Output            : {FINAL_OUTPUT}")
    print(f"  Summary           : {SUMMARY_PATH}")
    if errs:
        print(f"  Error log         : {ERROR_LOG_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()