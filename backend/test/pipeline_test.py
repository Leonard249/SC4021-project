"""
pipeline_test.py
SC4021 Information Retrieval 2026 — Full Pipeline Test Script

Loads a random sample of 50 posts from the corpus JSON, runs them through
the complete pipeline, and writes the fully annotated records to an output
JSON file.

Pipeline stages:
    1. MicrotextNormalizer  (nlp/syntactics/microtextnorm.py)
    2. SBD                  (nlp/syntactics/sbd.py)
    3. POSTagger            (nlp/syntactics/pos_tagger.py)
    4. NERTagger            (nlp/semantics/ner_tagger.py)
    5. SubjectivityDetector (nlp/semantics/subjectivity_detector.py)

Usage:
    python backend/test/pipeline_test.py 
    python backend/test/pipeline_test.py --sample 250 --seed 42
    python backend/test/pipeline_test.py --no-transformer

Output:
    pipeline_test_output.json   — fully annotated sampled records
    pipeline_test_summary.txt   — human-readable summary per post
"""

import sys
import os
import json
import random
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — allow imports from nlp/ regardless of where
# the script is run from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

NLP_DIR = BACKEND_DIR / "nlp"
SYNTACTICS_DIR = NLP_DIR / "syntactics"
SEMANTICS_DIR = NLP_DIR / "semantics"

DATA_DIR = PROJECT_ROOT / "data"
LEXICONS_DIR = DATA_DIR / "lexicons"
RECORDS_DIR = DATA_DIR / "processed"

sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports from pipeline modules
# ---------------------------------------------------------------------------
try:
    from backend.nlp.syntactics.microtextnorm import MicrotextNormalizer
except ImportError as e:
    sys.exit(f"[ERROR] Could not import MicrotextNormalizer: {e}\n"
             f"  Expected location: {SYNTACTICS_DIR / 'microtextnorm.py'}")

try:
    from backend.nlp.syntactics.sbd import SentenceBoundaryDisambiguator
except ImportError as e:
    sys.exit(f"[ERROR] Could not import SentenceBoundaryDisambiguator: {e}\n"
             f"  Expected location: {SYNTACTICS_DIR / 'sbd.py'}")

try:
    from backend.nlp.syntactics.pos_tagger import POSTagger
except ImportError as e:
    sys.exit(f"[ERROR] Could not import POSTagger: {e}\n"
             f"  Expected location: {SYNTACTICS_DIR / 'pos_tagger.py'}")

try:
    from backend.nlp.semantics.ner_tagger import NERTagger
except ImportError as e:
    sys.exit(f"[ERROR] Could not import NERTagger: {e}\n"
             f"  Expected location: {SEMANTICS_DIR / 'ner_tagger.py'}")

try:
    from backend.nlp.semantics.subjectivity_detector import SubjectivityDetector
except ImportError as e:
    sys.exit(f"[ERROR] Could not import SubjectivityDetector: {e}\n"
             f"  Expected location: {SEMANTICS_DIR / 'subjectivity_detector.py'}")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SC4021 full pipeline test on a random corpus sample."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=RECORDS_DIR / "training.json",
        help="Path to the corpus JSON file (array of records).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of records to sample (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (default: random).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "pipeline_test_output.json",
        help="Path for the output JSON file.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SCRIPT_DIR / "pipeline_test_summary.txt",
        help="Path for the human-readable summary file.",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Disable transformer fallback in subjectivity detection (faster).",
    )
    parser.add_argument(
        "--spellcheck",
        action="store_true",
        help="Enable spell correction (only applies to X/Twitter records).",
    )
    parser.add_argument(
        "--emoticons",
        type=Path,
        default=LEXICONS_DIR / "emoticon_dict.json",
        help="Path to emoticon dictionary JSON.",
    )
    parser.add_argument(
        "--slang",
        type=Path,
        default=LEXICONS_DIR / "slang_dict.json",
        help="Path to slang/acronym dictionary JSON.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def load_and_sample(
    corpus_path: Path,
    n: int,
    seed: int | None,
) -> list[dict]:
    """
    Load the corpus JSON array and return n randomly sampled records.
    Sampling is stratified by Source so the sample reflects the
    distribution of platforms in the full corpus.
    """
    logger.info(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    if not isinstance(corpus, list):
        sys.exit("[ERROR] Expected corpus JSON to be a top-level array.")

    total = len(corpus)
    logger.info(f"Corpus loaded: {total} records.")

    if n >= total:
        logger.warning(
            f"Requested sample ({n}) >= corpus size ({total}). "
            "Using entire corpus."
        )
        return corpus

    # Stratified sampling by Source
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for record in corpus:
        src = record.get("Source", "Unknown")
        by_source.setdefault(src, []).append(record)

    logger.info("Source distribution in corpus:")
    for src, records in sorted(by_source.items()):
        logger.info(f"  {src:<20} {len(records):>6} records")

    # Proportional allocation per source, minimum 1 per source if possible
    sampled: list[dict] = []
    remaining = n
    sources = sorted(by_source.keys())

    for i, src in enumerate(sources):
        src_records = by_source[src]
        is_last = (i == len(sources) - 1)
        if is_last:
            alloc = remaining
        else:
            proportion = len(src_records) / total
            alloc = max(1, round(proportion * n))
            alloc = min(alloc, remaining, len(src_records))

        chosen = rng.sample(src_records, min(alloc, len(src_records)))
        sampled.extend(chosen)
        remaining -= len(chosen)
        logger.info(f"  Sampled {len(chosen):>3} from {src}")

    # Shuffle the final sample so records aren't grouped by source
    rng.shuffle(sampled)
    logger.info(f"Final sample: {len(sampled)} records.")
    return sampled


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(
    records: list[dict],
    normalizer: MicrotextNormalizer,
    sbd: SentenceBoundaryDisambiguator,
    pos_tagger: POSTagger,
    ner_tagger: NERTagger,
    subjectivity_detector: SubjectivityDetector,
    apply_spellcheck: bool,
) -> tuple[list[dict], dict]:
    """
    Run each record through all five pipeline stages.
    Returns (annotated_records, timing_stats).
    """
    timings = {
        "normalization": 0.0,
        "sbd":           0.0,
        "pos_tagging":   0.0,
        "ner_tagging":   0.0,
        "subjectivity":  0.0,
    }
    total = len(records)

    for i, record in enumerate(records, 1):
        record_id = record.get("ID", f"record_{i}")
        logger.info(f"[{i:>3}/{total}] Processing {record_id}...")

        if not (record.get("Text") or record.get("Title") or "").strip():
            logger.warning(f"[{i:>3}/{total}] Skipping {record_id} — empty Text and Title.")
            record["pipeline_error"] = "empty text"
            continue

        try:
            # Stage 1 — Microtext Normalization
            t0 = time.perf_counter()
            normalizer.normalize_record(record, apply_spellcheck=apply_spellcheck)
            timings["normalization"] += time.perf_counter() - t0

            # Stage 2 - Sentence Boundary Disambiguation
            t0 = time.perf_counter()
            sbd.tag_record(record)
            timings["sbd"] += time.perf_counter() - t0

            # Stage 3 — POS Tagging
            t0 = time.perf_counter()
            pos_tagger.tag_record(record)
            timings["pos_tagging"] += time.perf_counter() - t0

            # Stage 4 — NER Tagging
            t0 = time.perf_counter()
            ner_tagger.tag_record(record)
            timings["ner_tagging"] += time.perf_counter() - t0

            # Stage 5 — Subjectivity Detection
            t0 = time.perf_counter()
            subjectivity_detector.detect_record(record)
            timings["subjectivity"] += time.perf_counter() - t0

        except Exception as e:
            logger.error(f"  Pipeline failed on {record_id}: {e}")
            # Write error state so the record is still present in output
            record["pipeline_error"] = str(e)

    return records, timings


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_json_output(records: list[dict], output_path: Path) -> None:
    """Write fully annotated records to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON output written to {output_path}")
 
 

def write_summary(records: list[dict], timings: dict, sample_size: int, seed: int | None, summary_path: Path) -> None:
    """
    Write a human-readable summary to a text file.
    Each post shows: source, subjectivity label/score, and sentence breakdown.
    """
    lines: list[str] = []
    divider = "=" * 80
 
    lines.append(divider)
    lines.append("SC4021 PIPELINE TEST — SUMMARY REPORT")
    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Sample    : {sample_size} records  |  Seed: {seed or 'random'}")
    lines.append(divider)
 
    # Timing summary
    total_time = sum(timings.values())
    lines.append("\nTIMINGS")
    lines.append("-" * 40)
    for stage, t in timings.items():
        lines.append(f"  {stage:<20} {t:>7.2f}s  ({t/sample_size*1000:.1f}ms/record)")
    lines.append(f"  {'TOTAL':<20} {total_time:>7.2f}s  ({total_time/sample_size*1000:.1f}ms/record)")
 
    # Subjectivity distribution
    labels = [r.get("Subjectivity", "unknown") for r in records]
    subj_count = labels.count("subjective")
    obj_count = labels.count("objective")
    lines.append(f"\nSUBJECTIVITY DISTRIBUTION")
    lines.append("-" * 40)
    lines.append(f"  Subjective : {subj_count:>3} ({subj_count/sample_size*100:.1f}%)")
    lines.append(f"  Objective  : {obj_count:>3} ({obj_count/sample_size*100:.1f}%)")
 
    # Per-record breakdown
    lines.append(f"\n\nPER-RECORD BREAKDOWN")
    lines.append(divider)
 
    for i, record in enumerate(records, 1):
        rid = record.get("ID", f"record_{i}")
        source = record.get("Source", "?")
        title = record.get("Title") or "(no title)"
        raw_text = record.get("Text", "")
        norm_text = record.get("Normalized_Text", "")
        label = record.get("Subjectivity", "unknown")
        score = record.get("Subjectivity_Score", 0.0)
 
        # Entities detected
        ner_tags = record.get("NER_Tags", [])
        entities = ", ".join(
            f"{e[0]}[{e[1]}]" for e in ner_tags
        ) or "none"
 
        lines.append(f"\n[{i:>3}] {rid}  |  {source}")
        lines.append(f"  Title      : {title[:75]}")
        lines.append(f"  Raw text   : {raw_text[:120]}{'...' if len(raw_text) > 120 else ''}")
        lines.append(f"  Normalized : {norm_text[:120]}{'...' if len(norm_text) > 120 else ''}")
        lines.append(f"  Entities   : {entities[:100]}")
        lines.append(f"  Subjectivity → {label.upper()}  (score: {score:.4f})")
 
        # Post sentence breakdown
        sentences = record.get("Subjectivity_Sentences", [])
        if sentences:
            lines.append(f"  Sentences  :")
            for s in sentences:
                preview = s["text"][:65] + "..." if len(s["text"]) > 68 else s["text"]
                marker = "★" if s["label"] == "subjective" else "○"
                lines.append(
                    f"    {marker} [{s['label'][:3].upper()} {s['score']:.2f} "
                    f"via {s['method'][:3]}] {preview}"
                )
 
        # Comment breakdown
        comments = record.get("Comments") or []
        if comments:
            # Summary line — count subjective vs objective comments
            c_labels = [c.get("Subjectivity", "unknown") for c in comments]
            c_subj = c_labels.count("subjective")
            c_obj  = c_labels.count("objective")
            lines.append(
                f"  Comments   : {len(comments)} total — "
                f"{c_subj} subjective, {c_obj} objective"
            )
            for j, comment in enumerate(comments, 1):
                c_id    = comment.get("comment_id", f"c{j}")
                c_author = comment.get("Author", "?")
                c_label = comment.get("Subjectivity", "unknown")
                c_score = comment.get("Subjectivity_Score", 0.0)
                c_text  = comment.get("Text", "")
                marker  = "★" if c_label == "subjective" else "○"
 
                lines.append(
                    f"    {marker} [{c_id} | {c_author}] "
                    f"{c_label.upper()} ({c_score:.3f}) — "
                    f"{c_text[:70]}{'...' if len(c_text) > 70 else ''}"
                )
 
                # Show sentence breakdown for each comment
                c_sentences = comment.get("Subjectivity_Sentences", [])
                for s in c_sentences:
                    preview = s["text"][:55] + "..." if len(s["text"]) > 58 else s["text"]
                    s_marker = "★" if s["label"] == "subjective" else "○"
                    lines.append(
                        f"        {s_marker} [{s['label'][:3].upper()} {s['score']:.2f} "
                        f"via {s['method'][:3]}] {preview}"
                    )
 
        if "pipeline_error" in record:
            lines.append(f"  !! ERROR: {record['pipeline_error']}")
 
        lines.append("-" * 80)
 
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Summary written to {summary_path}")
 
 
def print_console_summary(records: list[dict], timings: dict) -> None:
    """Print a compact summary table to stdout after the run."""
    total = sum(timings.values())
    n = len(records)
 
    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)
    print(f"{'Stage':<22} {'Time':>8}  {'ms/rec':>8}")
    print("-" * 70)
    for stage, t in timings.items():
        print(f"  {stage:<20} {t:>7.2f}s  {t/n*1000:>7.1f}ms")
    print(f"  {'TOTAL':<20} {total:>7.2f}s  {total/n*1000:>7.1f}ms")
 
    labels = [r.get("Subjectivity", "unknown") for r in records]
    subj = labels.count("subjective")
    obj = labels.count("objective")
    print(f"\nPost subjectivity : {subj} subjective ({subj/n*100:.1f}%)  |  "
          f"{obj} objective ({obj/n*100:.1f}%)")
 
    # Comment-level stats
    all_comments = [c for r in records for c in (r.get("Comments") or [])]
    if all_comments:
        c_labels = [c.get("Subjectivity", "unknown") for c in all_comments]
        c_subj = c_labels.count("subjective")
        c_obj  = c_labels.count("objective")
        c_n    = len(all_comments)
        print(f"Comment subjectivity : {c_subj} subjective ({c_subj/c_n*100:.1f}%)  |  "
              f"{c_obj} objective ({c_obj/c_n*100:.1f}%)  [{c_n} comments total]")
 
    # Show 5 most confident subjective and 5 most confident objective
    sorted_records = sorted(records, key=lambda r: r.get("Subjectivity_Score", 0))
    print("\nTop 5 most objective:")
    for r in sorted_records[:5]:
        title = (r.get("Title") or r.get("Text", ""))[:55]
        print(f"  {r.get('Subjectivity_Score', 0):.3f}  [{r.get('Source','?')}]  {title}")
 
    print("\nTop 5 most subjective:")
    for r in sorted_records[-5:][::-1]:
        title = (r.get("Title") or r.get("Text", ""))[:55]
        print(f"  {r.get('Subjectivity_Score', 0):.3f}  [{r.get('Source','?')}]  {title}")
    print("=" * 70)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
 
    # Validate paths
    if not args.corpus.exists():
        sys.exit(f"[ERROR] Corpus file not found: {args.corpus}")
    if not args.emoticons.exists():
        sys.exit(f"[ERROR] Emoticon dict not found: {args.emoticons}")
    if not args.slang.exists():
        sys.exit(f"[ERROR] Slang dict not found: {args.slang}")
 
    # Log config
    logger.info("SC4021 Pipeline Test")
    logger.info(f"  Corpus     : {args.corpus}")
    logger.info(f"  Sample     : {args.sample}")
    logger.info(f"  Seed       : {args.seed or 'random'}")
    logger.info(f"  Transformer: {'disabled' if args.no_transformer else 'enabled'}")
    logger.info(f"  Spellcheck : {'enabled' if args.spellcheck else 'disabled'}")
 
    # Sample records
    records = load_and_sample(args.corpus, args.sample, args.seed)
 
    # Initialise pipeline components
    logger.info("Initialising pipeline components...")
 
    normalizer = MicrotextNormalizer(
        emoticons_path=args.emoticons,
        slang_path=args.slang,
    )

    sbd = SentenceBoundaryDisambiguator()
    pos_tagger = POSTagger(model="en_core_web_sm")
    ner_tagger = NERTagger(model="en_core_web_sm")

    detector = SubjectivityDetector(use_transformer=not args.no_transformer)
 
    logger.info("All components ready. Starting pipeline...")
 
    # Run pipeline
    annotated, timings = run_pipeline(
        records=records,
        normalizer=normalizer,
        sbd=sbd, 
        pos_tagger=pos_tagger,
        ner_tagger=ner_tagger,
        subjectivity_detector=detector,
        apply_spellcheck=args.spellcheck,
    )
 
    # Write outputs
    write_json_output(annotated, args.output)
    write_summary(annotated, timings, args.sample, args.seed, args.summary)
    print_console_summary(annotated, timings)
 
    logger.info("Done.")
 
 
if __name__ == "__main__":
    main()