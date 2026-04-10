"""
classifier.py
SC4021 Information Retrieval 2026 — End-to-End NLP Pipeline Orchestrator

File location: backend/nlp/syntactics/classifier.py

Runs the full pipeline in order:
    Stage 1  — MicrotextNormalizer   (syntactics/microtextnorm.py)
    Stage 2  — SentenceBoundaryDisambiguator (syntactics/sbd.py)
    Stage 3  — POSTagger             (syntactics/pos_tagger.py)
    Stage 4  — NERTagger             (semantics/ner_tagger.py)
    Stage 5  — SubjectivityDetector  (semantics/subjectivity_detector.py)
    Stage 6  — AspectExtractor       (pragmatics/aspect_extractor.py)
    Stage 7  — SarcasmDetector       (pragmatics/sarcasm_detector.py)
    Stage 8  — PolarityEnsemble      (pragmatics/ensemble.py)
                 └─ internally routes to:
                      pragmatics/length_routing/sentic_vader.py       (< 60 words)
                      pragmatics/length_routing/transformer_polarity.py (>= 60 words)

Input  : JSON file at INPUT_PATH  (list of records or a single record dict)
Output : JSON file at OUTPUT_PATH (data/processed/draft.json)

Usage
-----
# Run with defaults (edit INPUT_PATH below):
    python classifier.py

# Or import and call directly:
    from classifier import run_pipeline
    records = run_pipeline(
        input_path="path/to/data.json",
        output_path="data/processed/draft.json",
        apply_spellcheck=False,
    )
"""

import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — resolves sibling packages regardless of working directory.
# Assumes this file lives at:  backend/nlp/syntactics/classifier.py
# ---------------------------------------------------------------------------
_THIS_FILE  = Path(__file__).resolve()
_SYNTACTICS = _THIS_FILE.parent                     # backend/nlp/syntactics/
_NLP_ROOT = _THIS_FILE.parents[1]                 # backend/nlp/
_BACKEND = _THIS_FILE.parents[2]                 # backend/

for _p in [
    str(_SYNTACTICS),                               # microtextnorm, sbd, pos_tagger
    str(_NLP_ROOT / "semantics"),                   # ner_tagger, subjectivity_detector
    str(_NLP_ROOT / "pragmatics"),                  # aspect_extractor, sarcasm_detector, ensemble
    str(_NLP_ROOT / "pragmatics" / "length_routing"),  # sentic_vader, transformer_polarity
    str(_BACKEND),                                  # utils/ (spacy_utils etc.)
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


from syntactics.microtextnorm import MicrotextNormalizer           # Stage 1
from syntactics.sbd import SentenceBoundaryDisambiguator           # Stage 2
from syntactics.pos_tagger import POSTagger                        # Stage 3
from semantics.ner_tagger import NERTagger                         # Stage 4
from semantics.subjectivity_detector import SubjectivityDetector   # Stage 5
from pragmatics.aspect_extractor import AspectExtractor            # Stage 6
from pragmatics.sarcasm_detector import SarcasmDetector            # Stage 7
from pragmatics.ensemble import PolarityEnsemble                   # Stage 8

# Input JSON
# INPUT_PATH: str = "../../data/processed/raw_data.json"
INPUT_PATH: str = "../../data/new_processed/db_labelled_new.json"

# Output JSON: fully annotated records ready for indexing / evaluation.
OUTPUT_PATH: str = "../../data/results/classified_eval_new.json"

# Lexicon paths (relative to backend/nlp/syntactics/ or absolute).
EMOTICON_DICT:str = "../../data/lexicons/emoticon_dict.json"
SLANG_DICT:str = "../../data/lexicons/slang_dict.json"
MPQA_LEXICON:str = "../../data/lexicons/mpqa_subjclues.tff"

# Pipeline toggles
APPLY_SPELLCHECK:bool = False   # Stage 1 — slow; enable for X/Twitter only
USE_TRANSFORMER:bool = True    # Stage 5 — set False for lexicon-only (faster)
SPACY_MODEL:str = "en_core_web_sm"   # or en_core_web_md / en_core_web_lg
LOG_LEVEL:int = logging.INFO
BATCH_LOG_INTERVAL: int  = 100     # log progress every N records

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("classifier")


def load_records(path: str | Path) -> list[dict]:
    """
    Load input JSON. Accepts either a list of records or a single record dict.
    Returns a list in all cases.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or dict, got {type(data).__name__}")

    logger.info(f"Loaded {len(data)} records from {path.resolve()}")
    return data


def save_records(records: list[dict], path: str | Path) -> None:
    """Save annotated records to JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(records)} annotated records → {path.resolve()}")


# ---------------------------------------------------------------------------
# Pipeline builder — instantiates all components once, reuses across records
# ---------------------------------------------------------------------------

class NLPPipeline:
    """
    Lazy-initialised NLP pipeline wrapper.

    Instantiates each component once and exposes a single `run(records)`
    method that processes the full list end-to-end.

    Stages that load ML models (SubjectivityDetector, SarcasmDetector,
    PolarityEnsemble) do so lazily on first use; heavy models are therefore
    not loaded until they are actually needed.
    """

    def __init__(
        self,
        emoticon_dict:str = EMOTICON_DICT,
        slang_dict:str = SLANG_DICT,
        mpqa_lexicon:str = MPQA_LEXICON,
        spacy_model:str = SPACY_MODEL,
        apply_spellcheck:bool = APPLY_SPELLCHECK,
        use_transformer:bool = USE_TRANSFORMER,
    ):
        self.apply_spellcheck = apply_spellcheck

        logger.info("=== Initialising NLP pipeline components ===")

        # Stage 1 — Microtext Normalization
        logger.info("Stage 1 — MicrotextNormalizer")
        self.normalizer = MicrotextNormalizer(
            emoticons_path=emoticon_dict,
            slang_path=slang_dict,
        )

        # Stage 2 — Sentence Boundary Disambiguation
        logger.info("Stage 2 — SentenceBoundaryDisambiguator")
        self.sbd = SentenceBoundaryDisambiguator()

        # Stage 3 — POS Tagging & Lemmatization
        logger.info("Stage 3 — POSTagger")
        self.pos_tagger = POSTagger(model=spacy_model)

        # Stage 4 — Named Entity Recognition
        logger.info("Stage 4 — NERTagger")
        self.ner_tagger = NERTagger(model=spacy_model)

        # Stage 5 — Subjectivity Detection (hybrid lexicon + transformer)
        logger.info("Stage 5 — SubjectivityDetector")
        self.subjectivity = SubjectivityDetector(
            use_transformer=use_transformer,
            mpqa_path=mpqa_lexicon,
        )

        # Stage 6 — Aspect Extraction
        logger.info("Stage 6 — AspectExtractor")
        self.aspect_extractor = AspectExtractor()

        # Stage 7 — Sarcasm Detection
        logger.info("Stage 7 — SarcasmDetector")
        self.sarcasm = SarcasmDetector()

        # Stage 8 — Polarity Ensemble (length-aware routing → VADER+SenticNet / Transformer)
        logger.info("Stage 8 — PolarityEnsemble")
        self.ensemble = PolarityEnsemble()

        logger.info("=== All components initialised ===\n")

    def run(self, records: list[dict]) -> list[dict]:
        """
        Run all pipeline stages over the full record list.

        Corpus-level batch methods are preferred where available (faster
        because spaCy and HuggingFace benefit from batching).  Per-record
        fallback is used for stages that do not expose a batch API.

        Returns the same list with all annotation fields written in-place.
        """
        total = len(records)
        logger.info(f"Starting pipeline over {total} records.\n")
        t0 = time.perf_counter()

        # ------ Stage 1: Microtext Normalization ----------------------
        logger.info("▶ Stage 1/8 — Microtext Normalization")
        t1 = time.perf_counter()
        records = self.normalizer.normalize_corpus(
            records, apply_spellcheck=self.apply_spellcheck
        )
        logger.info(f"  ✓ completed in {time.perf_counter() - t1:.1f}s\n")

        # ------ Stage 2: Sentence Boundary Disambiguation -------------
        logger.info("▶ Stage 2/8 — Sentence Boundary Disambiguation")
        t1 = time.perf_counter()
        records = self.sbd.tag_corpus(records)
        logger.info(f"  ✓ completed in {time.perf_counter() - t1:.1f}s\n")

        # ------ Stage 3: POS Tagging & Lemmatization ------------------
        logger.info("▶ Stage 3/8 — POS Tagging & Lemmatization (batch)")
        t1 = time.perf_counter()
        records = self.pos_tagger.tag_corpus(records)
        logger.info(f"  ✓ completed in {time.perf_counter() - t1:.1f}s\n")

        # ------ Stage 4: Named Entity Recognition ---------------------
        logger.info("▶ Stage 4/8 — Named Entity Recognition (batch)")
        t1 = time.perf_counter()
        records = self.ner_tagger.tag_corpus(records)
        logger.info(f"  ✓ completed in {time.perf_counter() - t1:.1f}s\n")

        # ------ Stage 5: Subjectivity Detection -----------------------
        logger.info("▶ Stage 5/8 — Subjectivity Detection (hybrid lexicon + transformer)")
        t1 = time.perf_counter()
        records = self.subjectivity.detect_corpus(records)
        logger.info(f"  ✓ completed in {time.perf_counter() - t1:.1f}s\n")

        # ------ Stage 6: Aspect Extraction ----------------------------
        logger.info("▶ Stage 6/8 — Aspect Extraction (subjective records only)")
        t1 = time.perf_counter()
        records = self.aspect_extractor.extract_corpus(records)
        logger.info(f"  ✓ completed in {time.perf_counter() - t1:.1f}s\n")

        # ------ Stage 7: Sarcasm Detection ----------------------------
        logger.info("▶ Stage 7/8 — Sarcasm Detection (batched transformer)")
        t1 = time.perf_counter()
        records = self.sarcasm.detect_corpus(records)
        logger.info(f"  ✓ completed in {time.perf_counter() - t1:.1f}s\n")

        # ------ Stage 8: Polarity Ensemble (length-aware routing) -----
        logger.info("▶ Stage 8/8 — Polarity Ensemble (VADER+SenticNet / Transformer)")
        t1 = time.perf_counter()
        records = self.ensemble.classify_corpus(records)
        logger.info(f"  ✓ completed in {time.perf_counter() - t1:.1f}s\n")

        # ------ Summary -----------------------------------------------
        elapsed = time.perf_counter() - t0
        self._log_summary(records, elapsed)

        return records

    @staticmethod
    def _log_summary(records: list[dict], elapsed: float) -> None:
        """Print a brief stats summary after the full pipeline completes."""
        n_total = len(records)
        n_subjective = sum(1 for r in records if r.get("Subjectivity") == "subjective")
        n_objective = sum(1 for r in records if r.get("Subjectivity") == "objective")
        n_irrelevant = sum(1 for r in records if r.get("Subjectivity") == "Irrelevant")
        
        n_positive = sum(1 for r in records if r.get("Overall_Document_Polarity") == "positive")
        n_negative = sum(1 for r in records if r.get("Overall_Document_Polarity") == "negative")
        n_neutral = sum(1 for r in records if r.get("Overall_Document_Polarity") == "neutral")
        rate = n_total / elapsed if elapsed > 0 else 0

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"  Total records: {n_total}")
        logger.info(f"  Subjective: {n_subjective}  ({100*n_subjective/max(n_total,1):.1f}%)")
        logger.info(f"  Objective (skipped): {n_objective}")
        logger.info(f"  Irrelevant (dropped): {n_irrelevant}")
        logger.info(f"  Polarity → positive: {n_positive}")
        logger.info(f"  Polarity → negative: {n_negative}")
        logger.info(f"  Polarity → neutral: {n_neutral}")
        logger.info(f"  Total time: {elapsed:.1f}s  ({rate:.1f} records/s)")
        logger.info("=" * 60)

def run_pipeline(
    input_path:       str | Path = INPUT_PATH,
    output_path:      str | Path = OUTPUT_PATH,
    emoticon_dict:    str = EMOTICON_DICT,
    slang_dict:       str = SLANG_DICT,
    mpqa_lexicon:     str = MPQA_LEXICON,
    spacy_model:      str = SPACY_MODEL,
    apply_spellcheck: bool = APPLY_SPELLCHECK,
    use_transformer:  bool = USE_TRANSFORMER,
) -> list[dict]:
    """
    Load → run full pipeline → save → return annotated records.

    Parameters
    ----------
    input_path       : path to raw input JSON (list of records or single record)
    output_path      : path where annotated JSON will be written
    emoticon_dict    : path to emoticon_dict.json lexicon
    slang_dict       : path to slang_dict.json lexicon
    mpqa_lexicon     : path to mpqa_subjclues.tff lexicon
    spacy_model      : spaCy model name (en_core_web_sm / md / lg)
    apply_spellcheck : run Stage 6 spell correction (slow; X/Twitter only by default)
    use_transformer  : use HuggingFace transformer for subjectivity (Stage 5)

    Returns
    -------
    list of fully annotated record dicts (also saved to output_path)
    """
    records = load_records(input_path)

    pipeline = NLPPipeline(
        emoticon_dict=emoticon_dict,
        slang_dict=slang_dict,
        mpqa_lexicon=mpqa_lexicon,
        spacy_model=spacy_model,
        apply_spellcheck=apply_spellcheck,
        use_transformer=use_transformer,
    )

    records = pipeline.run(records)
    save_records(records, output_path)
    return records

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SC4021 NLP pipeline — raw JSON → fully annotated JSON"
    )
    parser.add_argument(
        "--input",  "-i",
        default=INPUT_PATH,
        help=f"Path to raw input JSON (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_PATH,
        help=f"Path for annotated output JSON (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--spacy-model",
        default=SPACY_MODEL,
        help=f"spaCy model (default: {SPACY_MODEL})",
    )
    parser.add_argument(
        "--spellcheck",
        action="store_true",
        default=APPLY_SPELLCHECK,
        help="Enable spell correction in Stage 1 (slow)",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        default=False,
        help="Disable transformer in Stage 5 (lexicon-only, faster)",
    )
    parser.add_argument(
        "--emoticons",
        default=EMOTICON_DICT,
        help=f"Path to emoticon_dict.json (default: {EMOTICON_DICT})",
    )
    parser.add_argument(
        "--slang",
        default=SLANG_DICT,
        help=f"Path to slang_dict.json (default: {SLANG_DICT})",
    )
    parser.add_argument(
        "--mpqa",
        default=MPQA_LEXICON,
        help=f"Path to mpqa_subjclues.tff (default: {MPQA_LEXICON})",
    )

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        emoticon_dict=args.emoticons,
        slang_dict=args.slang,
        mpqa_lexicon=args.mpqa,
        spacy_model=args.spacy_model,
        apply_spellcheck=args.spellcheck,
        use_transformer=not args.no_transformer,
    )