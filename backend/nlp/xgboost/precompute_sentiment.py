"""
precompute_sentiment.py
SC4021 — Pre-compute RoBERTa sentiment probabilities

Runs cardiffnlp/twitter-roberta-base-sentiment on every record's
Normalized_Text and saves the output probabilities to disk as a JSON
lookup dict keyed by record ID.

This is a one-time precomputation step. The saved file is then loaded
by FeatureExtractor as Group I features — giving XGBoost a direct
polarity signal that the symbolic pipeline lacks.

Model output labels:
    LABEL_0 → negative
    LABEL_1 → neutral
    LABEL_2 → positive

Output
------
    data/models/sentiment_probs.json
    {
        "<record_id>": {
            "p_positive": 0.72,
            "p_negative": 0.08,
            "p_neutral":  0.20
        },
        ...
    }

Usage
-----
    python precompute_sentiment.py
    python precompute_sentiment.py --input path/to/other.json --batch-size 16
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import pipeline, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[2]

DEFAULT_INPUT  = _PROJECT / "data" / "results" / "classified_eval_new.json"
DEFAULT_OUTPUT = _PROJECT / "data" / "models" / "sentiment_probs.json"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# Maps model output labels to human-readable names
LABEL_MAP = {
    "LABEL_0": "p_negative",
    "LABEL_1": "p_neutral",
    "LABEL_2": "p_positive",
}

LOG_EVERY = 100   # print progress every N records


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str, max_words: int = 400) -> str:
    """
    Strip pipeline artifacts and truncate to max_words before feeding
    to RoBERTa (which has a 512-token limit).
    """
    # Remove code blocks and emoticon tokens added by Stage 1
    text = re.sub(r"<CODE>", " code ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Word-level truncation (cheap proxy for token-level)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def precompute(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    batch_size: int = 32,
):
    # ── Load records ─────────────────────────────────────────────────────
    print(f"Loading records from: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"  {len(records)} records loaded\n")

    # ── Set up model ─────────────────────────────────────────────────────
    device = 0 if torch.cuda.is_available() else -1
    device_name = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    print(f"Loading model: {MODEL_NAME}  (device: {device_name})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        top_k=None,           # return all class probabilities
        device=device,
        truncation=True,
        max_length=512,
    )
    print("  Model loaded.\n")

    # ── Build input texts ─────────────────────────────────────────────────
    ids   = [r.get("ID", str(i)) for i, r in enumerate(records)]
    texts = [clean_text(r.get("Normalized_Text", "") or "") for r in records]

    # Replace empty strings with a placeholder so the model doesn't crash
    texts = [t if t.strip() else "no content" for t in texts]

    # ── Run inference in batches ──────────────────────────────────────────
    print(f"Running inference (batch_size={batch_size})...")
    results = {}
    total = len(texts)

    for start in range(0, total, batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_ids   = ids[start : start + batch_size]

        batch_out = classifier(batch_texts)

        for record_id, label_scores in zip(batch_ids, batch_out):
            # label_scores is a list of {"label": ..., "score": ...} dicts
            probs = {LABEL_MAP[d["label"]]: round(d["score"], 6) for d in label_scores}
            results[record_id] = probs

        processed = min(start + batch_size, total)
        if processed % LOG_EVERY == 0 or processed == total:
            print(f"  {processed}/{total} records processed...")

    # ── Save ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} entries → {output_path}")

    # ── Quick sanity check ────────────────────────────────────────────────
    sample_id = ids[0]
    print(f"\nSample ({sample_id}): {results[sample_id]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute RoBERTa sentiment probabilities for XGBoost features"
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=DEFAULT_INPUT,
        help=f"Path to classified JSON (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=DEFAULT_OUTPUT,
        help=f"Where to save sentiment_probs.json (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference (default: 32, reduce if OOM)"
    )
    args = parser.parse_args()

    precompute(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
    )
