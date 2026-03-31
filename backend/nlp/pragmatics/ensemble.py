"""
ensemble.py
SC4021 Information Retrieval 2026 — Polarity Ensemble & Length-Aware Router

Pipeline position:
    ... → SarcasmDetector → PolarityEnsemble ← here (final pragmatics stage)
                                ↓
                    Aspect_Sentiments + Overall_Document_Polarity

This module combines length-aware routing and ensemble aggregation.
It operates at the aspect level: each entry in Targeted_Aspects is classified
individually using the appropriate classifier, then the per-aspect results are
aggregated into an overall document polarity.

Input (per aspect)
------------------
    Target_Sentence     — the sentence the aspect appears in
    Sentence_Word_Count — used for routing (short / medium / long)
    Is_Sarcastic        — from aspect["Sarcasm"]["Is_Sarcastic"]
    pos_tags (flat)     — from the parent record's POS_Tags (for SenticNet)

Length-aware routing (per aspect)
----------------------------------
    Sentence_Word_Count < 60    → SenticVaderClassifier
    Sentence_Word_Count >= 60   → TransformerPolarityClassifier
        → internally: <= 400 words → medium path (single pass)
                      >  400 words → long path   (chunk → classify → aggregate)

Sarcasm correction (per aspect)
--------------------------------
If Is_Sarcastic == True for an aspect, the polarity label of that aspect is
flipped (positive ↔ negative) and its Final_Score is negated before being
stored. Neutral aspects are not flipped.

Final_Score convention
----------------------
Scores are stored on a signed −1.0 … +1.0 scale in Aspect_Sentiments:

    positive aspect  →  Final_Score =  confidence   (0.0 to  1.0)
    negative aspect  →  Final_Score = −confidence   (0.0 to −1.0)
    neutral  aspect  →  Final_Score =  0.0

This makes the overall aggregation intuitive: mean(Final_Scores) > 0 is
positive-leaning, < 0 is negative-leaning.

Overall_Document_Polarity
--------------------------
    mean(Final_Scores) >= 0.1  → "positive"
    mean(Final_Scores) <= -0.1 → "negative"
    otherwise                  → "neutral"

Fields written to each subjective record / comment
---------------------------------------------------
    Aspect_Sentiments  — list of per-aspect dicts (see output format below)
    Overall_Document_Polarity — "positive" | "negative" | "neutral"

Output format per aspect in Aspect_Sentiments:
    {
        "Aspect":        "VS Code",
        "Routing_Path":  "short",
        "Final_Polarity": "negative",
        "Final_Score":   -0.65
    }

Objective containers are skipped:
    Aspect_Sentiments         = []
    Overall_Document_Polarity = "neutral"

Requires:
    pip install vaderSentiment senticnet transformers torch
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Resolve classifiers/ relative to this file so the import works regardless
# of where the script is called from.
# sys.path.insert(0, str(Path(__file__).resolve().parent))

from length_routing.sentic_vader import SenticVaderClassifier, flatten_pos_tags
from length_routing.transformer_polarity import TransformerPolarityClassifier

logger = logging.getLogger(__name__)


SHORT_THRESHOLD = 60    # Sentence_Word_Count below this → SenticVader

# Overall_Document_Polarity thresholds (on signed −1…+1 scale)
OVERALL_POSITIVE_THRESHOLD = 0.1
OVERALL_NEGATIVE_THRESHOLD = -0.1

class PolarityEnsemble:
    """
    Per-aspect polarity router and ensemble for the SC4021 pragmatics layer.

    For each subjective record (and comment), iterates over Targeted_Aspects,
    routes each aspect's Target_Sentence to the appropriate classifier,
    applies sarcasm correction, then aggregates into Overall_Document_Polarity.

    Usage
    -----
    ensemble = PolarityEnsemble()
    record = ensemble.classify_record(record)

    For a full corpus:
    records = ensemble.classify_corpus(records)
    """

    def __init__(self, short_threshold: int = SHORT_THRESHOLD) -> None:
        self.short_threshold = short_threshold
        self._sentic_vader  = SenticVaderClassifier()
        self._transformer   = TransformerPolarityClassifier()

    def classify_record(self, record: dict) -> dict:
        """
        Classify polarity for a single record and all its comments in-place.

        For comments, the parent record's POS_Tags are passed to SenticVader
        as contextual lemmas (comments are often too short to carry their own
        rich POS annotation).

        Returns the modified record.
        """
        # Flatten parent POS_Tags once — reused for all comments.
        parent_pos_tags = flatten_pos_tags(record.get("POS_Tags") or [])

        # Post 
        self._classify_container(record, flat_pos_tags=parent_pos_tags)

        # Comments (inherit parent POS context) 
        for comment in record.get("Comments") or []:
            # Use comment's own POS_Tags if available, else fall back to parent.
            comment_pos = flatten_pos_tags(comment.get("POS_Tags") or [])
            effective_pos = comment_pos if comment_pos else parent_pos_tags
            self._classify_container(comment, flat_pos_tags=effective_pos)

        return record

    def classify_corpus(self, records: list[dict]) -> list[dict]:
        """Classify polarity for an entire list of records."""
        total = len(records)
        for i, record in enumerate(records, 1):
            try:
                self.classify_record(record)
            except Exception as e:
                logger.error(
                    f"PolarityEnsemble failed on record "
                    f"{record.get('ID', i)}: {e}"
                )
            if i % 500 == 0:
                logger.info(
                    f"PolarityEnsemble: {i}/{total} records processed."
                )
        logger.info(f"PolarityEnsemble: complete. {total} records processed.")
        return records

    # Container-level processing

    def _classify_container(
        self,
        container: dict,
        flat_pos_tags: list[list],
    ) -> None:
        """
        Classify every aspect in container["Targeted_Aspects"] and write:
            Aspect_Sentiments         — list of per-aspect result dicts
            Overall_Document_Polarity — aggregated polarity label

        Objective containers are skipped (empty list + "neutral").
        """
        if container.get("Subjectivity", "objective") != "subjective":
            container["Aspect_Sentiments"]         = []
            container["Overall_Document_Polarity"] = "neutral"
            return

        aspects: list[dict] = container.get("Targeted_Aspects") or []
        if not aspects:
            container["Aspect_Sentiments"]         = []
            container["Overall_Document_Polarity"] = "neutral"
            return

        aspect_sentiments: list[dict] = []

        for aspect in aspects:
            sent = aspect.get("Target_Sentence", "")
            wc = aspect.get("Sentence_Word_Count", len(sent.split()))
            is_sarcastic = (
                aspect.get("Sarcasm", {}).get("Is_Sarcastic", False)
            )

            # Route to the correct classifier.
            raw_result = self._route(sent, wc, flat_pos_tags)

            # Apply sarcasm correction.
            final_polarity, final_score = self._apply_sarcasm_correction(
                raw_result["Label"],
                raw_result["Confidence"],
                is_sarcastic,
            )

            aspect_sentiments.append({
                "Aspect": aspect.get("Aspect_Name", ""),
                "Routing_Path": raw_result.get("Routing_Path", "unknown"),
                "Final_Polarity": final_polarity,
                "Final_Score": round(final_score, 4),
            })

        container["Aspect_Sentiments"]         = aspect_sentiments
        container["Overall_Document_Polarity"] = self._aggregate(
            aspect_sentiments
        )

    # Routing

    def _route(
        self,
        target_sentence: str,
        sentence_word_count: int,
        flat_pos_tags: list[list],
    ) -> dict:
        """Dispatch to SenticVader or TransformerPolarity based on word count."""
        if sentence_word_count < self.short_threshold:
            return self._sentic_vader.classify(target_sentence, flat_pos_tags)
        else:
            return self._transformer.classify(target_sentence, sentence_word_count)

    # Sarcasm correction
    @staticmethod
    def _apply_sarcasm_correction(
        label: str,
        confidence: float,
        is_sarcastic: bool,
    ) -> tuple[str, float]:
        """
        Convert (label, confidence) → (final_polarity, final_score) on −1…+1.

        If is_sarcastic, flip positive ↔ negative before converting.
        Neutral is never flipped.

        final_score convention:
            positive → +confidence
            negative → −confidence
            neutral  →  0.0
        """
        if is_sarcastic and label != "neutral":
            label = "negative" if label == "positive" else "positive"

        if label == "positive":
            final_score = confidence
        elif label == "negative":
            final_score = -confidence
        else:
            final_score = 0.0

        return label, final_score

    # Aggregation
    @staticmethod
    def _aggregate(aspect_sentiments: list[dict]) -> str:
        """
        Compute Overall_Document_Polarity from the mean of Final_Scores.

            mean >= +0.1  → "positive"
            mean <= −0.1  → "negative"
            otherwise → "neutral"
        """
        if not aspect_sentiments:
            return "neutral"

        scores = [a["Final_Score"] for a in aspect_sentiments]
        mean   = sum(scores) / len(scores)

        if mean >= OVERALL_POSITIVE_THRESHOLD:
            return "positive"
        if mean <= OVERALL_NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"


def load_json(path: str | Path) -> list[dict] | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset_path = Path('../../../data/my_test/ensemble_input.json')

    data = load_json(dataset_path)
    records: list[dict] = data if isinstance(data, list) else [data]

    ensemble = PolarityEnsemble()
    ensemble.classify_corpus(records)

    result_to_print = records if len(records) > 1 else records[0]
    # print(json.dumps(result_to_print, indent=4))
    print(result_to_print)