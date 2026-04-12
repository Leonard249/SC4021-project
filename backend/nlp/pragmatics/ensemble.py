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

from .length_routing.sentic_vader import SenticVaderClassifier, flatten_pos_tags
from .length_routing.transformer_polarity import TransformerPolarityClassifier

logger = logging.getLogger(__name__)


SHORT_THRESHOLD = 1    # Sentence_Word_Count below this → SenticVader

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


        subj_counts = {"subjective": 0, "objective": 0, "irrelevant": 0, "other": 0}
        for r in records:
            lbl = (r.get("Subjectivity") or "").lower()
            key = lbl if lbl in subj_counts else "other"
            subj_counts[key] += 1
            for c in r.get("Comments") or []:
                lbl_c = (c.get("Subjectivity") or "").lower()
                key_c = lbl_c if lbl_c in subj_counts else "other"
                subj_counts[key_c] += 1
        logger.info(
            f"PolarityEnsemble | Subjectivity gate — "
            f"subjective: {subj_counts['subjective']}, "
            f"objective: {subj_counts['objective']}, "
            f"irrelevant: {subj_counts['irrelevant']}, "
            f"other/missing: {subj_counts['other']}. "
            f"Only 'subjective' records reach polarity classification."
        )

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
    def _classify_container(self, container: dict, flat_pos_tags: list[list]) -> None:
        subj_label = container.get("Subjectivity", "objective").lower()

        # if subj_label == "irrelevant":
        #     container["Aspect_Sentiments"] = []
        #     container["Overall_Document_Polarity"] = "irrelevant"
        #     container["Dominant_Routing_Path"] = "none"
        #     return

        # if subj_label != "subjective":
        #     container["Aspect_Sentiments"] = []
        #     container["Overall_Document_Polarity"] = "neutral"
        #     container["Dominant_Routing_Path"] = "none"
        #     return

        # --- Step 1: aspect-level analysis (kept for Aspect_Sentiments output) ---
        aspects: list[dict] = container.get("Targeted_Aspects") or []
        aspect_sentiments: list[dict] = []

        for aspect in aspects:
            sent = aspect.get("Target_Sentence", "")
            wc = aspect.get("Sentence_Word_Count", len(sent.split()))
            sarcasm_data = aspect.get("Sarcasm", {})
            is_sarcastic = sarcasm_data.get("Is_Sarcastic", False)
            sarcasm_conf = sarcasm_data.get("Sarcasm_Confidence", 0.0)

            raw_result = self._route(sent, wc, flat_pos_tags)
            final_polarity, final_score = self._apply_sarcasm_correction(
                raw_result["Label"], raw_result["Confidence"], is_sarcastic, sarcasm_conf
            )
            aspect_sentiments.append({
                "Aspect": aspect.get("Aspect_Name", ""),
                "Routing_Path": raw_result.get("Routing_Path", "unknown"),
                "Final_Polarity": final_polarity,
                "Final_Score": round(final_score, 4),
            })

        container["Aspect_Sentiments"] = aspect_sentiments

        # --- Step 2: DOCUMENT-LEVEL polarity — always run on full text ---
        full_text = container.get("Normalized_Text", "").strip()
        if full_text:
            wc = len(full_text.split())
            raw = self._transformer.classify(full_text, wc, confidence_floor=0.25)
            label = raw.get("Label", "neutral")
            confidence = raw.get("Confidence", 0.0)

            # --- Step 3: Aspect tiebreaker ---
            # If transformer is unsure (neutral), use aspect scores as tiebreaker
            if label == "neutral" and aspect_sentiments:
                scores = [a["Final_Score"] for a in aspect_sentiments]
                aspect_mean = sum(scores) / len(scores)
                if aspect_mean >= 0.25:
                    label = "positive"
                elif aspect_mean <= -0.25:
                    label = "negative"
                # else: stay neutral

            container["Overall_Document_Polarity"] = label
            container["Dominant_Routing_Path"] = "transformer_direct"
        else:
            container["Overall_Document_Polarity"] = "neutral"
            container["Dominant_Routing_Path"] = "none"

    # Routing

    def _route(
        self,
        target_sentence: str,
        sentence_word_count: int,
        flat_pos_tags: list[list],
    ) -> dict:
        """Dispatch to SenticVader or TransformerPolarity based on word count."""
        
        if sentence_word_count < self.short_threshold:
            result = self._sentic_vader.classify(target_sentence, flat_pos_tags)
            result["Routing_Path"] = "vader"
            return result
        else:
            result = self._transformer.classify(target_sentence, sentence_word_count)
            result["Routing_Path"] = "transformer"
            return result
        

    # Sarcasm correction
    @staticmethod
    def _apply_sarcasm_correction(
        label: str,
        confidence: float,
        is_sarcastic: bool,
        sarcasm_confidence: float = 0.0
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
        # Only flip if we are VERY confident it is sarcasm
        if is_sarcastic and sarcasm_confidence > 0.99 and label != "neutral":
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
    def _aggregate(aspect_sentiments: list[dict]) -> tuple[str, str]:
        if not aspect_sentiments:
            return "neutral", "none"

        scores = [a["Final_Score"] for a in aspect_sentiments]
        mean_score = sum(scores) / len(scores)

        # Use the most extreme score to prevent dilution
        max_score = max(scores)
        min_score = min(scores)

        # Pick the pole with the stronger signal
        if abs(min_score) >= abs(max_score):
            extreme_score = min_score
        else:
            extreme_score = max_score

        # 60% mean + 40% extreme — mirrors the subjectivity aggregation logic
        blended = 0.6 * mean_score + 0.4 * extreme_score

        strongest_aspect = max(aspect_sentiments, key=lambda a: abs(a["Final_Score"]))
        dominant_path = strongest_aspect.get("Routing_Path", "unknown")

        if blended >= 0.1:
            return "positive", dominant_path
        elif blended <= -0.1:
            return "negative", dominant_path

        return "neutral", dominant_path
    



def load_json(path: str | Path) -> list[dict] | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    project_root = Path(__file__).resolve().parents[3]
    dataset_path = project_root / "data" / "my_test" / "ensemble_input.json"

    data = load_json(dataset_path)
    records: list[dict] = data if isinstance(data, list) else [data]

    ensemble = PolarityEnsemble()
    ensemble.classify_corpus(records)

    result_to_print = records if len(records) > 1 else records[0]
    # print(json.dumps(result_to_print, indent=4))
    print(result_to_print)