"""
sentic_vader.py
SC4021 Information Retrieval 2026 — Short-Text Polarity Classifier

Pipeline position:
    ... → SarcasmDetector → PolarityEnsemble → SenticVaderClassifier ← here
                                                (dispatched for Target_Sentences < 60 words)

Input
-----
    target_sentence : str       — Target_Sentence from a Targeted_Aspects entry
    pos_tags        : list      — flat list of [token, POS, lemma] triples
                                  sourced from the parent record's POS_Tags
                                  (used for SenticNet lemma lookup)

Output (in-memory dict, not written to record — Ensemble owns record writes)
------
    {
        "Label":        "positive" | "negative" | "neutral",
        "Score":         float,   # 0.0–1.0 on positive axis
        "Confidence":    float,   # 0.0–1.0, distance from 0.5 midpoint × 2
        "Routing_Path":  "short",
        "Classifier":    "sentic_vader"
    }

Fusion strategy
---------------
VADER compound score and SenticNet mean polarity_value are each normalised
to [0, 1] then fused with a weighted average:

    vader_norm = (compound + 1) / 2          maps −1…+1 → 0…1
    sn_norm    = (mean_polarity + 1) / 2      maps −1…+1 → 0…1

    final_score = VADER_WEIGHT * vader_norm + SENTICNET_WEIGHT * sn_norm

    VADER_WEIGHT     = 0.55   (preferred — handles negation and punctuation)
    SENTICNET_WEIGHT = 0.45

If SenticNet is unavailable or returns no hits, the classifier falls back to
VADER alone (effective weight 1.0).

Label thresholds:
    score >= 0.55  → "positive"
    score <= 0.45  → "negative"
    otherwise      → "neutral"

Requires:
    pip install vaderSentiment senticnet
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

VADER_WEIGHT      = 0.55
SENTICNET_WEIGHT  = 0.45
POSITIVE_THRESHOLD = 0.55
NEGATIVE_THRESHOLD = 0.45


class SenticVaderClassifier:
    """
    Short-text polarity classifier combining VADER and SenticNet.

    Called by PolarityEnsemble for Target_Sentences with fewer than 60 words.

    Usage
    -----
    classifier = SenticVaderClassifier()
    result = classifier.classify(target_sentence, pos_tags)
    """

    def __init__(self) -> None:
        self._vader     = None
        self._senticnet = None
        self._load_vader()
        self._load_senticnet()

    def classify(
        self,
        target_sentence: str,
        pos_tags: Optional[list[list]] = None,
    ) -> dict:
        """
        Classify the polarity of a short Target_Sentence.

        Parameters
        ----------
        target_sentence : Target_Sentence from a Targeted_Aspects entry.
        pos_tags        : flat list of [token, POS, lemma] triples from the
                          parent record's POS_Tags (used for SenticNet lookup).
                          Pass [] or None when unavailable.

        Returns
        -------
        {
            "Label":       "positive" | "negative" | "neutral",
            "Score":        float,   # 0.0–1.0
            "Confidence":   float,   # 0.0–1.0
            "Routing_Path": "short",
            "Classifier":   "sentic_vader"
        }
        """
        clean = re.sub(r"<CODE>|\[[^\]]+\]", "", target_sentence).strip()
        if not clean:
            return self._neutral_result()

        vader_norm = self._vader_score(clean)
        sn_norm    = self._senticnet_score(pos_tags or [])

        if sn_norm is None:
            # SenticNet unavailable / no hits — VADER only.
            final = vader_norm
        else:
            final = VADER_WEIGHT * vader_norm + SENTICNET_WEIGHT * sn_norm

        label = self._label(final)
        confidence = round(abs(final - 0.5) * 2, 4)

        return {
            "Label":label,
            "Score":round(final, 4),
            "Confidence":confidence,
            "Routing_Path":"short",
            "Classifier":"sentic_vader",
        }

    # VADER
    def _vader_score(self, text: str) -> float:
        if self._vader is None:
            return 0.5
        compound = self._vader.polarity_scores(text)["compound"]
        return (compound + 1.0) / 2.0

    def _load_vader(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.info("SenticVaderClassifier: VADER loaded.")
        except ImportError:
            logger.warning(
                "vaderSentiment not installed — VADER signal disabled. "
                "Install with: pip install vaderSentiment"
            )

    # SenticNet
    def _senticnet_score(self, pos_tags: list[list]) -> Optional[float]:
        """
        Look up each lemma in SenticNet; return mean polarity_value
        normalised to [0, 1]. Returns None if no hits or SenticNet absent.
        """
        if self._senticnet is None:
            return None

        polarities: list[float] = []
        for entry in pos_tags:
            if len(entry) < 3:
                continue
            lemma = entry[2].lower().strip()
            if not lemma:
                continue
            try:
                concept = self._senticnet.concept(lemma)
                pv = float(concept.get("polarity_value", 0.0))
                polarities.append(pv)
            except Exception:
                continue  # lemma not in SenticNet

        if not polarities:
            return None

        mean_pv = sum(polarities) / len(polarities)
        return (mean_pv + 1.0) / 2.0

    def _load_senticnet(self) -> None:
        try:
            from senticnet.senticnet import SenticNet
            self._senticnet = SenticNet()
            logger.info("SenticVaderClassifier: SenticNet loaded.")
        except ImportError:
            logger.warning(
                "senticnet not installed — SenticNet signal disabled. "
                "Install with: pip install senticnet"
            )
        except Exception as e:
            logger.warning(f"SenticVaderClassifier: SenticNet failed to load: {e}")

    @staticmethod
    def _label(score: float) -> str:
        if score >= POSITIVE_THRESHOLD:
            return "positive"
        if score <= NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    @staticmethod
    def _neutral_result() -> dict:
        return {
            "Label":"neutral",
            "Score":0.5,
            "Confidence":0.0,
            "Routing_Path":"short",
            "Classifier":"sentic_vader",
        }


# Utility — flatten sentence-aligned POS_Tags into a single list
def flatten_pos_tags(pos_tags: list[list[list]]) -> list[list]:
    """
    Convert sentence-aligned POS_Tags (list of lists of triples) into a flat
    list of [token, POS, lemma] triples for SenticNet lemma lookup.
    """
    flat: list[list] = []
    for sentence_tags in pos_tags:
        if isinstance(sentence_tags, list):
            flat.extend(sentence_tags)
    return flat

if __name__ == "__main__":
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    # Updated to the exact path requested
    dataset_path = Path('../../../../data/my_test/sentic_vader_input.json')

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    records: list[dict] = data if isinstance(data, list) else [data]

    classifier = SenticVaderClassifier()

    for record in records:
        flat_pos = flatten_pos_tags(record.get("POS_Tags") or [])
        for asp in record.get("Targeted_Aspects") or []:
            sent = asp.get("Target_Sentence", "")
            wc   = asp.get("Sentence_Word_Count", len(sent.split()))
            
            if wc >= 60:
                continue  # would be handled by transformer
            
            # Fetch the in-memory dictionary
            result_to_print = classifier.classify(sent, flat_pos)
            
            # Print exactly the dictionary output formatted with indent=4
            print(json.dumps(result_to_print, indent=4))