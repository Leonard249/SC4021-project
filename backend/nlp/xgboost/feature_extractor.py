"""
feature_extractor.py
SC4021 — XGBoost Feature Extractor

Converts a single pipeline-output record (post Stage 1-8) into a flat
numeric feature vector for XGBoost training and inference.

Features (50 total across 9 groups):
    A — Subjectivity signals        (Stage 5 output)
    B — Document-level VADER        (computed on Normalized_Text)
    C — Aspect-level signals        (Stage 8 Aspect_Sentiments)
    D — Sarcasm signals             (Stage 7, stored in Targeted_Aspects)
    E — NER signals                 (Stage 4 NER_Tags)
    F — Text structure              (Normalized_Word_Count, Sentences)
    G — Source bucket (4 buckets)   (social / blog / qa / article)
    H — Per-aspect VADER            (VADER on each Target_Sentence)
    I — RoBERTa sentiment probs     (precomputed via precompute_sentiment.py)

Changes from v1:
    - G: replaced 9 source one-hots with 4 semantic buckets (prevents
         overfitting to small source populations like stackoverflow n=56)
    - H: added per-aspect VADER on Target_Sentence strings for targeted
         sentiment signal beyond document-level VADER

Usage
-----
    extractor = FeatureExtractor()
    extractor = FeatureExtractor(sentiment_probs_path="data/models/sentiment_probs.json")
    vector = extractor.extract(record)          # → list[float], length 50
    names  = extractor.feature_names()          # → list[str],  length 50
    X, y   = extractor.extract_corpus(records)  # → (np.ndarray, np.ndarray)
"""

import json
import re
import math
import numpy as np
from collections import Counter
from pathlib import Path

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None

# Label encoding — must be consistent between training and inference
LABEL_TO_INT = {"positive": 0, "negative": 1, "neutral": 2, "irrelevant": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# Source → semantic bucket mapping (group G)
# Buckets prevent overfitting to small source populations (e.g. stackoverflow n=56)
SOURCE_BUCKET = {
    "Twitter":        "social",
    "HackerNews":     "social",
    "Reddit":         "social",
    "personal_blog":  "blog",
    "medium":         "blog",
    "substack":       "blog",
    "other_blog":     "blog",
    "corporate_blog": "blog",
    "stackoverflow":  "qa",
}
SOURCE_BUCKETS = ["social", "blog", "qa", "article"]  # 'article' = fallback


class FeatureExtractor:
    """
    Feature extractor for XGBoost polarity classifier.

    Parameters
    ----------
    sentiment_probs_path : str | Path | None
        Path to sentiment_probs.json produced by precompute_sentiment.py.
        If None or file not found, Group I features default to zeros.
    """

    def __init__(self, sentiment_probs_path=None):
        self._sentiment_probs: dict = {}
        if sentiment_probs_path is not None:
            path = Path(sentiment_probs_path)
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self._sentiment_probs = json.load(f)
                print(f"  Sentiment probs loaded: {len(self._sentiment_probs)} entries from {path}")
            else:
                print(f"  Warning: sentiment_probs_path not found ({path}) — Group I features will be zeros.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, record: dict) -> list[float]:
        """
        Convert one pipeline-output record into a flat feature vector.
        Returns a list of floats in the same order as feature_names().
        Missing / null values are filled with 0.0.
        """
        feats = []
        feats.extend(self._group_a_subjectivity(record))
        feats.extend(self._group_b_vader(record))
        feats.extend(self._group_c_aspects(record))
        feats.extend(self._group_d_sarcasm(record))
        feats.extend(self._group_e_ner(record))
        feats.extend(self._group_f_text(record))
        feats.extend(self._group_g_source(record))
        feats.extend(self._group_h_aspect_vader(record))
        feats.extend(self._group_i_roberta(record))
        return feats

    def feature_names(self) -> list[str]:
        """Return the ordered list of feature names (matches extract() order)."""
        return (
            self._names_a()
            + self._names_b()
            + self._names_c()
            + self._names_d()
            + self._names_e()
            + self._names_f()
            + self._names_g()
            + self._names_h()
            + self._names_i()
        )

    def extract_corpus(
        self, records: list[dict]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels for a list of labeled records.

        Returns
        -------
        X : np.ndarray, shape (n, n_features)
        y : np.ndarray, shape (n,) — integer-encoded labels
        """
        X_rows, y_rows = [], []
        for record in records:
            label_str = (record.get("label") or "").lower()
            if label_str not in LABEL_TO_INT:
                continue  # skip records without a valid label
            X_rows.append(self.extract(record))
            y_rows.append(LABEL_TO_INT[label_str])

        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    def _names_a(self):
        return [
            "subjectivity_score",
            "is_subjective",
            "is_irrelevant",
            "is_objective",
            "n_subjective_sentences",
            "n_objective_sentences",
            "n_total_sentences_scored",
            "pct_subjective_sentences",
            "max_sentence_subj_score",
            "mean_sentence_subj_score",
            "n_transformer_sentences",
        ]

    def _names_b(self):
        return ["vader_compound", "vader_pos", "vader_neg", "vader_neu"]

    def _names_c(self):
        return [
            "n_aspects",
            "mean_aspect_score",
            "max_aspect_score",
            "min_aspect_score",
            "std_aspect_score",
            "n_positive_aspects",
            "n_negative_aspects",
            "n_neutral_aspects",
            "pct_positive_aspects",
            "pct_negative_aspects",
            "current_polarity_positive",
            "current_polarity_negative",
            "current_polarity_neutral",
        ]

    def _names_d(self):
        return [
            "n_sarcastic_aspects",
            "mean_sarcasm_confidence",
            "max_sarcasm_confidence",
        ]

    def _names_e(self):
        return [
            "n_ai_tool_entities",
            "n_org_entities",
            "n_total_entities",
            "n_distinct_entity_types",
        ]

    def _names_f(self):
        return [
            "log_word_count",
            "n_sentences",
            "avg_sentence_length",
        ]

    def _names_g(self):
        return [f"source_{b}" for b in SOURCE_BUCKETS]

    def _names_h(self):
        return [
            "aspect_vader_mean",
            "aspect_vader_max",
            "aspect_vader_min",
            "aspect_vader_std",
        ]

    def _names_i(self):
        return [
            "roberta_p_positive",
            "roberta_p_negative",
            "roberta_p_neutral",
            "roberta_sentiment_gap",   # p_positive - p_negative (signed polarity strength)
        ]

    # ------------------------------------------------------------------
    # Group A — Subjectivity signals
    # ------------------------------------------------------------------

    def _group_a_subjectivity(self, record: dict) -> list[float]:
        subj_label = record.get("Subjectivity", "objective")
        subj_score = record.get("Subjectivity_Score") or 0.0
        sentences  = record.get("Subjectivity_Sentences") or []

        is_subjective = float(subj_label == "subjective")
        is_irrelevant = float(subj_label == "Irrelevant")
        is_objective  = float(subj_label == "objective")

        n_subj = sum(1 for s in sentences if s.get("label") == "subjective")
        n_obj  = sum(1 for s in sentences if s.get("label") == "objective")
        n_total = len(sentences)
        pct_subj = n_subj / n_total if n_total > 0 else 0.0

        scores = [s.get("score", 0.0) for s in sentences]
        max_score  = max(scores) if scores else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0

        n_transformer = sum(1 for s in sentences if s.get("method") == "transformer")

        return [
            subj_score,
            is_subjective,
            is_irrelevant,
            is_objective,
            float(n_subj),
            float(n_obj),
            float(n_total),
            pct_subj,
            max_score,
            mean_score,
            float(n_transformer),
        ]

    # ------------------------------------------------------------------
    # Group B — Document-level VADER
    # ------------------------------------------------------------------

    def _group_b_vader(self, record: dict) -> list[float]:
        if _vader is None:
            return [0.0, 0.0, 0.0, 0.0]

        text = record.get("Normalized_Text", "")
        clean = re.sub(r"<CODE>|\[[^\]]+\]", " ", text).strip()
        scores = _vader.polarity_scores(clean)
        return [
            scores["compound"],
            scores["pos"],
            scores["neg"],
            scores["neu"],
        ]

    # ------------------------------------------------------------------
    # Group C — Aspect-level signals
    # ------------------------------------------------------------------

    def _group_c_aspects(self, record: dict) -> list[float]:
        aspects = record.get("Aspect_Sentiments") or []
        n = len(aspects)

        if n == 0:
            current_pol = (record.get("Overall_Document_Polarity") or "neutral").lower()
            return [
                0.0, 0.0, 0.0, 0.0, 0.0,   # counts/scores
                0.0, 0.0, 0.0,               # pos/neg/neu counts
                0.0, 0.0,                    # pct pos/neg
                float(current_pol == "positive"),
                float(current_pol == "negative"),
                float(current_pol == "neutral"),
            ]

        scores   = [a.get("Final_Score", 0.0) for a in aspects]
        polarity = [a.get("Final_Polarity", "neutral") for a in aspects]

        n_pos = sum(1 for p in polarity if p == "positive")
        n_neg = sum(1 for p in polarity if p == "negative")
        n_neu = sum(1 for p in polarity if p == "neutral")

        mean_s = sum(scores) / n
        max_s  = max(scores)
        min_s  = min(scores)
        std_s  = float(np.std(scores)) if n > 1 else 0.0

        current_pol = (record.get("Overall_Document_Polarity") or "neutral").lower()

        return [
            float(n),
            mean_s,
            max_s,
            min_s,
            std_s,
            float(n_pos),
            float(n_neg),
            float(n_neu),
            n_pos / n,
            n_neg / n,
            float(current_pol == "positive"),
            float(current_pol == "negative"),
            float(current_pol == "neutral"),
        ]

    # ------------------------------------------------------------------
    # Group D — Sarcasm signals
    # ------------------------------------------------------------------

    def _group_d_sarcasm(self, record: dict) -> list[float]:
        aspects = record.get("Targeted_Aspects") or []

        if not aspects:
            return [0.0, 0.0, 0.0]

        sarcasms = [
            a.get("Sarcasm", {}) for a in aspects if "Sarcasm" in a
        ]
        if not sarcasms:
            return [0.0, 0.0, 0.0]

        n_sarcastic = sum(1 for s in sarcasms if s.get("Is_Sarcastic", False))
        confidences = [s.get("Sarcasm_Confidence", 0.0) for s in sarcasms]
        mean_conf   = sum(confidences) / len(confidences)
        max_conf    = max(confidences)

        return [float(n_sarcastic), mean_conf, max_conf]

    # ------------------------------------------------------------------
    # Group E — NER signals
    # ------------------------------------------------------------------

    def _group_e_ner(self, record: dict) -> list[float]:
        ner_tags = record.get("NER_Tags") or []

        if not ner_tags:
            return [0.0, 0.0, 0.0, 0.0]

        entity_types = [tag[1] for tag in ner_tags if len(tag) >= 2]
        type_counts  = Counter(entity_types)
        n_ai_tool    = float(type_counts.get("AI_TOOL", 0))
        n_org        = float(type_counts.get("ORG", 0))
        n_total      = float(len(ner_tags))
        n_distinct   = float(len(type_counts))

        return [n_ai_tool, n_org, n_total, n_distinct]

    # ------------------------------------------------------------------
    # Group F — Text structure
    # ------------------------------------------------------------------

    def _group_f_text(self, record: dict) -> list[float]:
        word_count = record.get("Normalized_Word_Count") or 1
        sentences  = record.get("Sentences") or []
        n_sent     = len(sentences) or 1

        log_wc      = math.log1p(word_count)
        avg_sent_len = word_count / n_sent

        return [log_wc, float(n_sent), avg_sent_len]

    # ------------------------------------------------------------------
    # Group G — Source bucket (4 semantic categories)
    # ------------------------------------------------------------------

    def _group_g_source(self, record: dict) -> list[float]:
        source = record.get("Source", "")
        bucket = SOURCE_BUCKET.get(source, "article")
        return [float(bucket == b) for b in SOURCE_BUCKETS]

    # ------------------------------------------------------------------
    # Group H — Per-aspect VADER on Target_Sentence
    # ------------------------------------------------------------------

    def _group_h_aspect_vader(self, record: dict) -> list[float]:
        """
        Run VADER on each aspect's Target_Sentence (not the full document).
        This gives targeted sentiment signal at the sentence level rather
        than diluting it across the whole post.
        """
        if _vader is None:
            return [0.0, 0.0, 0.0, 0.0]

        aspects = record.get("Targeted_Aspects") or []
        if not aspects:
            return [0.0, 0.0, 0.0, 0.0]

        compounds = []
        for aspect in aspects:
            sentence = aspect.get("Target_Sentence", "")
            if not sentence:
                continue
            clean = re.sub(r"<CODE>|\[[^\]]+\]", " ", sentence).strip()
            compounds.append(_vader.polarity_scores(clean)["compound"])

        if not compounds:
            return [0.0, 0.0, 0.0, 0.0]

        mean_c = sum(compounds) / len(compounds)
        max_c  = max(compounds)
        min_c  = min(compounds)
        std_c  = float(np.std(compounds)) if len(compounds) > 1 else 0.0

        return [mean_c, max_c, min_c, std_c]

    # ------------------------------------------------------------------
    # Group I — Pre-computed RoBERTa sentiment probabilities
    # ------------------------------------------------------------------

    def _group_i_roberta(self, record: dict) -> list[float]:
        """
        Look up pre-computed cardiffnlp/twitter-roberta-base-sentiment
        probabilities for this record.

        Returns [p_positive, p_negative, p_neutral, sentiment_gap] where
        sentiment_gap = p_positive - p_negative (signed strength measure).

        Falls back to [0, 0, 0, 0] if the record ID is not in the lookup
        (e.g. sentiment_probs.json was not provided or record is new).
        """
        record_id = record.get("ID", "")
        probs = self._sentiment_probs.get(record_id)

        if probs is None:
            return [0.0, 0.0, 0.0, 0.0]

        p_pos = probs.get("p_positive", 0.0)
        p_neg = probs.get("p_negative", 0.0)
        p_neu = probs.get("p_neutral",  0.0)
        gap   = p_pos - p_neg

        return [p_pos, p_neg, p_neu, gap]
