"""
transformer_polarity.py
SC4021 Information Retrieval 2026 — Transformer Polarity Classifier

Pipeline position:
    ... → SarcasmDetector → PolarityEnsemble → TransformerPolarityClassifier ← here
                                                (dispatched for Target_Sentences >= 60 words)

Input
-----
    target_sentence     : str   — Target_Sentence from a Targeted_Aspects entry
    sentence_word_count : int   — Sentence_Word_Count from the same aspect entry

Output (in-memory dict, not written to record — Ensemble owns record writes)
------
    {
        "Label":        "positive" | "negative" | "neutral",
        "Score":         float,   # 0.0–1.0 on positive axis
        "Confidence":    float,   # 0.0–1.0
        "Classifier":    "transformer_polarity",
        "Routing_Path":  "medium" | "long",
        "Chunks":        int      # 1 for medium, N for long
    }

Routing paths
-------------
  medium (60–400 words) — Target_Sentence passed directly to the model
                          in a single forward pass.

  long   (> 400 words)  — Target_Sentence split into overlapping word-level
                          chunks (CHUNK_WORD_LIMIT words each, CHUNK_OVERLAP
                          word overlap). Each chunk is classified independently
                          and results are aggregated via a weighted average
                          where the weight of each chunk equals its word count
                          (longer chunks contribute more to the final score).

Model
-----
    cardiffnlp/twitter-roberta-base-sentiment-latest

    All raw scores are normalised to the positive axis (0 = very negative,
    1 = very positive) so the aggregation math is uniform across paths:
        positive prediction → raw P(positive)
        negative prediction → 1 − P(negative)
        neutral             → 0.5

Label thresholds:
    score >= 0.55  → "positive"
    score <= 0.45  → "negative"
    otherwise      → "neutral"

Requires:
    pip install transformers torch
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL= "cardiffnlp/twitter-roberta-base-sentiment-latest"
CHUNK_WORD_LIMIT= 300
CHUNK_OVERLAP= 50
LONG_THRESHOLD= 400   # words; above this → chunking path
MAX_CHARS= 1500  # hard character cap per model input

POSITIVE_THRESHOLD = 0.55
NEGATIVE_THRESHOLD = 0.45

LABEL_MAP: dict[str, str] = {
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral",
    "LABEL_0":  "negative",
    "LABEL_1":  "neutral",
    "LABEL_2":  "positive",
}

class TransformerPolarityClassifier:
    """
    Transformer-based polarity classifier for medium and long Target_Sentences.

    Called by PolarityEnsemble for aspects with Sentence_Word_Count >= 60.

    Usage
    -----
    classifier = TransformerPolarityClassifier()
    result = classifier.classify(target_sentence, sentence_word_count)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chunk_word_limit: int = CHUNK_WORD_LIMIT,
        chunk_overlap: int = CHUNK_OVERLAP,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.chunk_word_limit = chunk_word_limit
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self._pipeline = None  # lazy-loaded

    def classify(self, target_sentence: str, sentence_word_count: int) -> dict:
        """
        Classify the polarity of a medium or long Target_Sentence.

        Parameters
        ----------
        target_sentence     : Target_Sentence from a Targeted_Aspects entry.
        sentence_word_count : Sentence_Word_Count from the same aspect entry.
                              Determines whether to use the medium or long path.

        Returns
        -------
        {
            "Label": "positive" | "negative" | "neutral",
            "Score": float,
            "Confidence": float,
            "Classifier": "transformer_polarity",
            "Routing_Path": "medium" | "long",
            "Chunks": int
        }
        """
        clean = re.sub(r"<CODE>|\[[^\]]+\]", "", target_sentence).strip()
        if not clean:
            return self._neutral_result("medium")

        if sentence_word_count > LONG_THRESHOLD:
            return self._classify_long(clean)
        else:
            return self._classify_medium(clean)

    # Medium path
    def _classify_medium(self, text: str) -> dict:
        """Single forward pass for medium-length sentences."""
        self._load_pipeline()
        raw    = self._run_model([text[:MAX_CHARS]])[0]
        score, label = self._parse_result(raw)
        return {
            "Label": label,
            "Score": round(score, 4),
            "Confidence": round(abs(score - 0.5) * 2, 4),
            "Classifier": "transformer_polarity",
            "Routing_Path": "medium",
            "Chunks": 1,
        }
    
    # Long path
    def _classify_long(self, text: str) -> dict:
        """
        Chunk the text, classify each chunk, then aggregate by word count.

        Weighted average: longer chunks contribute proportionally more to
        the final score than shorter tail chunks.
        """
        chunks = self._make_chunks(text)
        if not chunks:
            return self._neutral_result("long")

        self._load_pipeline()
        raw_results = self._run_model([c[:MAX_CHARS] for c in chunks])

        weighted_sum = 0.0
        total_weight = 0.0
        for chunk, raw in zip(chunks, raw_results):
            score, _  = self._parse_result(raw)
            weight = len(chunk.split())  # number of words in this chunk
            weighted_sum += score * weight
            total_weight += weight

        final = weighted_sum / total_weight if total_weight else 0.5
        label = self._label(final)

        return {
            "Label": label,
            "Score": round(final, 4),
            "Confidence": round(abs(final - 0.5) * 2, 4),
            "Classifier": "transformer_polarity",
            "Routing_Path": "long",
            "Chunks": len(chunks),
        }

    def _make_chunks(self, text: str) -> list[str]:
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        step   = max(1, self.chunk_word_limit - self.chunk_overlap)
        start  = 0
        while start < len(words):
            end = min(start + self.chunk_word_limit, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += step
        return chunks

    def _run_model(self, texts: list[str]) -> list[dict]:
        results: list[dict] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                out = self._pipeline(batch)
                if isinstance(out, dict):
                    out = [out]
                results.extend(out)
            except Exception as e:
                logger.error(
                    f"TransformerPolarityClassifier: batch error at {i}: {e}"
                )
                results.extend([{"label": "neutral", "score": 1.0}] * len(batch))
        return results

    def _parse_result(self, raw: dict) -> tuple[float, str]:
        """
        Normalise raw model output to (positive_axis_score, label).
            positive  → P(positive)
            negative  → 1 − P(negative)
            neutral   → 0.5
        """
        raw_label = raw.get("label", "neutral")
        label = LABEL_MAP.get(raw_label, "neutral")
        raw_score = float(raw.get("score", 0.5))

        if label == "positive":
            score = raw_score
        elif label == "negative":
            score = 1.0 - raw_score
        else:
            score = 0.5

        return score, label

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(
                f"TransformerPolarityClassifier: loading '{self.model_name}'..."
            )
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )
            logger.info("TransformerPolarityClassifier: model loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load polarity model '{self.model_name}': {e}\n"
                "Install with: pip install transformers torch"
            ) from e

    @staticmethod
    def _label(score: float) -> str:
        if score >= POSITIVE_THRESHOLD:
            return "positive"
        if score <= NEGATIVE_THRESHOLD:
            return "negative"
        return "neutral"

    @staticmethod
    def _neutral_result(route: str) -> dict:
        return {
            "Label": "neutral",
            "Score": 0.5,
            "Confidence": 0.0,
            "Classifier": "transformer_polarity",
            "Routing_Path": route,
            "Chunks": 0,
        }

if __name__ == "__main__":
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    dataset_path = Path('../../../../data/my_test/transformer_input.json')

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    records: list[dict] = data if isinstance(data, list) else [data]

    classifier = TransformerPolarityClassifier()

    for record in records:
        for asp in record.get("Targeted_Aspects") or []:
            sent = asp.get("Target_Sentence", "")
            wc   = asp.get("Sentence_Word_Count", len(sent.split()))
            
            if wc < 60:
                continue  # would be handled by SenticVader
            
            # Fetch the in-memory dictionary
            result_to_print = classifier.classify(sent, wc)
            
            # Print exactly the dictionary output formatted with indent=4
            print(json.dumps(result_to_print, indent=4))