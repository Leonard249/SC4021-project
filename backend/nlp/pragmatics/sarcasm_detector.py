"""
sarcasm_detector.py
SC4021 Information Retrieval 2026 — Sarcasm Detection Module

Pipeline position:
    MicrotextNormalizer → SBD → POSTagger → NERTagger → SubjectivityDetector
    → AspectExtractor → SarcasmDetector ← here → LengthAwareRouting → Ensemble

Subjectivity gate
-----------------
Only containers (posts or comments) whose Subjectivity == "subjective" are
processed. Objective containers are skipped and their Targeted_Aspects are
left unchanged (no Sarcasm key is added).

Parent context for comments
---------------------------
Comments are often too short for the irony model to judge accurately without
knowing what they are replying to. When processing a comment's aspects, the
parent post's Normalized_Text (truncated to 200 chars) is prepended to each
Target_Sentence before classification:

    input = "<parent text> | <comment sentence>"

The Sarcasm result written back into the aspect still reflects the comment
sentence, but the model benefits from the topic context supplied by the parent.
This mirrors the context-prepending strategy used in SubjectivityDetector.

Output
------
A "Sarcasm" key is added in-place to every aspect dict inside Targeted_Aspects:

    {
        "Aspect_Name":        "VS Code",
        "Entity_Type":        "EDITOR",
        "Target_Sentence":    "I've always loved Cline but ...",
        "Sentence_Word_Count": 14,
        "Sarcasm": {
            "Is_Sarcastic":       false,
            "Sarcasm_Confidence": 0.2326
        }
    }

No other fields are modified. The full record structure is preserved exactly.

Model
-----
    cardiffnlp/twitter-roberta-base-irony
    LABEL_0 / non_irony → "Not Sarcastic"
    LABEL_1 / irony     → "Sarcastic"

Sarcasm_Confidence is always P(Sarcastic) regardless of which label was
predicted by the model.

Requires:
    pip install transformers torch
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LABEL_MAP: dict[str, str] = {
    "LABEL_0":   "Not Sarcastic",
    "non_irony": "Not Sarcastic",
    "LABEL_1":   "Sarcastic",
    "irony":     "Sarcastic",
}

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-irony"

# Hard cap before tokenisation — RoBERTa fits ~512 tokens; 600 chars is safe.
MAX_SENTENCE_CHARS = 600

# How many chars of the parent post to prepend as context for comments.
PARENT_CONTEXT_CHARS = 200

class SarcasmDetector:
    """
    Sarcasm / irony detector for the SC4021 pipeline.

    Processes only subjective containers. For comments, the parent post's
    Normalized_Text is prepended to each Target_Sentence before inference
    so the model has topic context.

    Usage
    -----
    detector = SarcasmDetector()
    record = detector.detect_record(record)

    For a full corpus (recommended — enables corpus-wide batching):
    records = detector.detect_corpus(records)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        label_map: Optional[dict[str, str]] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.label_map: dict[str, str] = label_map or LABEL_MAP
        self.batch_size = batch_size
        self._pipeline = None  # lazy-loaded on first use

    def detect_record(self, record: dict) -> dict:
        """
        Run sarcasm detection for a single record and all its comments.

        - Post aspects are classified using the Target_Sentence alone.
        - Comment aspects are classified with the parent post's
          Normalized_Text prepended for context.

        Only subjective containers are processed. Returns the modified record.
        """
        # --- Post ---
        if record.get("Subjectivity") == "subjective":
            self._process_container(record, parent_context="")

        # --- Comments (with parent context) ---
        parent_context = record.get("Normalized_Text", "")[:PARENT_CONTEXT_CHARS]
        for comment in record.get("Comments") or []:
            if comment.get("Subjectivity") == "subjective":
                self._process_container(comment, parent_context=parent_context)

        return record

    def detect_corpus(self, records: list[dict]) -> list[dict]:
        """
        Run sarcasm detection for an entire corpus using a single batched
        forward pass across all records for maximum throughput.

        Subjective-only gate and parent-context injection are both applied.
        """
        # Collect (container, aspect_index, model_input_text) for every
        # aspect that still needs a Sarcasm result.
        work_items: list[tuple[dict, int, str]] = []

        for record in records:
            parent_context = (
                record.get("Normalized_Text", "")[:PARENT_CONTEXT_CHARS]
            )

            # Post aspects — no context prefix.
            if record.get("Subjectivity") == "subjective":
                for idx, aspect in enumerate(
                    record.get("Targeted_Aspects") or []
                ):
                    if "Sarcasm" in aspect:
                        continue
                    sent = aspect.get("Target_Sentence", "")
                    if sent:
                        work_items.append((record, idx, sent))

            # Comment aspects — prepend parent context.
            for comment in record.get("Comments") or []:
                if comment.get("Subjectivity") != "subjective":
                    continue
                for idx, aspect in enumerate(
                    comment.get("Targeted_Aspects") or []
                ):
                    if "Sarcasm" in aspect:
                        continue
                    sent = aspect.get("Target_Sentence", "")
                    if sent:
                        model_input = (
                            f"{parent_context} | {sent}"
                            if parent_context
                            else sent
                        )
                        work_items.append((comment, idx, model_input))

        if not work_items:
            logger.info("SarcasmDetector: no aspects to classify.")
            return records

        # Deduplicate model inputs for batching efficiency.
        unique_inputs = list({item[2] for item in work_items})
        logger.info(
            f"SarcasmDetector: classifying {len(unique_inputs)} unique "
            f"inputs across {len(records)} records..."
        )
        cache = self._batch_classify(unique_inputs)

        # Write Sarcasm back into each aspect dict.
        for container, idx, model_input in work_items:
            container["Targeted_Aspects"][idx]["Sarcasm"] = cache.get(
                model_input,
                {"Is_Sarcastic": False, "Sarcasm_Confidence": 0.0},
            )

        logger.info("SarcasmDetector: corpus detection complete.")
        return records

    # Container-level processing (used by detect_record)

    def _process_container(
        self, container: dict, parent_context: str
    ) -> None:
        """
        Classify all Target_Sentences in container's Targeted_Aspects.

        parent_context is prepended to each sentence when non-empty (used
        for comments). Posts always pass an empty string.
        """
        aspects: list[dict] = container.get("Targeted_Aspects") or []
        if not aspects:
            return

        # Build model input strings, injecting parent context where present.
        model_inputs: list[str] = []
        for aspect in aspects:
            sent = aspect.get("Target_Sentence", "")
            model_input = (
                f"{parent_context} | {sent}" if parent_context else sent
            )
            model_inputs.append(model_input)

        unique_inputs = list({s for s in model_inputs if s})
        cache = self._batch_classify(unique_inputs)

        for aspect, model_input in zip(aspects, model_inputs):
            aspect["Sarcasm"] = cache.get(
                model_input,
                {"Is_Sarcastic": False, "Sarcasm_Confidence": 0.0},
            )

    # Classification

    def _batch_classify(self, inputs: list[str]) -> dict[str, dict]:
        """
        Classify a list of unique model-input strings in batches.

        Returns {input_text: {"Is_Sarcastic": bool, "Sarcasm_Confidence": float}}.
        Sarcasm_Confidence is always P(Sarcastic).
        """
        if not inputs:
            return {}

        self._load_pipeline()
        truncated = [s[:MAX_SENTENCE_CHARS] for s in inputs]
        raw_results: list[dict] = []

        for i in range(0, len(truncated), self.batch_size):
            batch = truncated[i : i + self.batch_size]
            try:
                batch_out = self._pipeline(batch)
                if isinstance(batch_out, dict):
                    batch_out = [batch_out]
                raw_results.extend(batch_out)
            except Exception as e:
                logger.error(
                    f"SarcasmDetector: batch error at offset {i}: {e}"
                )
                raw_results.extend(
                    [{"label": "LABEL_0", "score": 1.0}] * len(batch)
                )

        cache: dict[str, dict] = {}
        for original, result in zip(inputs, raw_results):
            raw_label = result.get("label", "LABEL_0")
            human_label = self.label_map.get(raw_label, "Not Sarcastic")
            raw_score = float(result.get("score", 0.0))
            is_sarcastic = human_label == "Sarcastic"
            # Always report confidence as P(Sarcastic).
            sarcasm_prob = raw_score if is_sarcastic else round(1.0 - raw_score, 4)
            cache[original] = {
                "Is_Sarcastic":       is_sarcastic,
                "Sarcasm_Confidence": round(sarcasm_prob, 4),
            }

        return cache

    def _load_pipeline(self) -> None:
        """Lazy-load the HuggingFace classification pipeline."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(
                f"SarcasmDetector: loading '{self.model_name}'..."
            )
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )
            logger.info("SarcasmDetector: model loaded.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load sarcasm model '{self.model_name}': {e}\n"
                "Install with: pip install transformers torch"
            ) from e

def load_json(path: str | Path) -> list[dict] | dict:
    """Load a JSON file; returns a list or a single dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dataset_path = '../../../data/my_test/sacarstic_input.json'

    data = load_json(dataset_path)
    records: list[dict] = data if isinstance(data, list) else [data]

    detector = SarcasmDetector(model_name=DEFAULT_MODEL)
    detector.detect_corpus(records)

    result_to_print = records if len(records) > 1 else records[0]
    print(json.dumps(result_to_print, indent=4))