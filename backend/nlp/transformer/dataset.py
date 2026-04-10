"""
dataset.py
SC4021 — PyTorch Dataset for DistilBERT fine-tuning

Wraps the pipeline-output records into a format consumed by the
HuggingFace Trainer. Input text is Normalized_Text (already cleaned
by the symbolic pipeline stages 1-4).

Label encoding (must match XGBoost for fair comparison):
    0 → positive
    1 → negative
    2 → neutral
    3 → irrelevant
"""

import re
import torch
from torch.utils.data import Dataset

LABEL_TO_INT = {"positive": 0, "negative": 1, "neutral": 2, "irrelevant": 3}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def clean_text(text: str) -> str:
    """
    Strip pipeline artifacts before feeding to the transformer.

    The symbolic pipeline adds <CODE> placeholders and [emoticon] tokens.
    We convert <CODE> to the word 'code' (semantically meaningful) and
    remove emoticon bracket tokens since the transformer's vocabulary
    won't have entries for "[smiling face]" etc.
    """
    text = re.sub(r"<CODE>", " code ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)   # remove [emoticon text]
    text = re.sub(r"\s+", " ", text).strip()
    return text or "no content"


class PolarityDataset(Dataset):
    """
    Parameters
    ----------
    records   : list of pipeline-output dicts (must have 'Normalized_Text' and 'label')
    tokenizer : HuggingFace tokenizer
    max_length: token limit (DistilBERT max = 512; use 256 for speed on CPU)
    """

    def __init__(self, records: list[dict], tokenizer, max_length: int = 256):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        self.texts  = []
        self.labels = []

        for record in records:
            label_str = (record.get("label") or "").lower()
            if label_str not in LABEL_TO_INT:
                continue
            title = clean_text(record.get("Title", "") or "")
            body  = clean_text(record.get("Normalized_Text", "") or "")
            text  = f"{title}\n\n{body}" if title else body
            self.texts.append(text)
            self.labels.append(LABEL_TO_INT[label_str])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }
