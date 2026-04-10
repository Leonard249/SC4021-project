"""
train.py
SC4021 — DistilBERT Fine-tuning for Polarity Classification

Fine-tunes cardiffnlp/twitter-roberta-base-sentiment-latest on the 1200 labeled records from
classified_eval_new.json. Uses the same 80/20 stratified split as the
XGBoost baseline (random_state=42) for a fair accuracy comparison.

The symbolic pipeline's Normalized_Text is used as input — the pipeline
serves as a preprocessing stage that makes the fine-tuning more
data-efficient by removing noise before the transformer sees the text.

Outputs
-------
    data/models/roberta_polarity/   — saved model + tokenizer
    data/models/roberta_results.json — final test metrics

Usage
-----
    python train.py
    python train.py --epochs 5 --batch-size 8 --max-length 128
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[2]

sys.path.insert(0, str(_HERE))
from dataset import PolarityDataset, LABEL_TO_INT, INT_TO_LABEL

DEFAULT_INPUT   = _PROJECT / "data" / "results" / "classified_eval_new.json"
DEFAULT_OUT_DIR = _PROJECT / "data" / "models" / "roberta_polarity"
DEFAULT_RESULTS = _PROJECT / "data" / "models" / "roberta_results.json"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CLASSES    = ["positive", "negative", "neutral", "irrelevant"]
SEP        = "─" * 68

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Called by Trainer after each evaluation epoch."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(4)), zero_division=0, average=None
    )
    return {
        "accuracy":    round(acc, 4),
        "f1_positive": round(f1[0], 4),
        "f1_negative": round(f1[1], 4),
        "f1_neutral":  round(f1[2], 4),
        "f1_irrelevant": round(f1[3], 4),
    }


def print_test_results(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0
    )

    print(f"\n{SEP}")
    print("  TEST SET RESULTS")
    print(SEP)
    print(f"\n  {'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("  " + "─" * 54)
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<14} {prec[i]:>10.4f} {rec[i]:>8.4f} {f1[i]:>8.4f} {support[i]:>9}")
    print("  " + "─" * 54)
    print(f"  {'Overall accuracy':<41} {acc:>10.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    col_w = 13
    true_pred = "True \\ Pred"
    header = f"  {true_pred:<14}" + "".join(f"{c:>{col_w}}" for c in CLASSES) + f"{'Total':>{col_w}}"
    print(f"\n{SEP}")
    print("  CONFUSION MATRIX")
    print(SEP)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, cls in enumerate(CLASSES):
        row = cm[i]
        print(
            f"  {cls:<14}"
            + "".join(f"{v:>{col_w}}" for v in row)
            + f"{row.sum():>{col_w}}"
        )
    print("  " + "─" * (len(header) - 2))
    col_totals = cm.sum(axis=0)
    print(
        f"  {'Total':<14}"
        + "".join(f"{v:>{col_w}}" for v in col_totals)
        + f"{col_totals.sum():>{col_w}}"
    )

    return {
        "accuracy": acc,
        "per_class": {
            CLASSES[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(4)
        },
    }


# ---------------------------------------------------------------------------
# Weighted loss Trainer
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights to the cross-entropy loss."""

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # shape (num_classes,)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = self.class_weights
        if weights is not None:
            weights = weights.to(logits.device)

        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    input_path: Path = DEFAULT_INPUT,
    out_dir: Path = DEFAULT_OUT_DIR,
    results_path: Path = DEFAULT_RESULTS,
    model_name: str = MODEL_NAME,
    max_length: int = 512,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    test_size: float = 0.20,
    random_state: int = 42,
    save: bool = True,
):
    # ── Device info ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\nDevice: {device.upper()}")
    if device == "mps":
        print("  Apple MPS detected — using Metal GPU acceleration.")
    elif device == "cpu":
        print("  Warning: no GPU detected. Training on CPU will be slow (~1-3 min/epoch).")
        print("  Consider reducing --epochs or --max-length if time is tight.\n")

    # ── Load records ──────────────────────────────────────────────────────
    print(f"Loading records from: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    # Filter to labeled records only
    labeled = [r for r in records if (r.get("label") or "").lower() in LABEL_TO_INT]
    print(f"  {len(labeled)} labeled records")
    print(f"  Distribution: {dict(Counter(r['label'] for r in labeled))}")

    # ── Train / test split (same seed as XGBoost for fair comparison) ─────
    labels_for_split = [LABEL_TO_INT[r["label"].lower()] for r in labeled]
    train_records, test_records = train_test_split(
        labeled,
        test_size=test_size,
        stratify=labels_for_split,
        random_state=random_state,
    )
    print(f"\n  Train: {len(train_records)} records  |  Test: {len(test_records)} records")

    # ── Class weights (inverse frequency on training set) ─────────────────
    train_label_counts = Counter(LABEL_TO_INT[r["label"].lower()] for r in train_records)
    total_train = sum(train_label_counts.values())
    class_weights = torch.tensor(
        [total_train / (4 * train_label_counts.get(i, 1)) for i in range(4)],
        dtype=torch.float32,
    )
    print(f"\n  Class weights: { {CLASSES[i]: round(float(class_weights[i]), 3) for i in range(4)} }")

    # ── Tokenizer & datasets ──────────────────────────────────────────────
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = PolarityDataset(train_records, tokenizer, max_length)
    test_dataset  = PolarityDataset(test_records,  tokenizer, max_length)
    print(f"  max_length={max_length} tokens  |  train={len(train_dataset)}  test={len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nLoading model: {model_name}  (4 output classes)")
    label2id = {cls: i for i, cls in enumerate(CLASSES)}
    id2label = {i: cls for i, cls in enumerate(CLASSES)}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # ── Training arguments ────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=random_state,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
    )

    print(f"\n{SEP}")
    print(f"  TRAINING  ({num_epochs} epochs max, early stopping patience=3)")
    print(SEP)
    trainer.train()

    # ── Evaluate on test set ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FINAL EVALUATION ON TEST SET")
    print(SEP)

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    results = print_test_results(y_true, y_pred)

    # ── Save ──────────────────────────────────────────────────────────────
    if save:
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n{SEP}")
        print(f"  Model saved    → {out_dir}")
        print(f"  Results saved  → {results_path}")
        print(SEP)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on pipeline-output labeled records"
    )
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"HuggingFace model name (default: {MODEL_NAME})")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Token limit (max 512; reduce to 128-256 on CPU)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    train(
        input_path=args.input,
        out_dir=args.out_dir,
        model_name=args.model,
        max_length=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save=not args.no_save,
    )
