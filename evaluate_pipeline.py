"""
evaluate_pipeline.py
--------------------
Evaluates the NLP pipeline output in data/results/classified_eval_new.json
against the human-annotated ground truth labels.

Predicted label derivation:
  - Subjectivity == "Irrelevant"  →  predicted = "irrelevant"
  - Otherwise                     →  predicted = Overall_Document_Polarity

Outputs:
  - Per-class metrics: Precision, Recall, F1, Support
  - Overall Accuracy
  - Confusion Matrix (true label as rows, predicted label as columns)
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
INPUT_PATH  = BASE / "data" / "results" / "classified_eval_new.json"

CLASSES = ["positive", "negative", "neutral", "irrelevant"]

# ─── Load ─────────────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── Label resolution ─────────────────────────────────────────────────────────

def get_predicted(record: dict) -> str:
    """
    Derive the pipeline's predicted label from a record.
    Records marked Irrelevant by the subjectivity stage never reach polarity.
    """
    if record.get("Subjectivity") == "Irrelevant":
        return "irrelevant"
    return (record.get("Overall_Document_Polarity") or "neutral").lower()


def get_true(record: dict) -> str:
    return (record.get("label") or "").lower()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list[str], y_pred: list[str], classes: list[str]
) -> dict:
    """Compute per-class and overall metrics without sklearn."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    metrics = {}
    for cls in classes:
        support   = tp[cls] + fn[cls]
        precision = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        recall    = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        accuracy  = tp[cls] / support if support > 0 else 0.0
        metrics[cls] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "support":   support,
            "accuracy":  accuracy,
        }

    overall_accuracy = sum(tp.values()) / len(y_true) if y_true else 0.0
    metrics["__overall__"] = overall_accuracy
    return metrics


def confusion_matrix(
    y_true: list[str], y_pred: list[str], classes: list[str]
) -> dict[str, dict[str, int]]:
    """Return {true_label: {pred_label: count}}."""
    matrix = {c: {p: 0 for p in classes} for c in classes}
    for true, pred in zip(y_true, y_pred):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1
        elif true in matrix:
            matrix[true][pred] = matrix[true].get(pred, 0) + 1
    return matrix


# ─── Pretty printers ──────────────────────────────────────────────────────────

SEP = "─" * 70

def print_per_class(metrics: dict, classes: list[str]) -> None:
    print(f"\n{'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9} {'Accuracy':>10}")
    print("─" * 62)
    for cls in classes:
        m = metrics[cls]
        print(
            f"  {cls:<12} {m['precision']:>10.4f} {m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} {m['support']:>9} {m['accuracy']:>10.4f}"
        )
    print("─" * 62)
    print(f"  {'Overall accuracy':<43} {metrics['__overall__']:>10.4f}")


def print_confusion(matrix: dict, classes: list[str]) -> None:
    col_w = 13
    # Header row
    true_pred_label = "True \\ Pred"
    header = f"  {true_pred_label:<14}" + "".join(f"{c:>{col_w}}" for c in classes) + f"{'Total':>{col_w}}"
    print(header)
    print("─" * len(header))

    for true_cls in classes:
        row_vals  = [matrix[true_cls].get(pred_cls, 0) for pred_cls in classes]
        row_total = sum(row_vals)
        row = f"  {true_cls:<14}" + "".join(f"{v:>{col_w}}" for v in row_vals) + f"{row_total:>{col_w}}"
        print(row)

    # Column totals
    print("─" * len(header))
    col_totals = [sum(matrix[t].get(p, 0) for t in classes) for p in classes]
    grand      = sum(col_totals)
    print(
        f"  {'Total':<14}" +
        "".join(f"{v:>{col_w}}" for v in col_totals) +
        f"{grand:>{col_w}}"
    )


def print_misclassification_detail(
    y_true: list[str], y_pred: list[str],
    records: list[dict], classes: list[str]
) -> None:
    """For each class, list where misclassified samples went."""
    print()
    for cls in classes:
        wrong = [(t, p) for t, p in zip(y_true, y_pred) if t == cls and t != p]
        if not wrong:
            print(f"  {cls.upper():<14}  No misclassifications ✓")
            continue
        counts = Counter(p for _, p in wrong)
        total  = len(wrong)
        support = sum(1 for t in y_true if t == cls)
        print(f"  {cls.upper():<14}  {total} misclassified / {support} total  "
              f"({100*total/support:.1f}% error rate)")
        for pred_cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(20 * cnt / total)
            print(f"    → predicted '{pred_cls}': {cnt:>4}  {bar}")
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    records = load(INPUT_PATH)
    print(f"\nLoaded {len(records)} records from {INPUT_PATH.name}")

    y_true = [get_true(r)      for r in records]
    y_pred = [get_predicted(r) for r in records]

    # Warn about any unexpected labels
    unexpected_true = set(y_true) - set(CLASSES)
    unexpected_pred = set(y_pred) - set(CLASSES)
    if unexpected_true:
        print(f"⚠  Unexpected true labels (excluded from metrics): {unexpected_true}")
    if unexpected_pred:
        print(f"⚠  Unexpected predicted labels: {unexpected_pred}")

    metrics = compute_metrics(y_true, y_pred, CLASSES)
    matrix  = confusion_matrix(y_true, y_pred, CLASSES)

    # ── Per-class metrics ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PER-CLASS METRICS")
    print(SEP)
    print_per_class(metrics, CLASSES)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CONFUSION MATRIX  (rows = true label,  cols = predicted label)")
    print(SEP)
    print_confusion(matrix, CLASSES)

    # ── Misclassification breakdown ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  MISCLASSIFICATION BREAKDOWN  (per true class)")
    print(SEP)
    print_misclassification_detail(y_true, y_pred, records, CLASSES)

    print(SEP)


if __name__ == "__main__":
    main()
