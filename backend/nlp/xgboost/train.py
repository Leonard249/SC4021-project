"""
train.py
SC4021 — XGBoost Training & Evaluation

Loads data/results/classified_eval_new.json, extracts features,
performs an 80/20 stratified train/test split, trains XGBoost with
5-fold cross-validation on the training set, then evaluates on the
held-out test set.

Outputs
-------
  data/models/xgb_polarity.pkl      — trained XGBoost model (joblib)
  data/models/feature_names.json    — ordered feature name list

Usage
-----
    python train.py
    python train.py --input path/to/other.json
    python train.py --no-save        # skip saving the model
"""

import argparse
import json
import sys
import joblib
import numpy as np
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Path setup — resolve sibling feature_extractor regardless of CWD
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from feature_extractor import FeatureExtractor, LABEL_TO_INT, INT_TO_LABEL

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_PROJECT = _HERE.parents[2]   # SC4021-project/
DEFAULT_INPUT  = _PROJECT / "data" / "results" / "classified_eval_new.json"
DEFAULT_MODEL     = _PROJECT / "data" / "models" / "xgb_polarity.pkl"
DEFAULT_FEATS     = _PROJECT / "data" / "models" / "feature_names.json"
DEFAULT_MATRIX    = _PROJECT / "data" / "models" / "features.npz"
DEFAULT_SENT_PROBS = _PROJECT / "data" / "models" / "sentiment_probs.json"

CLASSES = ["positive", "negative", "neutral", "irrelevant"]
SEP = "─" * 68


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_metrics(y_true, y_pred, split_name: str):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0
    )

    print(f"\n{SEP}")
    print(f"  {split_name} RESULTS")
    print(SEP)
    print(f"\n  {'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("  " + "─" * 54)
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:<14} {prec[i]:>10.4f} {rec[i]:>8.4f} {f1[i]:>8.4f} {support[i]:>9}")
    print("  " + "─" * 54)
    print(f"  {'Overall accuracy':<41} {acc:>10.4f}")


def print_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    col_w = 13
    true_pred = "True \\ Pred"
    header = f"  {true_pred:<14}" + "".join(f"{c:>{col_w}}" for c in CLASSES) + f"{'Total':>{col_w}}"
    print(f"\n{SEP}")
    print("  CONFUSION MATRIX  (rows = true,  cols = predicted)")
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


def print_feature_importance(model, feature_names: list[str], top_n: int = 20):
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print(f"\n{SEP}")
    print(f"  TOP {top_n} FEATURE IMPORTANCES")
    print(SEP)
    for name, score in ranked[:top_n]:
        bar = "█" * int(score * 400)
        print(f"  {name:<35} {score:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Cross-validation on training set
# ---------------------------------------------------------------------------

def cross_validate(X_train, y_train, params: dict, n_splits: int = 5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs = []

    print(f"\n{SEP}")
    print(f"  {n_splits}-FOLD CROSS-VALIDATION  (on training set)")
    print(SEP)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accs.append(acc)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, labels=list(range(4)), zero_division=0
        )
        neg_f1 = f1[LABEL_TO_INT["negative"]]
        print(
            f"  Fold {fold}: accuracy={acc:.4f}  "
            f"neg_f1={neg_f1:.4f}  "
            f"[{' '.join(f'{f:.2f}' for f in f1)}]"
        )

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n  Mean CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return mean_acc


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    input_path: Path = DEFAULT_INPUT,
    model_path: Path = DEFAULT_MODEL,
    feats_path: Path = DEFAULT_FEATS,
    matrix_path: Path = DEFAULT_MATRIX,
    sentiment_probs_path: Path = DEFAULT_SENT_PROBS,
    save: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
):
    # ── 1. Load data ─────────────────────────────────────────────────────
    print(f"\nLoading records from: {input_path}")
    records = load_records(input_path)
    print(f"  Total records: {len(records)}")

    # ── 2. Extract features (or load cached matrix) ───────────────────────
    sent_path = sentiment_probs_path if sentiment_probs_path.exists() else None
    if sent_path:
        print(f"\nSentiment probs: {sent_path}")
    else:
        print("\nSentiment probs: not found — Group I features will be zeros")

    extractor = FeatureExtractor(sentiment_probs_path=sent_path)
    feature_names = extractor.feature_names()

    if matrix_path.exists():
        print(f"\nLoading cached feature matrix from: {matrix_path}")
        npz = np.load(matrix_path)
        X, y = npz["X"], npz["y"]
    else:
        print("\nExtracting features...")
        X, y = extractor.extract_corpus(records)
        if save:
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(matrix_path, X=X, y=y)
            print(f"  Feature matrix cached → {matrix_path}")

    print(f"  Feature matrix: {X.shape}  (records × features)")
    print(f"  Label distribution: {dict(Counter(INT_TO_LABEL[i] for i in y))}")

    # ── 3. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"\n  Train: {len(X_train)} records  |  Test: {len(X_test)} records")

    # ── 4. XGBoost hyperparameters ────────────────────────────────────────
    xgb_params = dict(
        objective="multi:softmax",
        num_class=4,
        n_estimators=400,
        max_depth=3,           # reduced from 5 — main fix for overfitting
        learning_rate=0.05,
        subsample=0.7,         # reduced from 0.8
        colsample_bytree=0.6,  # reduced from 0.8
        min_child_weight=5,    # increased from 3
        gamma=0.2,             # increased from 0.1
        reg_alpha=1.0,         # increased from 0.1
        reg_lambda=2.0,        # increased from 1.0
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    # ── 5. Cross-validation on training set ───────────────────────────────
    cross_validate(X_train, y_train, xgb_params)

    # ── 6. Train final model on full training set ─────────────────────────
    print(f"\n{SEP}")
    print("  TRAINING FINAL MODEL  (full training set)")
    print(SEP)

    final_model = XGBClassifier(**xgb_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ── 7. Evaluate on test set ───────────────────────────────────────────
    y_pred_test = final_model.predict(X_test)
    print_metrics(y_test, y_pred_test, "TEST SET")
    print_confusion(y_test, y_pred_test)

    # Also evaluate on training set (to check for overfitting)
    y_pred_train = final_model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"\n  Train accuracy (overfitting check): {train_acc:.4f}")

    # ── 8. Feature importance ─────────────────────────────────────────────
    print_feature_importance(final_model, feature_names, top_n=20)

    # ── 9. Save model ─────────────────────────────────────────────────────
    if save:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_path)
        feats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(feats_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        print(f"\n{SEP}")
        print(f"  Model saved  → {model_path}")
        print(f"  Features     → {feats_path}")
        print(SEP)

    return final_model, extractor


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost polarity classifier on pipeline output"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to classified JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Where to save the model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the model and feature matrix to disk",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data held out for testing (default: 0.20)",
    )
    parser.add_argument(
        "--matrix-out",
        type=Path,
        default=DEFAULT_MATRIX,
        help=f"Where to cache the feature matrix (default: {DEFAULT_MATRIX})",
    )
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Ignore cached feature matrix and re-extract from source JSON",
    )
    parser.add_argument(
        "--sentiment-probs",
        type=Path,
        default=DEFAULT_SENT_PROBS,
        help=f"Path to sentiment_probs.json (default: {DEFAULT_SENT_PROBS})",
    )
    args = parser.parse_args()

    if args.reextract and args.matrix_out.exists():
        args.matrix_out.unlink()
        print(f"Deleted cached matrix: {args.matrix_out}")

    train(
        input_path=args.input,
        model_path=args.model_out,
        matrix_path=args.matrix_out,
        sentiment_probs_path=args.sentiment_probs,
        save=not args.no_save,
        test_size=args.test_size,
    )
