# Experiment Reproduction Guide

This document explains how to reproduce the classification model results for the SC4021 project.
All scripts are deterministic (`random_state=42`) — given the same input data they will produce identical models and metrics.

## Prerequisites

- Conda environment `sc4021` with all dependencies installed
- `data/results/classified_eval_new.json` — **must be obtained manually** (too large for git). Place it at `data/results/classified_eval_new.json` relative to the project root before running any scripts.

---

## XGBoost Classifier

**Location:** `backend/nlp/xgboost/`

### Step 1 — Pre-compute RoBERTa sentiment probabilities

This is a one-time step that generates Group I features used by XGBoost.

```bash
conda run -n sc4021 python backend/nlp/xgboost/precompute_sentiment.py
```

Output: `data/models/sentiment_probs.json`

Model used: `cardiffnlp/twitter-roberta-base-sentiment` (downloaded automatically from HuggingFace)

### Step 2 — Train XGBoost

```bash
conda run -n sc4021 python backend/nlp/xgboost/train.py
```

Output:
- `data/models/xgb_polarity.pkl` — trained XGBoost model
- `data/models/feature_names.json` — ordered feature name list
- `data/models/features.npz` — cached feature matrix (speeds up re-runs)

**Useful flags:**

| Flag | Description |
|------|-------------|
| `--reextract` | Delete cached feature matrix and re-extract from source JSON |
| `--no-save` | Run without saving any model files |
| `--test-size 0.20` | Fraction held out for testing (default: 0.20) |

### Feature Groups (50 features total)

| Group | Description |
|-------|-------------|
| A | Subjectivity signals (Stage 5 output) |
| B | Document-level VADER sentiment |
| C | Aspect-level signals (Stage 8) |
| D | Sarcasm signals (Stage 7) |
| E | NER signals (Stage 4) |
| F | Text structure (word count, sentence count) |
| G | Source bucket (social / blog / qa / article) |
| H | Per-aspect VADER on Target_Sentence |
| I | Pre-computed RoBERTa sentiment probabilities |

---

## Transformer Classifier (RoBERTa)

**Location:** `backend/nlp/transformer/`

### Train

```bash
conda run -n sc4021 python backend/nlp/transformer/train.py
```

Output:
- `data/models/roberta_polarity/` — saved model and tokenizer
- `data/models/roberta_results.json` — final test metrics

**Useful flags:**

| Flag | Description |
|------|-------------|
| `--max-length 256` | Reduce token limit (default: 512; use 256 on CPU for speed) |
| `--epochs 10` | Max training epochs (default: 10, early stopping patience=3) |
| `--batch-size 16` | Batch size per device (default: 32) |
| `--lr 2e-5` | Learning rate (default: 2e-5) |
| `--no-save` | Run without saving model files |

### Design Choices

- **Base model:** `cardiffnlp/twitter-roberta-base-sentiment-latest` — sentiment-aware checkpoint, fine-tuned on Twitter data; well-suited for short opinionated text
- **Input:** `Title + "\n\n" + Normalized_Text` — the symbolic pipeline's cleaned text with the post title prepended
- **Class weights:** Inverse-frequency weights computed from the training split are applied to the cross-entropy loss to address class imbalance
- **Split:** 80/20 stratified train/test split (`random_state=42`) — identical to XGBoost for fair comparison
- **Early stopping:** Patience of 3 epochs on validation accuracy

---

## Label Encoding

| Label | Integer |
|-------|---------|
| positive | 0 |
| negative | 1 |
| neutral | 2 |
| irrelevant | 3 |

---

## Reproducibility Notes

- Both models use `random_state=42` with the same 80/20 stratified split, so test sets are identical.
- XGBoost feature extraction is cached to `data/models/features.npz` after the first run. Delete it or use `--reextract` if the source JSON changes.
- HuggingFace model weights are downloaded automatically and cached locally (`~/.cache/huggingface/`).
- `data/models/` is not committed to git. Run the scripts above to regenerate all artifacts.
