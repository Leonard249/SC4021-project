# SC4021 Information Retrieval — Group 33

## Project Overview

This project builds a comprehensive information retrieval and sentiment analysis system. We crawl developer discussions from four heterogeneous platforms, construct a unified search index with advanced filtering capabilities, and develop a multi-layer NLP classification pipeline.

---

## Repository Structure

```
SC4021-project/
│
├── reddit_crawlers/          # Reddit scraper
├── hn-crawler/               # HackerNews full pipeline (crawl → clean → filter)
├── x_scraper/                # X / Twitter scraper (twikit)
├── articles_extraction/      # Tech blog / Medium article scraper (trafilatura)
│
├── backend/
│   ├── nlp/
│   │   ├── classifier.py         # End-to-end pipeline orchestrator
│   │   ├── syntactics/           # Microtext normalisation, SBD, POS tagging
│   │   ├── semantics/            # NER tagging, subjectivity detection
│   │   └── pragmatics/           # Aspect extraction, sarcasm, polarity ensemble
│   │       └── length_routing/   # VADER+SenticNet (short) / Transformer (long)
│   ├── utils/                    # spaCy utilities, shared helpers
│   ├── lexicons/                 # Custom lexicons
│   └── results/                  # Evaluation outputs, distribution plots
│
├── random_check_75/          # Random-sample label verification tool
│   ├── index.html            # Browser-based labelling UI
│   ├── server.py             # Flask server for the UI
│   ├── sample75.json         # 75-sample random draw for human review
│   └── check_labels.py       # Discrepancy analysis script
│
├── enhanced-search/          # Indexing + Search engine (see enhanced-search/README.md)
│   ├── search_engine.py         # Flask REST API
│   ├── search_ui_enhanced.html  # Web UI
│   ├── prepare_output_data.py   # Embedding + field extraction from json
│   ├── index_data.py            # Bulk Elasticsearch indexing
│   └── generate_submission_data.py  # Query experiments + submission exports
│
├── data/                     # Shared data assets
├── diagrams/                 # Architecture and pipeline diagrams
├── images/                   # Report figures
└── README.md                 # This file
```

---

## Component Summaries

### 1. Data Collection

| Source | Folder | Tool | Target |
|--------|--------|------|--------|
| Reddit | `reddit_crawlers/` | PRAW / Pushshift | Posts + comments on AI coding tools |
| HackerNews | `hn-crawler/` | HN Algolia API | Stories + comments, full pipeline |
| X / Twitter | `x_scraper/` | twikit (async) | Tweets on AI coding keywords |
| Tech Articles | `articles_extraction/` | trafilatura + newspaper3k | Medium, blogs, dev.to |

All sources produce documents conforming to the shared JSON schema (see below).

---

### 2. NLP Classification Pipeline (`backend/nlp/`)

**Orchestrator:** `classifier.py` — runs all 8 stages end-to-end.

| Stage | Module | What it does |
|-------|--------|--------------|
| 1 | `syntactics/microtextnorm.py` | Clean HTML/Markdown, normalise URLs, fix casing |
| 2 | `syntactics/sbd.py` | Sentence boundary disambiguation |
| 3 | `syntactics/pos_tagger.py` | POS tagging + lemmatization |
| 4 | `semantics/ner_tagger.py` | Named entity recognition (AI_TOOL, ORG, PL, EDITOR, FRAMEWORK, TECH_CONCEPT) |
| 5 | `semantics/subjectivity_detector.py` | Hybrid lexicon + transformer subjectivity scoring |
| 6 | `pragmatics/aspect_extractor.py` | Targeted aspect extraction per sentence |
| 7 | `pragmatics/sarcasm_detector.py` | Sarcasm detection per aspect sentence |
| 8 | `pragmatics/ensemble.py` | Length-aware routing + polarity ensemble (final label) |

**Length-aware routing (Stage 8):**
- `< 60 words` → VADER + SenticNet (`sentic_vader.py`)
- `60–400 words` → Transformer direct pass (`transformer_polarity.py`)
- `> 400 words` → Chunked transformer with aggregation

**Output fields per document:** `Overall_Document_Polarity`, `Subjectivity`, `Subjectivity_Score`, `NER_Tags`, `Targeted_Aspects`, `Aspect_Sentiments`, sarcasm flags.

---

### 3. Evaluation

**Random 75-sample check (`random_check_75/`):**
- Flask + browser UI (`index.html`) for human review of classifier output
- Reviewer accepts or rejects each predicted label; corrected label recorded
- `check_labels.py` computes accept/reject/label-change stats

**Classifier evaluation notebooks (`backend/nlp/`):**
- `eval_dataset.ipynb` — builds ground-truth evaluation set
- `eval_notebook.ipynb` — computes Precision, Recall, F1 against human labels
- Output plots: `ground_truth_distribution.png`, `predicted_distribution.png`, `source_distribution.png`

---

### 4. Indexing & Search Engine (`enhanced-search/`)

See [enhanced-search/README.md](enhanced-search/README.md) for full details.

**Quick summary:**
- **69,047 documents** indexed into Elasticsearch 7.17.4
- **3 search methods:** Keyword (BM25), Semantic (384-dim embeddings), Hybrid (α=0.5 fusion)
- **4 innovations:** Timeline Search, Multifaceted Search, Sentiment-Aware Analytics, Enhanced Visualisations
- **14 REST API endpoints** via Flask on port 5001
- **5-tab web UI** — Results, Timeline, Multifaceted, Sentiment Analysis, Advanced Visualisations

---

## How to Run

### Datasets

The datasets are provided in the **dataset zip file**:

| File | Location in zip | Description |
|------|-----------------|-------------|
| `classified_eval.json` | `dataset zip → pipeline results → classified_eval.json` | Fully classified dataset — use this for the search engine |
| `raw_data.json` | `dataset zip → crawled raw data → raw_data.json` | Raw crawled data — use this as input to the classifier |

---

### A. Running the Classifier

**Prerequisites:** Install dependencies first.

```bash
pip install -r requirements.txt
```

**Steps:**

1. Open `backend/nlp/classifier.py`
2. Near the top, set the input and output file paths:
   ```python
   INPUT_PATH  = "path/to/raw_data.json"        # raw crawled data
   OUTPUT_PATH = "path/to/classified_eval.json"  # where to write results
   ```
3. Run the classifier:
   ```bash
   python backend/nlp/classifier.py
   ```

The classifier runs all 8 pipeline stages (normalisation → POS → NER → subjectivity → aspect extraction → sarcasm → polarity) and writes the fully annotated output to the path you specified.

---

### B. Running the Search Engine

**Prerequisites:**
- Python 3.10+
- Elasticsearch 7.17.4 running on `localhost:9200` (see [enhanced-search/README.md](enhanced-search/README.md) for install options)
- `classified_eval.json` placed in the `enhanced-search/` directory

**Steps:**

```bash
cd enhanced-search

# 1. Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Generate embeddings + prepare index data  (~15-30 min)
python prepare_output_data.py

# 3. Bulk-index into Elasticsearch  (~5 min)
python index_data.py

# 4. Start the search API
python search_engine.py

# 5. Open the UI in your browser
open search_ui_enhanced.html
```

The search engine runs at `http://localhost:5001`. Open `search_ui_enhanced.html` directly in a browser to access the UI.

For full details on all API endpoints and UI tabs, see [enhanced-search/README.md](enhanced-search/README.md).

---

## System Architecture

```
[ Raw Input Text from Reddit/HackerNews/Medium/X ]
│
▼
┌─────────────────────────────────────────────────────┐
│  LAYER 1: SYNTACTICS LAYER (Preprocessing)          │
│  - Microtext Normalization (Clean HTML/Markdown)    │
│  - Sentence Boundary Disambiguation                 │
│  - POS Tagging & Lemmatization (Prep for SenticNet) │
└────────────────────┬────────────────────────────────┘
                     │ (Structured, Clean Text)
                     ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 2: SEMANTICS LAYER (Stage 1: Subjectivity)   │
│  - Concept Extraction (Find Entities/Phrases)       │
│  - Rule-Based Subjectivity Detection:               │
│    ├─ TextBlob subjectivity score (soft signal)     │
│    ├─ First-person pronoun density                  │
│    ├─ Hedging language detection                    │
│    └─ Source type prior (Reddit = subjective)       │
└─────────┬──────────────────────────────┬────────────┘
          │                              │
 [ OBJECTIVE / NEUTRAL ]    [ SUBJECTIVE / OPINIONATED ]
  Record as Neutral &                   │
  Discard                               │ Continue
                                        ▼
                     ┌─────────────────────────────────────────────────────┐
                     │  LAYER 3: PRAGMATICS LAYER (Polarity & Context)     │
                     │  - Sarcasm Detection (e.g., /s tag checking)        │
                     │                                                     │
                     │    ──────────  Length-Aware Routing ─────────       │
                     │                                                     │
                     │  Word count < 60       → SHORT path                 │
                     │  60 ≤ Words ≤ 400      → MEDIUM path                │
                     │  Word count > 400      → LONG path (chunked)        │
                     │                                                     │
                     │  ┌──────────────┬─────────────────┬──────────────┐  │
                     │  │  SHORT path  │   MEDIUM path   │  LONG path   │  │
                     │  │      │       │        │        │      │       │  │
                     │  │      ▼       │        ▼        │      ▼       │  │
                     │  │  VADER +     │  Transformer    │  Chunk →     │  │
                     │  │  SenticNet   │  (direct)       │  classify    │  │
                     │  │  concepts    │                 │  each chunk  │  │
                     │  │              │                 │  → aggregate │  │
                     │  └──────────────┴─────────────────┴──────────────┘  │
                     │                         │                           │
                     │                         ▼                           │
                     │           ┌──────────────────────────┐              │
                     │           │  Ensemble / Aggregation  │              │
                     │           │  Weighted majority vote  │              │
                     │           │  or average confidence   │              │
                     │           └──────────────────────────┘              │
                     └─────────────────────────┬───────────────────────────┘
                                               │
                                               ▼
                          Polarity: Positive / Negative + Confidence Score
```

---

## Sample JSON Schema

```json
{
  "ID": "string",                   // Unique identifier for the post
  "Source": "string",               // Platform source (e.g., Reddit)
  "Type": "string",                 // Type of entry (Post, Comment, etc.)
  "Author": "string",               // Username of the author
  "Title": "string|null",           // Title of the post (nullable if not present)
  "Text": "string",                 // Raw text content of the post
  "Score": "integer",               // Upvotes or score
  "Date": "string YYYY-MM-DD",      // Date of posting
  "Word_Count": "integer",          // Word count of text
  "Comments": [                     // Array of comment objects
    {
      "comment_id": "string",       // Unique identifier for the comment
      "parent_id": "string",        // ID of parent (post ID or another comment ID)
      "Source": "string",           // Platform source (e.g., Reddit)
      "Author": "string",           // Comment author
      "Text": "string",             // Raw comment text
      "Score": "integer",           // Comment score
      "Date": "string YYYY-MM-DD",  // Comment date
      "Word_Count": "integer"       // Word count of text
    }
  ]
}
```

---

## Timeline Search Example

```bash
curl -X GET "http://localhost:9200/opinions/_search" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "range": {
        "Date": {
          "gte": "2026-01-01",
          "lte": "now"
        }
      }
    }
  }'
```
