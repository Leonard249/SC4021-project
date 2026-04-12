# Enhanced Search Engine - SC4021 Information Retrieval

## Overview

This component implements the **Indexing and Retrieval** portion (40 points) of the SC4021 Information Retrieval assignment. The system provides a scalable opinion search engine that retrieves and analyses discussions about AI coding tools from a large crawled and classified dataset.

**Key Features:**
- Elasticsearch-based inverted index with 69,047 documents
- Three search methods: Keyword (BM25), Semantic (Embeddings), Hybrid (Fusion)
- Sentiment-aware search with classified labels (positive / negative / neutral)
- Named entity recognition (NER) fields: AI tools, aspects, entities
- Aspect-based sentiment analysis and sarcasm detection
- Four innovations: Timeline Search, Multifaceted Exploration, Sentiment Analysis, Enhanced Visualisations
- 14 REST API endpoints served via Flask on port 5001
- Web-based UI with five interactive tabs

---

## Assignment Coverage

### Indexing
- Inverted-index text search engine (Elasticsearch 7.17.4)
- REST HTTP/JSON API with 14 endpoints
- 69,047 documents indexed (exceeds 10,000 requirement)
- Full classified/annotated data — sentiment labels, subjectivity scores, NER tags, aspect sentiments

### Querying
- Simple, user-friendly web-based UI (`search_ui_enhanced.html`)
- Real-time search with per-query timing metrics
- Detailed result cards (source, date, author, score, sentiment label)
- Date-range and source filtering

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Search Engine** | Elasticsearch 7.17.4 |
| **Backend API** | Flask (REST + CORS) |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`, 384-dim) |
| **Frontend** | JavaScript + Chart.js + Plotly.js + WordCloud2.js |
| **Data Format** | JSON (hierarchical + NDJSON for indexing) |
| **Deployment** | Homebrew (macOS 13) / Docker (macOS 14+) |

---

## System Architecture

### 1. Data Preparation Pipeline

**Script:** `prepare_output_data.py`

Reads the fully classified dataset (`output.json`) produced by the NLP pipeline and prepares it for indexing.

**Process:**
1. Load `output.json` — 47,444 top-level posts + comments
2. Clean text (remove null bytes, truncate to 5,000 chars)
3. Extract classification fields:
   - `label` — Overall_Document_Polarity (positive / negative / neutral)
   - `subjectivity` / `subjectivity_score`
   - `aspect_sentiments` — flat string of `"Aspect: polarity"` pairs
   - `entities` — all named entities (original casing)
   - `ai_tools` — AI_TOOL NER entities only (lowercase, for keyword matching)
   - `aspects` — targeted aspect names
   - `has_sarcasm` — True only when ≥2 aspect sentences score ≥0.85 confidence
4. Deduplicate by ID and first 200 chars of text
5. Generate 384-dim embeddings with `all-MiniLM-L6-v2`
6. Output: `indexed_dataset.json` (NDJSON) + `indexed_embeddings.npy` + `embedding_info.json`

**Output stats:**
- Documents: 69,047
- Label distribution: Neutral 25,908 · Negative 22,255 · Positive 20,884
- Top AI tools mentioned: claude, cursor, gemini, copilot, chatgpt, windsurf, replit, tabnine, codeium, llama

---

### 2. Elasticsearch Indexing

**Script:** `index_data.py`

**Index name:** `ai_coding_search`

**Full field mapping:**

```json
{
  "text":               "Full document text (BM25 analyzed)",
  "embedding":          "384-dim dense_vector (cosine similarity)",
  "source":             "Platform keyword (Reddit, HackerNews, etc.)",
  "author":             "Username",
  "date":               "Timestamp string",
  "score":              "Upvote/engagement score (integer)",
  "title":              "Post title",
  "type":               "post or comment",
  "post_id":            "Parent post ID (links comments to posts)",
  "label":              "Sentiment label: positive / negative / neutral",
  "subjectivity":       "subjective or objective",
  "subjectivity_score": "Float 0.0–1.0",
  "aspect_sentiments":  "Flat string: 'Aspect: polarity; ...'",
  "entities":           "List of named entity strings",
  "ai_tools":           "List of AI_TOOL entity strings (lowercase)",
  "aspects":            "List of targeted aspect names",
  "has_sarcasm":        "Boolean sarcasm flag"
}
```

**Specifications:**
- Index size: ~2.1 GB
- Bulk indexing batch size: 200
- Embedding dimensions: 384

---

### 3. Search API

**Script:** `search_engine.py` — Flask REST API on port 5001

| Category | Endpoint | Description |
|----------|----------|-------------|
| **Search** | `GET /api/search/keyword` | BM25 keyword search |
| | `GET /api/search/semantic` | Vector cosine similarity |
| | `GET /api/search/hybrid` | Fusion of BM25 + semantic |
| | `GET /api/search/by_source` | Filter by platform |
| | `GET /api/search/by_period` | Filter by time period |
| **Analytics** | `GET /api/timeline` | Temporal distribution |
| | `GET /api/facets` | Source/type/author aggregations |
| | `GET /api/sentiment_trend` | Sentiment over time |
| | `GET /api/tool_comparison` | Sentiment by AI tool |
| | `GET /api/entity_search` | Search by entity name |
| | `GET /api/top_entities` | Top mentioned entities |
| **Documents** | `GET /api/doc/<id>` | Fetch single document |
| | `GET /api/doc/<id>/siblings` | Fetch sibling comments |
| **Stats** | `GET /api/stats` | Index statistics |

---

### 4. User Interface

**File:** `search_ui_enhanced.html`

Five interactive tabs:

| Tab | Description |
|-----|-------------|
| **Search Results** | Side-by-side comparison of all three search methods with sentiment badges |
| **Timeline Analysis** | Monthly + yearly distribution charts with peak period ranking |
| **Multifaceted View** | Source/type facets, author breakdown, source×type heatmap |
| **Sentiment Analysis** | Sentiment trends over time, per-tool sentiment comparison, top entities |
| **Advanced Visualisations** | Word cloud (top 60 terms), score distribution histogram, 3D performance landscape |

---

## Search Methods

### Keyword Search (BM25)

- Uses Elasticsearch's inverted index
- BM25 ranking: k₁=1.2, b=0.75
- Supports date-range and source filters
- Avg. response time: ~71.7 ms

**Best for:** Exact phrase matching, known terminology

---

### Semantic Search (Embeddings)

- Query encoded to 384-dim vector via `all-MiniLM-L6-v2`
- Cosine similarity via Elasticsearch `script_score`
- Handles synonyms and related concepts
- Avg. response time: ~119.9 ms

**Best for:** Conceptual search, paraphrased queries

---

### Hybrid Search (Fusion)

- Combines BM25 and semantic scores
- Fusion formula: `Score = 0.5 × Semantic + 0.5 × BM25`
- Best overall relevance
- Avg. response time: ~113.6 ms

**Best for:** General-purpose search

---

## Four Innovations

### Innovation 1: Timeline Search

**Problem:** Users need to understand how opinions evolve over time for rapidly changing AI tools.

**Implementation:**
- Date-range filtering via Elasticsearch range queries
- Monthly and yearly aggregations returned per query
- Client-side regex date extraction for fallback

**UI features:** Monthly line chart, year-over-year bar chart, peak activity periods table

---

### Innovation 2: Multifaceted Search

**Problem:** Different platforms have different discussion styles (Reddit: casual, HackerNews: technical, Blogs: in-depth).

**Implementation:**
- Elasticsearch term aggregations by source, type, author
- Interactive facet panel with document counts
- Cross-tabulation (source × type) heatmap

**UI features:** Source bar chart, content-type breakdown, source×type heatmap

---

### Innovation 3: Sentiment-Aware Search and Analytics

**Problem:** Keyword and semantic search return results without any indication of opinion polarity or analytical depth.

**Implementation:**
- Every document carries a classified `label` field (positive / negative / neutral)
- `sentiment_trend` endpoint returns monthly sentiment counts for any query
- `tool_comparison` endpoint computes per-tool sentiment breakdown across all documents
- `top_entities` endpoint surfaces the most-mentioned AI tools and technical concepts
- Sarcasm detection flag (`has_sarcasm`) with threshold ≥0.85 and ≥2 sentence hits required to reduce false positives

**UI features:** Sentiment trend stacked area chart, tool comparison grouped bar chart, top entities ranked chart

---

### Innovation 4: Enhanced Visualisations

**Problem:** Text-only results lack pattern insights and analytical depth.

**Implementation:**
- Word cloud normalises frequencies to [10, 50] range to prevent overflow (WordCloud2.js)
- Score distribution histogram grouped by search method
- 3D performance landscape (query × method × score)

**UI features:** Word cloud (top 60 terms), histogram, 3D Plotly surface

---

## Performance Metrics

### Query Performance (live measurements, 5 queries)

| Method | Avg. Response Time | Notes |
|--------|--------------------|-------|
| Keyword (BM25) | 71.7 ms | Can be slower on high-frequency terms |
| Semantic | 119.9 ms | Includes query encoding + vector scan |
| Hybrid | 113.6 ms | Fusion of both |

### Indexing Performance

| Metric | Value |
|--------|-------|
| **Documents Indexed** | 69,047 |
| **Bulk Batch Size** | 200 documents |
| **Index Size** | ~2.1 GB |
| **Embedding Dimensions** | 384 |
| **Embedding Model** | all-MiniLM-L6-v2 |

---

## Installation & Setup

### Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13
- Elasticsearch 7.17.4
- Modern web browser (Chrome, Firefox, Safari)

---

### Option 1: Homebrew (macOS 13 and below)

```bash
# Install Java 11
brew install openjdk@11
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-11.jdk/Contents/Home

# Install Elasticsearch
brew install elasticsearch-full
brew services start elasticsearch-full
```

---

### Option 2: Docker (macOS 14+, Windows, Linux) — Recommended

```bash
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.17.4

docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  docker.elastic.co/elasticsearch/elasticsearch:7.17.4
```

---

### Python Setup

```bash
cd enhanced-search

python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 1: Prepare and Index Data (First Time Only)

Place `output.json` (the classified dataset) in the `enhanced-search/` directory, then:

```bash
# Generate indexed_dataset.json + indexed_embeddings.npy
python prepare_output_data.py
# ~15-30 minutes depending on hardware (69,047 documents)

# Bulk-index into Elasticsearch
python index_data.py
# ~5 minutes

# Verify
curl http://localhost:9200/ai_coding_search/_count
# Expected: {"count":69047}
```

---

### Step 2: Start the Search API

```bash
python search_engine.py
# Flask starts on http://localhost:5001
```

---

### Step 3: Open the UI

```bash
open search_ui_enhanced.html
# Or open the file manually in your browser
```

---

## Project Structure

```
enhanced-search/
├── search_engine.py              # Flask REST API (14 endpoints, port 5001)
├── search_ui_enhanced.html       # Web UI (5 tabs)
├── index_data.py                 # Elasticsearch bulk indexing
├── prepare_output_data.py        # Data prep from output.json (classified dataset)
├── prepare_raw_data.py           # Legacy: data prep from raw_data.json (unclassified)
├── generate_submission_data.py   # Query experiment + submission data export
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── embedding_info.json           # Embedding metadata + label distribution
│
├── images/
│   └── precision_recall_curve.png  # P/R curve (query: Claude productivity)
│
├── output.json                   # Classified dataset (not committed — too large)
├── indexed_dataset.json          # Processed NDJSON (gitignored)
└── indexed_embeddings.npy        # 384-dim embeddings (gitignored)
```

---

## API Quick Reference

```bash
# Keyword search with date filter
GET /api/search/keyword?q=claude&size=10&date_from=2024-01-01&date_to=2026-01-01

# Semantic search
GET /api/search/semantic?q=github+copilot+productivity&size=10

# Hybrid search
GET /api/search/hybrid?q=cursor+vs+copilot&size=20

# Filter by platform
GET /api/search/by_source?q=chatgpt&source=Reddit&size=10

# Filter by time period (year_month format: YYYY-MM)
GET /api/search/by_period?q=AI+coding&year_month=2025-03

# Timeline aggregation
GET /api/timeline?q=claude

# Faceted breakdown
GET /api/facets?q=copilot

# Sentiment trend over time
GET /api/sentiment_trend?q=cursor

# Tool sentiment comparison
GET /api/tool_comparison

# Entity search
GET /api/entity_search?entity=cursor

# Top entities
GET /api/top_entities?size=20

# Single document
GET /api/doc/<doc_id>

# Sibling comments for a post
GET /api/doc/<doc_id>/siblings

# Index statistics
GET /api/stats
```

---

## License

Academic Use Only — SC4021 Information Retrieval Assignment 2026
