# Enhanced Search Engine - SC4021 Information Retrieval

## Overview

This component implements the **Indexing and Retrieval** portion (40 points) of the SC4021 Information Retrieval assignment. The system provides a scalable opinion search engine that efficiently retrieves relevant discussions about AI coding tools from a large crawled dataset.

**Key Features:**
- Elasticsearch-based inverted index with 73,000+ documents
- Three search methods: Keyword (BM25), Semantic (Embeddings), Hybrid (Fusion)
- Real-time query performance (<50ms average response time)
- Three major innovations: Timeline Search, Multifaceted Exploration, Enhanced Visualizations
- Simple web-based UI for querying and analysis

---

## Assignment Coverage

### Indexing

**Indexing:**
- Inverted-index text search engine (Elasticsearch 7.17.4)
- REST-like HTTP/JSON APIs
- 73,664 documents indexed (exceeds 10,000 requirement)

**Querying:**
- Simple, user-friendly web-based UI
- Real-time search with performance metrics
- Detailed result information (source, date, author, score)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Search Engine** | Elasticsearch 7.17.4 |
| **Backend API** | Flask (REST + CORS) |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2, 384-dim) |
| **Frontend** | JavaScript + Chart.js + Plotly.js + WordCloud2.js |
| **Data Format** | JSON (hierarchical) |
| **Deployment** | Homebrew (macOS 13) / Docker (macOS 14+) |

---

## System Architecture

### 1. Data Preparation Pipeline

Two scripts are available depending on your input data:

**Script A:** `prepare_output_data.py` *(recommended — uses fully classified data)*

**Process:**
- Loads `output.json` (fully annotated data with sentiment, NER, aspect analysis)
- Cleans text and deduplicates entries
- Generates 384-dimensional embeddings using all-MiniLM-L6-v2
- Adds enriched fields: `entities`, `ai_tools`, `aspects`, `has_sarcasm`, `post_id`, `label`
- Outputs: `indexed_dataset.json` + `indexed_embeddings.npy`

**Script B:** `prepare_raw_data.py` *(alternative — uses raw crawled data)*

**Process:**
- Loads raw crawled data (73,664 documents)
- Cleans text (removes duplicates, null bytes, special characters)
- Generates 384-dimensional embeddings using all-MiniLM-L6-v2
- Outputs: `indexed_dataset.json` + `indexed_embeddings.npy`

### 2. Elasticsearch Indexing

**Script:** `index_data.py`

**Index Structure:**
```json
{
  "text": "Full document text (analyzed for BM25)",
  "embedding": "384-dim dense vector (for semantic search)",
  "source": "Platform (Reddit, HackerNews, Twitter, Blogs)",
  "author": "Username",
  "date": "Timestamp (YYYY-MM-DD)",
  "type": "post or comment",
  "title": "Optional title",
  "label": "Sentiment label (positive/negative/neutral) — from classified data",
  "entities": "Named entities extracted per document — from classified data",
  "ai_tools": "AI tool names mentioned — from classified data",
  "aspects": "Targeted aspect names — from classified data",
  "has_sarcasm": "Sarcasm flag (boolean) — from classified data",
  "post_id": "Links comments back to parent post — from classified data"
}
```

*Fields marked "from classified data" are only present when using `prepare_output_data.py`.*

**Specifications:**
- Index name: `ai_coding_search`
- Total documents: 73,664
- Index size: ~2.1 GB
- Embedding storage: ~110 MB
- Indexing method: Bulk indexing (batch size: 500)


### 3. Search API

**Script:** `search_engine.py`

**Flask REST API (Port 5001):**
- `/api/search/keyword` - BM25 keyword search
- `/api/search/semantic` - Vector similarity search
- `/api/search/hybrid` - Fusion of both methods
- `/api/timeline` - Temporal analysis with date filtering
- `/api/facets` - Source/type/author aggregations
- `/api/stats` - Index statistics

### 4. User Interface

**File:** `search_ui_enhanced.html`

**Features:**
- Simple, clean design (as per assignment requirements)
- Search input with date range filters
- Four interactive tabs:
  - **Search Results:** Side-by-side comparison of three methods
  - **Timeline Analysis:** Temporal distribution charts
  - **Multifaceted View:** Source/type breakdowns with heatmap
  - **Advanced Analytics:** Word clouds, performance charts, 3D plots
- Live statistics dashboard
- Professional metrics cards

---

## Search Methods

### Keyword Search (BM25)

**How it works:**
- Uses Elasticsearch's inverted index
- BM25 ranking algorithm (k1=1.2, b=0.75)
- Exact term matching with TF-IDF scoring

**Characteristics:**
- Fastest (~35ms average)
- High precision for exact matches
- Cannot capture semantic meaning
- Misses synonyms

**Best for:** Exact phrase matching, known terminology

---

### Semantic Search (Embeddings)

**How it works:**
- Converts query to 384-dimensional vector
- Computes cosine similarity with all document vectors
- Ranks by similarity score

**Characteristics:**
- Handles synonyms and related concepts
- Understands meaning, not just keywords
- Slightly slower (~42ms average)
- Different score scale (1-2 vs. 0-15)

**Best for:** Conceptual search, synonym handling

---

### Hybrid Search (Fusion)

**How it works:**
- Combines BM25 keyword scores + semantic similarity
- Weighted fusion: `Score = 0.5 × Semantic + 0.5 × Keyword`

**Characteristics:**
- Balanced precision and recall
- Best overall performance
- Leverages strengths of both methods

**Best for:** General-purpose search

---

## Three Innovations

### Innovation 1: Timeline Search 

**Problem Solved:**
Users need to understand how opinions evolve over time for rapidly changing AI tools.

**Implementation:**
- Date range filtering (from/to dates)
- Server-side Elasticsearch range queries
- Client-side date extraction with regex

**Visualizations:**
- Monthly distribution line chart
- Year-over-year bar chart comparison
- Peak activity periods ranking


---

### Innovation 2: Multifaceted Search 

**Problem Solved:**
Different platforms have different discussion styles (Reddit: casual, HackerNews: technical).

**Implementation:**
- Elasticsearch aggregations by source, type, author
- Interactive facet lists with document counts
- Cross-tabulation analysis

**Visualizations:**
- Source distribution bar chart
- Content type breakdown
- Source × Type heatmap


---

### Innovation 3: Enhanced Visualizations 

**Problem Solved:**
Text-only results lack pattern insights and analytical depth.

**Implementation:**
- Word cloud (top 60 terms by frequency)
- Source distribution pie chart
- Method performance comparison (avg + max scores)
- Score distribution histogram (grouped by method)
- 3D performance landscape


---

## Performance Metrics

### Query Performance

| Metric | Value |
|--------|-------|
| **Average Response Time** | 37.8 ms |
| **Keyword Search** | 34.1 ms |
| **Semantic Search** | 41.4 ms |
| **Hybrid Search** | 37.8 ms |
| **All Queries** | <50 ms (real-time) |

### Indexing Performance

| Metric | Value |
|--------|-------|
| **Documents Indexed** | 73,664 |
| **Indexing Speed** | ~245 docs/second |
| **Total Indexing Time** | ~5 minutes |
| **Index Size** | 2.1 GB |
| **Embedding Generation** | ~20 minutes |


---

## Installation & Setup

### Prerequisites

- **Python:** 3.10, 3.11, 3.12, or 3.13
- **pip:** Latest version (upgrade with `pip install --upgrade pip`)
- **Elasticsearch:** 7.17.4 (via Homebrew OR Docker)
- **Modern web browser:** Chrome, Firefox, Safari

**Check your Python version:**
```bash
python3 --version
```

**Note:** Python 3.9 or older is not supported. Please upgrade to Python 3.10+

---

### Python Setup (Required)

**Create a virtual environment** to avoid dependency conflicts:
```bash
# Navigate to project directory
cd enhanced-search

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Upgrade pip (important!)
pip install --upgrade pip

# Verify activation (should see (.venv) in prompt)
which python  # Should point to .venv/bin/python
```

---

### Option 1: Homebrew Installation (macOS 13 and below)

**1. Install Java 11:**
```bash
brew install openjdk@11
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-11.jdk/Contents/Home
```

**2. Install Elasticsearch:**
```bash
brew install elasticsearch-full
```

**3. Install Python Dependencies:**
```bash
# Make sure virtual environment is activated!
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Option 2: Docker Installation (macOS 14+, Windows, Linux) - Recommended

**1. Install Docker Desktop:**
- Download from: https://www.docker.com/products/docker-desktop

**2. Pull and Run Elasticsearch:**
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

**3. Install Python Dependencies:**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies (may take 5-10 minutes)
pip install -r requirements.txt
```

### Step 2: Prepare and Index Data (First Time Only)

**Option A — Prepare from classified output (recommended):**
```bash
python prepare_output_data.py
# Reads output.json (fully classified data)
# Generates enriched index with sentiment, NER, aspects
# Generates embeddings (~15-30 minutes)
# Outputs: indexed_dataset.json, indexed_embeddings.npy
```

**Option B — Prepare from raw crawled data:**
```bash
python prepare_raw_data.py
# Processes 73,664 raw documents
# Generates embeddings (~20 minutes)
# Outputs: indexed_dataset.json, indexed_embeddings.npy
```

**Index into Elasticsearch:**
```bash
python index_data.py
# Bulk indexes all documents (~5 minutes)
# Creates index: ai_coding_search
```

**Verify indexing:**
```bash
curl http://localhost:9200/ai_coding_search/_count
# Should return: {"count":73664}
```

---

### Step 3: Start Search Engine API
```bash
python search_engine.py
# Flask API starts on port 5001
# Available at: http://localhost:5001
```

---

### Step 4: Open User Interface
```bash
open search_ui_enhanced.html
# Or manually open the file in your browser
```

---

## Project Structure
```
enhanced-search/
├── search_engine.py              # Flask REST API
├── search_ui_enhanced.html       # Web-based UI
├── index_data.py                 # Elasticsearch indexing script
├── prepare_output_data.py        # Data prep from classified output.json (recommended)
├── prepare_raw_data.py           # Data prep from raw crawled data
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── output.json                   # Fully classified data (sentiment, NER, aspects)
├── raw_data.json                 # Original crawled data (73,664 docs)
├── indexed_dataset.json          # Processed data (generated — not committed)
└── indexed_embeddings.npy        # 384-dim embeddings (generated — not committed)
```

---

## API Documentation

### Keyword Search
```bash
GET /api/search/keyword?q=claude&size=10&date_from=2025-01-01&date_to=2026-03-29
```

### Semantic Search
```bash
GET /api/search/semantic?q=claude&size=10
```

### Hybrid Search
```bash
GET /api/search/hybrid?q=claude&size=10
```

### Timeline Analysis
```bash
GET /api/timeline?q=claude
```

### Faceted Breakdown
```bash
GET /api/facets?q=claude
```

### Index Statistics
```bash
GET /api/stats
```

---

## License

Academic Use Only - SC4021 Information Retrieval Assignment 2026
