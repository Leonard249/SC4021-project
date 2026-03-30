# HackerNews Opinion Crawler

Complete crawler system for collecting and processing HackerNews opinions on AI coding productivity (vibe coding).


This crawler supports an opinion search engine project investigating: **"Does AI boost coding productivity?"**

**Data Source**: HackerNews (via Algolia Search API)  
**Target**: 10,000+ opinionated comments and stories  
**Output**: Structured corpus with sentiment/subjectivity labels  

## System Architecture

```
┌─────────────────┐
│  hn_scraper.py  │  Step 1: Crawl HN via Algolia API
└────────┬────────┘
         │ raw_corpus.jsonl (20k-50k records)
         ▼
┌─────────────────┐
│ hn_preprocess.py│  Step 2: Clean, dedupe, normalize
└────────┬────────┘
         │ cleaned_corpus.jsonl (15k-40k records)
         ▼
┌─────────────────┐
│hn_relevance_    │  Step 3: Filter by relevance & opinion
│  filter.py      │
└────────┬────────┘
         │ filtered_corpus.jsonl (10k-20k records)
         ├───────────────────────────────┐
         ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│  hn_indexer.py  │            │hn_eval_generator│
│                 │            │     .py         │
└────────┬────────┘            └────────┬────────┘
         │                              │
         ▼                              ▼
  Elasticsearch              eval_dataset.csv
   (searchable)              (for labeling)
```

## Installation

### Prerequisites
- Python 3.8+
- Elasticsearch 7.x or 8.x (optional, for indexing)

### Install Dependencies

```bash
# Core dependencies (required)
pip install requests

# Elasticsearch (optional, for indexing)
pip install elasticsearch


## Quick Start

### Option 1: Run Complete Pipeline

```bash
# Run everything with default settings
python3 hn_pipeline.py --output-dir ./dataset

# Custom configuration
python3 hn_pipeline.py \
    --output-dir ./dataset \
    --comments-per-keyword 1500 \
    --stories-per-keyword 150 \
    --min-relevance 0.2 \
    --min-opinion 0.1 \
    --eval-size 1200
```


## Output Files

### Dataset Files
```
dataset/
├── raw_corpus.jsonl           # Raw crawled data
├── cleaned_corpus.jsonl        # After deduplication & cleaning
├── filtered_corpus.jsonl       # Final dataset (for indexing)
├── scored_corpus.jsonl         # All records with scores
├── corpus_stats.json           # Corpus statistics
├── validation_sample.txt       # Sample for quality check
├── eval_dataset.csv            # For manual annotation
├── annotation_guidelines.txt   # Labeling instructions
└── annotation_quick_reference.txt
```

### Record Schema

Each record contains:

```json
{
  "source": "hackernews",
  "type": "comment",
  "item_id": "38401234",
  "parent_id": "38401100",
  "story_id": "38400000",
  "author": "username",
  "created_at": "2025-01-15T10:30:00Z",
  "created_at_i": 1736936400,
  "url": "https://news.ycombinator.com/item?id=38401234",
  "story_title": "GitHub Copilot Review",
  "text_raw": "Original text with HTML...",
  "text_clean": "Cleaned plain text",
  "points": 42,
  "source_hash": "sha1 of source+id",
  "text_hash": "sha1 of cleaned text",
  "topic_tags": ["copilot", "productivity"],
  "matched_categories": ["ai_tools", "productivity"],
  "relevance_score": 0.85,
  "opinion_score": 0.67,
  "is_relevant": true
}
```

## Crawling Strategy

### Keywords (32 total)

**AI Tools:**
- copilot, cursor, github copilot, codex, codeium, tabnine
- AI pair programming, LLM coding, AI code completion

**Productivity:**
- developer productivity, coding productivity, faster coding
- accelerate development, boost productivity, coding flow

**Debate Language:**
- AI hallucination, buggy code, slower coding, learning curve
- technical debt, AI code review, automated coding

**Concepts:**
- vibe coding, SWE agent, agentic coding

### Target Volume

| Phase | Target | Expected |
|-------|--------|----------|
| Raw crawl | 20k-50k | ~35k |
| After cleaning | 15k-40k | ~25k |
| After filtering | 10k-20k | ~15k |
| Evaluation set | 1,000 | 1,000 |

## Configuration

### Relevance Filtering

**Minimum Relevance Score** (default: 0.15)
- Based on keyword matching across 5 categories
- Higher = more directly relevant to AI coding productivity

**Minimum Opinion Score** (default: 0.05)
- Based on opinion indicators (I/my, evaluative language, etc.)
- Higher = more subjective/opinionated content

**Adjust thresholds:**
```bash
# More strict (higher quality, smaller corpus)
--min-relevance 0.25 --min-opinion 0.15

# More lenient (larger corpus, more noise)
--min-relevance 0.10 --min-opinion 0.03
```

## Manual Annotation

### Requirements
-  Minimum 1,000 labeled records
-  ≥80% inter-annotator agreement
- Save as `eval.xls` in benchmark format


## Elasticsearch Integration

### Start Elasticsearch (Docker)

```bash
# If using docker-compose in your backend/ folder
cd backend
docker-compose up -d elasticsearch

# Or standalone Docker
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0
```

### Index the Corpus

```bash
python3 hn_indexer.py \
    --input ./dataset/filtered_corpus.jsonl \
    --es-host localhost \
    --es-port 9200 \
    --index-name hackernews_opinions
```

### Sample Queries

The indexer runs these automatically after indexing:

1. "copilot productivity" - Search for Copilot productivity opinions
2. "cursor vs copilot" - Compare Cursor and Copilot
3. "AI coding bugs" - Find opinions about AI coding bugs

### Custom Queries

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(["http://localhost:9200"])

# Multi-field search
result = es.search(
    index="hackernews_opinions",
    body={
        "query": {
            "multi_match": {
                "query": "vibe coding flow",
                "fields": ["text_clean^2", "story_title"]
            }
        },
        "size": 10
    }
)

# Filter by opinion score
result = es.search(
    index="hackernews_opinions",
    body={
        "query": {
            "bool": {
                "must": {
                    "match": {"text_clean": "copilot"}
                },
                "filter": {
                    "range": {"opinion_score": {"gte": 0.5}}
                }
            }
        }
    }
)
```









