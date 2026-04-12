## Sample JSON schema

````{
  "ID": "string",                // Unique identifier for the post
  "Source": "string",            // Platform source (e.g., Reddit)
  "Type": "string",              // Type of entry (Post, Comment, etc.)
  "Author": "string",            // Username of the author
  "Title": "string|null",        // Title of the post (nullable if not present)
  "Text": "string",              // Raw text content of the post
  "Score": "integer",            // Upvotes or score
  "Date": "string YYYY-MM-DD",   // Date of posting
  "Word_Count": "integer",       // Word count of text
  "Comments": [                  // Array of comment objects
    {
      "comment_id": "string",    // Unique identifier for the comment
      "parent_id": "string",     // ID of parent (post ID or another comment ID)
      "Source": "string",        // Platform source (e.g., Reddit)
      "Author": "string",        // Comment author
      "Text": "string",          // Raw comment text
      "Score": "integer",        // Comment score
      "Date": "string YYYY-MM-DD", // Comment date
      "Word_Count": "integer"    // Word count of text
    }
  ]
}```
``````

## Timeline search

```curl -X GET "http://localhost:9200/opinions/_search" -H 'Content-Type: application/json' -d '{
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
