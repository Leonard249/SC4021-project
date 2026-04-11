#!/usr/bin/env python3
"""
Index prepared data into Elasticsearch.
Supports the full classified dataset with sentiment labels,
subjectivity scores, and aspect-based sentiment analysis.
"""

from elasticsearch import Elasticsearch, helpers
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("INDEXING CLASSIFIED DATA INTO ELASTICSEARCH")
print("=" * 70)

print("\nConnecting to Elasticsearch...")
es = Elasticsearch(['http://localhost:9200'])
if not es.ping():
    print("ERROR: Elasticsearch not running! Start it first.")
    exit(1)
print("Connected!")

# Load data
print("\nLoading prepared data...")
all_entries = []
with open('indexed_dataset.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            all_entries.append(json.loads(line))

embeddings = np.load('indexed_embeddings.npy')
print(f"Loaded {len(all_entries):,} documents with {embeddings.shape[1]}-dim embeddings")

if len(all_entries) != len(embeddings):
    print(f"WARNING: document count ({len(all_entries)}) != embedding count ({len(embeddings)})")

# Create index
index_name = 'ai_coding_search'
print(f"\nCreating index: {index_name}")

if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)
    print("Deleted old index")

mapping = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "id":                 {"type": "keyword"},
            "text":               {"type": "text"},
            "embedding":          {"type": "dense_vector", "dims": 384},
            "source":             {"type": "keyword"},
            "author":             {"type": "text",    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
            "date":               {"type": "keyword"},
            "score":              {"type": "integer"},
            "title":              {"type": "text"},
            "type":               {"type": "keyword"},
            "post_id":            {"type": "keyword"},
            "label":              {"type": "keyword"},   # positive / negative / neutral
            "subjectivity":       {"type": "keyword"},   # subjective / objective
            "subjectivity_score": {"type": "float"},
            "aspect_sentiments":  {"type": "text"},
            "entities":           {"type": "keyword"},   # list of all named entities
            "ai_tools":           {"type": "keyword"},   # list of AI_TOOL entities only
            "aspects":            {"type": "keyword"},   # list of targeted aspect names
            "has_sarcasm":        {"type": "boolean"}    # sarcasm detected flag
        }
    }
}

es.indices.create(index=index_name, body=mapping)  # type: ignore
print("Index created with sentiment fields")

# Bulk index
print(f"\nIndexing {len(all_entries):,} documents...")

actions = []
indexed = 0
errors = 0

for idx, entry in enumerate(all_entries):
    doc = {
        "_index": index_name,
        "_id": str(entry['id']),
        "_source": {
            "id":                 str(entry.get('id', '')),
            "text":               str(entry.get('text', '')),
            "embedding":          embeddings[idx].tolist(),
            "source":             str(entry.get('source', 'unknown')),
            "author":             str(entry.get('author', '')),
            "date":               str(entry.get('date', '')),
            "score":              int(entry.get('score', 0)),
            "title":              str(entry.get('title', '')),
            "type":               str(entry.get('type', 'post')),
            "post_id":            str(entry.get('post_id', '')),
            "label":              str(entry.get('label', 'neutral')),
            "subjectivity":       str(entry.get('subjectivity', 'objective')),
            "subjectivity_score": float(entry.get('subjectivity_score', 0.0)),
            "aspect_sentiments":  str(entry.get('aspect_sentiments', '')),
            "entities":           entry.get('entities', []),
            "ai_tools":           entry.get('ai_tools', []),
            "aspects":            entry.get('aspects', []),
            "has_sarcasm":        bool(entry.get('has_sarcasm', False))
        }
    }
    actions.append(doc)

    if len(actions) >= 200:
        result = helpers.bulk(es, actions, raise_on_error=False, stats_only=True)
        indexed += result[0]  # type: ignore
        errors += result[1]   # type: ignore
        print(f"  Indexed {indexed:,}/{len(all_entries):,}...", end='\r')
        actions = []

if actions:
    result = helpers.bulk(es, actions, raise_on_error=False, stats_only=True)
    indexed += result[0]  # type: ignore
    errors += result[1]   # type: ignore

print(f"\nIndexing complete! {indexed:,} documents indexed, {errors} errors")

# Verify
es.indices.refresh(index=index_name)
count = es.count(index=index_name)['count']

print(f"\n{'='*70}")
print(f"INDEX STATISTICS")
print(f"{'='*70}")
print(f"Documents in index: {count:,}")
print(f"\nSentiment label distribution:")
label_agg = es.search(index=index_name, body={  # type: ignore
    "size": 0,
    "aggs": {"by_label": {"terms": {"field": "label", "size": 10}}}
})
for bucket in label_agg['aggregations']['by_label']['buckets']:
    print(f"  {bucket['key']:12s}: {bucket['doc_count']:,}")

print(f"\nIndex '{index_name}' ready!")
print(f"{'='*70}\n")
