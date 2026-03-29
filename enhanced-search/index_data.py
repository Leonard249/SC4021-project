#!/usr/bin/env python3
"""
Index prepared data into Elasticsearch
Uses JSON instead of CSV to avoid parsing issues
"""

from elasticsearch import Elasticsearch, helpers
import pandas as pd            
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("INDEXING DATA INTO ELASTICSEARCH")
print("="*70)

# Connect
print("\n🔌 Connecting to Elasticsearch...")
es = Elasticsearch(['http://localhost:9200'])
if not es.ping():
    print("ERROR: Elasticsearch not running!")
    exit(1)
print("Connected!")

# Load data (use JSON instead of CSV)
print("\nLoading prepared data...")
df = pd.read_json('indexed_dataset.json', lines=True)
embeddings = np.load('indexed_embeddings.npy')
print(f"Loaded {len(df):,} documents with embeddings")

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
            "id": {"type": "keyword"},
            "text": {"type": "text"},
            "embedding": {"type": "dense_vector", "dims": 384},
            "source": {"type": "keyword"},
            "author": {"type": "text"},
            "date": {"type": "keyword"},
            "score": {"type": "integer"},
            "title": {"type": "text"},
            "type": {"type": "keyword"}
        }
    }
}

es.indices.create(index=index_name, body=mapping) # type: ignore
print("Index created")

# Bulk index
print(f"\nIndexing {len(df):,} documents...")

actions = []
indexed = 0

for idx, row in df.iterrows():
    doc = {
        "_index": index_name,
        "_id": str(row['id']),
        "_source": {
            "id": str(row['id']),
            "text": str(row['text']),
            "embedding": embeddings[idx].tolist(),
            "source": str(row['source']),
            "author": str(row['author']),
            "date": str(row['date']),
            "score": int(row['score']),
            "title": str(row['title']),
            "type": str(row['type'])
        }
    }
    actions.append(doc)
    
    if len(actions) >= 100:
        helpers.bulk(es, actions, raise_on_error=False)
        indexed += len(actions)
        print(f"  Indexed {indexed:,}/{len(df):,}...")
        actions = []

if actions:
    helpers.bulk(es, actions, raise_on_error=False)
    indexed += len(actions)

print(f"\nIndexing complete! {indexed:,} documents")

# Verify
es.indices.refresh(index=index_name)
count = es.count(index=index_name)['count']

print(f"\n{'='*70}")
print(f"INDEX STATISTICS")
print(f"{'='*70}")
print(f"Documents: {count:,}")
print(f"\nIndex '{index_name}' ready for search!")
print(f"{'='*70}\n")
