#!/usr/bin/env python3
"""
Prepare raw unlabeled data for indexing
Fixed version with proper CSV handling
"""

import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import re

print("="*70)
print("PREPARING RAW DATA FOR INDEXING")
print("="*70)

def clean_text(text):
    """Clean text for CSV safety"""
    if not isinstance(text, str):
        return ""
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '')
    text = text.replace('\r', ' ')
    # Limit length
    return text[:5000]

# Load raw data
print("\nLoading raw_data.json...")
with open('raw_data.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data):,} posts")

# Extract all text entries
all_entries = []

print("\nProcessing entries...")
for post in tqdm(raw_data, desc="Processing"):
    # Post-level text
    if post.get('Text') and len(str(post['Text']).strip()) > 10:
        all_entries.append({
            'id': str(post.get('ID', f"post_{len(all_entries)}")),
            'text': clean_text(post['Text']),
            'source': str(post.get('Source', 'unknown')),
            'author': clean_text(post.get('Author', '')),
            'date': str(post.get('Date', '')),
            'score': int(post.get('Score', 0)) if post.get('Score') else 0,
            'title': clean_text(post.get('Title', '')),
            'type': 'post'
        })
    
    # Comments
    if post.get('Comments'):
        for comment in post['Comments']:
            if comment.get('Text') and len(str(comment['Text']).strip()) > 10:
                all_entries.append({
                    'id': str(comment.get('comment_id', f"comment_{len(all_entries)}")),
                    'text': clean_text(comment['Text']),
                    'source': str(comment.get('Source', 'unknown')),
                    'author': clean_text(comment.get('Author', '')),
                    'date': str(comment.get('Date', '')),
                    'score': int(comment.get('Score', 0)) if comment.get('Score') else 0,
                    'title': clean_text(post.get('Title', '')),
                    'type': 'comment'
                })

# Create DataFrame
df = pd.DataFrame(all_entries)

# Remove duplicates
print("\nRemoving duplicates...")
df = df.drop_duplicates(subset=['id'], keep='first')
df = df.drop_duplicates(subset=['text'], keep='first')

print(f"\n{'='*70}")
print(f"DATA SUMMARY")
print(f"{'='*70}")
print(f"Total entries: {len(df):,}")
print(f"Total words: ~{df['text'].str.split().str.len().sum():,}")
print(f"\nBy type:")
print(df['type'].value_counts())
print(f"\nBy source:")
print(df['source'].value_counts())

# Generate embeddings
print(f"\nGenerating embeddings for {len(df):,} documents...")
print("Estimated time: 15-25 minutes")

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = df['text'].tolist()

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True
)

# Save with proper escaping
print("\nSaving files...")
df.to_csv('indexed_dataset.csv', index=False, encoding='utf-8', escapechar='\\')
np.save('indexed_embeddings.npy', embeddings)

# Also save as JSON backup (safer)
df.to_json('indexed_dataset.json', orient='records', lines=True)

print(f"\n{'='*70}")
print(f"PREPARATION COMPLETE!")
print(f"{'='*70}")
print(f"Documents: {len(df):,}")
print(f"Embeddings: {embeddings.shape}") # type: ignore
print(f"\nFiles created:")
print(f"  - indexed_dataset.csv")
print(f"  - indexed_dataset.json (backup)")
print(f"  - indexed_embeddings.npy")
print(f"{'='*70}\n")
