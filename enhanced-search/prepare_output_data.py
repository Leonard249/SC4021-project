#!/usr/bin/env python3
"""
Prepare classification output data (classified_eval.json) for indexing.
Uses the fully classified/annotated data including sentiment labels,
subjectivity scores, aspect-based sentiment analysis, and named entities.
"""

import json
import numpy as np
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("PREPARING CLASSIFIED OUTPUT DATA FOR INDEXING")
print("=" * 70)

# Entity types we care about for the AI coding tools topic
AI_ENTITY_TYPES = {'AI_TOOL', 'TECH_CONCEPT', 'ORG', 'PL', 'EDITOR', 'FRAMEWORK'}


def clean_text(text):
    """Clean text for indexing."""
    if not isinstance(text, str):
        return ""
    return text.replace('\x00', '').replace('\r', ' ')[:5000]


def extract_entities(ner_tags):
    """
    Extract unique entity names from NER_Tags.
    NER_Tags format: [[name, type, start, end], ...]
    Returns a list of unique lowercase entity names for AI-relevant types.
    """
    if not ner_tags or not isinstance(ner_tags, list):
        return []
    seen = set()
    entities = []
    for tag in ner_tags:
        if isinstance(tag, list) and len(tag) >= 2:
            name = str(tag[0]).strip()
            etype = str(tag[1]).strip()
            key = name.lower()
            if name and len(name) > 1 and key not in seen:
                seen.add(key)
                entities.append(name)   # keep original casing for display
    return entities


def extract_ai_tools(ner_tags):
    """Extract only AI_TOOL entities for the tool comparison feature."""
    if not ner_tags or not isinstance(ner_tags, list):
        return []
    seen = set()
    tools = []
    for tag in ner_tags:
        if isinstance(tag, list) and len(tag) >= 2:
            name = str(tag[0]).strip()
            etype = str(tag[1]).strip()
            key = name.lower()
            if etype == 'AI_TOOL' and name and key not in seen:
                seen.add(key)
                tools.append(key)   # store lowercase for case-insensitive keyword matching
    return tools


def extract_aspects(targeted_aspects):
    """
    Extract unique aspect names from Targeted_Aspects.
    Returns list of aspect name strings.
    """
    if not targeted_aspects or not isinstance(targeted_aspects, list):
        return []
    seen = set()
    aspects = []
    for aspect in targeted_aspects:
        if isinstance(aspect, dict):
            name = str(aspect.get('Aspect_Name', '')).strip()
            key = name.lower()
            if name and len(name) > 1 and key not in seen:
                seen.add(key)
                aspects.append(name)
    return aspects


def extract_aspect_sentiments_structured(aspect_sentiments):
    """
    Extract aspect sentiment as a searchable string AND structured list.
    aspect_sentiments format: [{"Aspect": "...", "Final_Polarity": "...", "Final_Score": ...}, ...]
    Returns: (flat_string, list_of_dicts)
    """
    if not aspect_sentiments or not isinstance(aspect_sentiments, list):
        return "", []
    parts = []
    structured = []
    for asp in aspect_sentiments:
        if isinstance(asp, dict):
            name = asp.get('Aspect', '')
            polarity = asp.get('Final_Polarity', '')
            score = asp.get('Final_Score', 0.0)
            if name:
                parts.append(f"{name}: {polarity}")
                structured.append({'aspect': name, 'polarity': polarity, 'score': score})
    return "; ".join(parts), structured


def check_sarcasm(targeted_aspects):
    """
    Return True only if multiple aspect sentences (or one with very high confidence)
    are marked sarcastic. Threshold raised to 0.85 to reduce false positives —
    the sarcasm model over-triggers on enthusiastic language like 'amazing!'.
    Requiring at least 2 hits avoids single-sentence false positives.
    """
    if not targeted_aspects or not isinstance(targeted_aspects, list):
        return False
    hits = []
    for aspect in targeted_aspects:
        if isinstance(aspect, dict):
            sarcasm = aspect.get('Sarcasm', {})
            if isinstance(sarcasm, dict) and sarcasm.get('Is_Sarcastic'):
                conf = sarcasm.get('Sarcasm_Confidence', 0)
                if conf >= 0.85:
                    hits.append(conf)
    # Require at least 2 high-confidence hits, OR 1 hit with confidence >= 0.95
    return len(hits) >= 2 or any(c >= 0.95 for c in hits)


print("\nLoading classified_eval.json...")
with open('classified_eval.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data):,} top-level documents")

all_entries = []

print("\nProcessing entries...")
for post in tqdm(raw_data, desc="Processing posts"):
    post_id = str(post.get('ID', f"post_{len(all_entries)}"))
    post_text = clean_text(post.get('Text', ''))

    if post_text and len(post_text.strip()) > 10:
        asp_str, _ = extract_aspect_sentiments_structured(post.get('Aspect_Sentiments'))
        entities = extract_entities(post.get('NER_Tags'))
        ai_tools = extract_ai_tools(post.get('NER_Tags'))
        aspects = extract_aspects(post.get('Targeted_Aspects'))

        all_entries.append({
            'id': post_id,
            'text': post_text,
            'source': str(post.get('Source', 'unknown')),
            'author': clean_text(post.get('Author', '') or ''),
            'date': str(post.get('Date', '') or ''),
            'score': int(post.get('Score', 0) or 0),
            'title': clean_text(post.get('Title', '') or ''),
            'type': str(post.get('Type', 'Post')).lower(),
            'post_id': post_id,
            'label': str(post.get('Overall_Document_Polarity', 'neutral') or 'neutral'),
            'subjectivity': str(post.get('Subjectivity', 'objective') or 'objective'),
            'subjectivity_score': float(post.get('Subjectivity_Score', 0.0) or 0.0),
            'aspect_sentiments': asp_str,
            'entities': entities,          # list of all named entities
            'ai_tools': ai_tools,          # list of AI_TOOL entities only
            'aspects': aspects,            # list of targeted aspect names
            'has_sarcasm': check_sarcasm(post.get('Targeted_Aspects')),
        })

    # Process comments
    for comment in (post.get('Comments') or []):
        comment_text = clean_text(comment.get('Text', ''))
        if comment_text and len(comment_text.strip()) > 10:
            asp_str, _ = extract_aspect_sentiments_structured(comment.get('Aspect_Sentiments'))
            entities = extract_entities(comment.get('NER_Tags'))
            ai_tools = extract_ai_tools(comment.get('NER_Tags'))
            aspects = extract_aspects(comment.get('Targeted_Aspects'))

            all_entries.append({
                'id': str(comment.get('comment_id', f"comment_{len(all_entries)}")),
                'text': comment_text,
                'source': str(comment.get('Source', post.get('Source', 'unknown'))),
                'author': clean_text(comment.get('Author', '') or ''),
                'date': str(comment.get('Date', '') or ''),
                'score': int(comment.get('Score', 0) or 0),
                'title': clean_text(post.get('Title', '') or ''),
                'type': 'comment',
                'post_id': post_id,        # link to parent post
                'label': str(comment.get('Overall_Document_Polarity', 'neutral') or 'neutral'),
                'subjectivity': str(comment.get('Subjectivity', 'objective') or 'objective'),
                'subjectivity_score': float(comment.get('Subjectivity_Score', 0.0) or 0.0),
                'aspect_sentiments': asp_str,
                'entities': entities,
                'ai_tools': ai_tools,
                'aspects': aspects,
                'has_sarcasm': check_sarcasm(comment.get('Targeted_Aspects')),
            })

print(f"\nTotal entries extracted: {len(all_entries):,}")

# Remove duplicates
seen_ids = set()
seen_texts = set()
deduped = []
for entry in all_entries:
    if entry['id'] in seen_ids:
        continue
    text_key = entry['text'][:200]
    if text_key in seen_texts:
        continue
    seen_ids.add(entry['id'])
    seen_texts.add(text_key)
    deduped.append(entry)

all_entries = deduped
print(f"After deduplication: {len(all_entries):,}")

# Summary statistics
labels = Counter(e['label'] for e in all_entries)
types = Counter(e['type'] for e in all_entries)
sources = Counter(e['source'] for e in all_entries)
sarcasm_count = sum(1 for e in all_entries if e['has_sarcasm'])

all_tools = []
for e in all_entries:
    all_tools.extend(t.lower() for t in e['ai_tools'])
top_tools = Counter(all_tools).most_common(15)

print(f"\nLabel distribution:   {dict(labels)}")
print(f"Type distribution:    {dict(types)}")
print(f"Source distribution:  {dict(sources)}")
print(f"Sarcasm detected:     {sarcasm_count:,} ({100*sarcasm_count/len(all_entries):.1f}%)")
print(f"Top AI tools mentioned: {top_tools}")

# Generate embeddings
print(f"\nGenerating embeddings for {len(all_entries):,} documents...")
print("This may take 15-30 minutes depending on hardware.")

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [e['text'] for e in all_entries]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=True
)

# Save
print("\nSaving files...")

with open('indexed_dataset.json', 'w', encoding='utf-8') as f:
    for entry in all_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

np.save('indexed_embeddings.npy', embeddings)

with open('embedding_info.json', 'w') as f:
    json.dump({
        'model': 'all-MiniLM-L6-v2',
        'dimensions': 384,
        'num_documents': len(all_entries),
        'fields': [
            'id', 'text', 'source', 'author', 'date', 'score', 'title',
            'type', 'post_id', 'label', 'subjectivity', 'subjectivity_score',
            'aspect_sentiments', 'entities', 'ai_tools', 'aspects', 'has_sarcasm'
        ],
        'top_ai_tools': [t for t, _ in top_tools],
        'label_distribution': dict(labels),
    }, f, indent=2)

print(f"\n{'='*70}")
print(f"PREPARATION COMPLETE!")
print(f"{'='*70}")
print(f"Documents: {len(all_entries):,}")
print(f"Embeddings shape: {embeddings.shape}")
print(f"\nNew fields added vs raw indexing:")
print(f"  + entities       (all named entities per document)")
print(f"  + ai_tools       (AI tool names only, for tool comparison)")
print(f"  + aspects        (targeted aspect names)")
print(f"  + has_sarcasm    (sarcasm flag)")
print(f"  + post_id        (links comments back to their parent post)")
print(f"  + label          (positive/negative/neutral)")
print(f"  + subjectivity   (subjective/objective)")
print(f"\nNext step: run  python3 index_data.py  to load into Elasticsearch")
print(f"{'='*70}\n")
