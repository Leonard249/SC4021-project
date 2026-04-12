#!/usr/bin/env python3

from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import time
import warnings
import re
from collections import defaultdict
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

es = Elasticsearch(['http://localhost:9200'])
model = SentenceTransformer('all-MiniLM-L6-v2')
INDEX_NAME = 'ai_coding_search'

# Characters returned as a preview in search results
PREVIEW_CHARS = 600


def extract_year_month(date_str):
    """Extract year-month from various date formats"""
    if not date_str or date_str == '':
        return None
    match = re.search(r'(\d{4})-(\d{2})', str(date_str))
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    match = re.search(r'(\d{4})/(\d{2})', str(date_str))
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def format_result(hit):
    """Format a single Elasticsearch hit into a result dict."""
    src = hit['_source']
    full_text = src.get('text', '')
    preview = full_text[:PREVIEW_CHARS]
    truncated = len(full_text) > PREVIEW_CHARS
    return {
        'id':                 src.get('id', ''),
        'text':               preview + ('...' if truncated else ''),
        'full_length':        len(full_text),
        'truncated':          truncated,
        'score':              hit['_score'],
        'source':             src.get('source', ''),
        'date':               src.get('date', ''),
        'author':             src.get('author', ''),
        'title':              src.get('title', ''),
        'type':               src.get('type', ''),
        'post_id':            src.get('post_id', ''),
        'label':              src.get('label', 'unlabeled'),
        'subjectivity':       src.get('subjectivity', ''),
        'subjectivity_score': src.get('subjectivity_score', 0.0),
        'aspect_sentiments':  src.get('aspect_sentiments', ''),
        'ai_tools':           src.get('ai_tools', []),
        'has_sarcasm':        src.get('has_sarcasm', False),
    }


@app.route('/api/search/keyword', methods=['GET'])
def keyword_search():
    query = request.args.get('q', '')
    size = int(request.args.get('size', 10))
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')

    if not query:
        return jsonify({'error': 'Query required'}), 400

    start = time.time()

    query_body: dict = {
        "query": {"match": {"text": query}},
        "size": size
    }

    if date_from or date_to:
        filters = []
        if date_from:
            filters.append({"range": {"date": {"gte": date_from}}})
        if date_to:
            filters.append({"range": {"date": {"lte": date_to}}})
        query_body["query"] = {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": filters
            }
        }

    result = es.search(index=INDEX_NAME, body=query_body)  # type: ignore
    search_time = (time.time() - start) * 1000

    return jsonify({
        'query': query,
        'method': 'keyword (BM25)',
        'date_from': date_from,
        'date_to': date_to,
        'total_hits': result['hits']['total']['value'],
        'search_time_ms': round(search_time, 2),
        'results': [format_result(hit) for hit in result['hits']['hits']]
    })


@app.route('/api/search/semantic', methods=['GET'])
def semantic_search():
    query = request.args.get('q', '')
    size = int(request.args.get('size', 10))
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')

    if not query:
        return jsonify({'error': 'Query required'}), 400

    start = time.time()
    query_embedding = model.encode([query])[0].tolist()

    base_query: dict = {"match_all": {}}

    if date_from or date_to:
        filters = []
        if date_from:
            filters.append({"range": {"date": {"gte": date_from}}})
        if date_to:
            filters.append({"range": {"date": {"lte": date_to}}})
        base_query = {
            "bool": {
                "must": [{"match_all": {}}],
                "filter": filters
            }
        }

    result = es.search(index=INDEX_NAME, body={  # type: ignore
        "query": {
            "script_score": {
                "query": base_query,
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        },
        "size": size
    })

    search_time = (time.time() - start) * 1000

    return jsonify({
        'query': query,
        'method': 'semantic (embeddings)',
        'date_from': date_from,
        'date_to': date_to,
        'total_hits': result['hits']['total']['value'],
        'search_time_ms': round(search_time, 2),
        'results': [format_result(hit) for hit in result['hits']['hits']]
    })


@app.route('/api/search/hybrid', methods=['GET'])
def hybrid_search():
    query = request.args.get('q', '')
    size = int(request.args.get('size', 10))
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    alpha = 0.5

    if not query:
        return jsonify({'error': 'Query required'}), 400

    start = time.time()
    query_embedding = model.encode([query])[0].tolist()

    base_query: dict = {"match": {"text": query}}

    if date_from or date_to:
        filters = []
        if date_from:
            filters.append({"range": {"date": {"gte": date_from}}})
        if date_to:
            filters.append({"range": {"date": {"lte": date_to}}})
        base_query = {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": filters
            }
        }

    result = es.search(index=INDEX_NAME, body={  # type: ignore
        "query": {
            "script_score": {
                "query": base_query,
                "script": {
                    "source": """
                        double textScore = _score;
                        double semScore = cosineSimilarity(params.qvec, 'embedding') + 1.0;
                        return params.alpha * semScore + (1.0 - params.alpha) * textScore;
                    """,
                    "params": {"qvec": query_embedding, "alpha": alpha}
                }
            }
        },
        "size": size
    })

    search_time = (time.time() - start) * 1000

    return jsonify({
        'query': query,
        'method': f'hybrid (α={alpha})',
        'date_from': date_from,
        'date_to': date_to,
        'total_hits': result['hits']['total']['value'],
        'search_time_ms': round(search_time, 2),
        'results': [format_result(hit) for hit in result['hits']['hits']]
    })


@app.route('/api/search/by_source', methods=['GET'])
def search_by_source():
    """Return documents filtered by a specific source, optionally also matching a query."""
    query = request.args.get('q', '')
    source = request.args.get('source', '')
    size = int(request.args.get('size', 20))

    if not source:
        return jsonify({'error': 'source parameter required'}), 400

    start = time.time()

    must_clause = [{"match": {"text": query}}] if query else [{"match_all": {}}]
    query_body = {
        "query": {
            "bool": {
                "must": must_clause,
                "filter": [{"term": {"source": source}}]
            }
        },
        "size": size
    }

    result = es.search(index=INDEX_NAME, body=query_body)  # type: ignore
    search_time = (time.time() - start) * 1000

    return jsonify({
        'query': query,
        'source': source,
        'total_hits': result['hits']['total']['value'],
        'search_time_ms': round(search_time, 2),
        'results': [format_result(hit) for hit in result['hits']['hits']]
    })


@app.route('/api/search/by_period', methods=['GET'])
def search_by_period():
    """Return documents for a specific year-month period (e.g. 2025-03), optionally filtered by query."""
    query = request.args.get('q', '')
    year_month = request.args.get('year_month', '')   # e.g. "2025-03"
    size = int(request.args.get('size', 20))

    if not year_month:
        return jsonify({'error': 'year_month parameter required (e.g. 2025-03)'}), 400

    start = time.time()

    # Match documents whose date keyword starts with the year-month prefix
    must_clause = [{"match": {"text": query}}] if query else [{"match_all": {}}]
    query_body = {
        "query": {
            "bool": {
                "must": must_clause,
                "filter": [{"prefix": {"date": year_month}}]
            }
        },
        "size": size
    }

    result = es.search(index=INDEX_NAME, body=query_body)  # type: ignore
    search_time = (time.time() - start) * 1000

    return jsonify({
        'query': query,
        'year_month': year_month,
        'total_hits': result['hits']['total']['value'],
        'search_time_ms': round(search_time, 2),
        'results': [format_result(hit) for hit in result['hits']['hits']]
    })


@app.route('/api/timeline', methods=['GET'])
def timeline():
    """Get timeline data grouped by month."""
    query = request.args.get('q', '')

    body: dict = {
        "size": 10000,
        "_source": ["date"]
    }

    if query:
        body["query"] = {"match": {"text": query}}

    result = es.search(index=INDEX_NAME, body=body)  # type: ignore

    timeline_data: dict = defaultdict(int)
    for hit in result['hits']['hits']:
        date_str = hit['_source'].get('date', '')
        year_month = extract_year_month(date_str)
        if year_month:
            timeline_data[year_month] += 1

    sorted_timeline = sorted(timeline_data.items())

    return jsonify({
        'query': query,
        'interval': 'month',
        'timeline': [{'date': date, 'count': count} for date, count in sorted_timeline]
    })


@app.route('/api/facets', methods=['GET'])
def facets():
    """Get faceted breakdown including source×type counts for heatmap."""
    query = request.args.get('q', '')

    facet_body: dict = {
        "size": 0,
        "aggs": {
            "by_source": {
                "terms": {"field": "source", "size": 20},
                "aggs": {
                    # Nested aggregation: for each source, count by type
                    "by_type": {"terms": {"field": "type", "size": 10}}
                }
            },
            "by_type": {
                "terms": {"field": "type"}
            },
            "by_label": {
                "terms": {"field": "label", "size": 10}
            },
            "top_authors": {
                "terms": {"field": "author.keyword", "size": 10}
            }
        }
    }

    if query:
        facet_body["query"] = {"match": {"text": query}}

    result = es.search(index=INDEX_NAME, body=facet_body)  # type: ignore
    aggs = result['aggregations']

    # Build source list
    sources = [
        {'source': b['key'], 'count': b['doc_count']}
        for b in aggs['by_source']['buckets']
    ]

    # Build global type list
    all_types = [b['key'] for b in aggs['by_type']['buckets']]

    # Build source×type matrix (only sources with count > 0)
    heatmap_sources = []
    heatmap_matrix = []   # list of rows, one per source; each row has one value per type

    for src_bucket in aggs['by_source']['buckets']:
        if src_bucket['doc_count'] == 0:
            continue
        type_counts = {tb['key']: tb['doc_count'] for tb in src_bucket['by_type']['buckets']}
        row = [type_counts.get(t, 0) for t in all_types]
        if any(v > 0 for v in row):
            heatmap_sources.append(src_bucket['key'])
            heatmap_matrix.append(row)

    return jsonify({
        'query': query,
        'facets': {
            'sources': sources,
            'types': [{'type': b['key'], 'count': b['doc_count']} for b in aggs['by_type']['buckets']],
            'labels': [{'label': b['key'], 'count': b['doc_count']} for b in aggs['by_label']['buckets']],
            'top_authors': [
                {'author': b['key'] if b['key'] else 'Anonymous', 'count': b['doc_count']}
                for b in aggs['top_authors']['buckets'][:10]
            ],
            'heatmap': {
                'sources': heatmap_sources,
                'types':   all_types,
                'matrix':  heatmap_matrix
            }
        }
    })


@app.route('/api/sentiment_trend', methods=['GET'])
def sentiment_trend():
    """
    Returns month-by-month sentiment breakdown (positive/negative/neutral counts).
    Useful for plotting a sentiment trend chart over time.
    """
    query = request.args.get('q', '')

    body: dict = {
        "size": 10000,
        "_source": ["date", "label"]
    }
    if query:
        body["query"] = {"match": {"text": query}}

    result = es.search(index=INDEX_NAME, body=body)  # type: ignore

    monthly: dict = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})

    for hit in result['hits']['hits']:
        ym = extract_year_month(hit['_source'].get('date', ''))
        label = hit['_source'].get('label', 'neutral')
        if ym:
            if label in ('positive', 'negative', 'neutral'):
                monthly[ym][label] += 1
            monthly[ym]['total'] += 1

    sorted_months = sorted(monthly.items())
    return jsonify({
        'query': query,
        'trend': [
            {
                'date':     ym,
                'positive': counts['positive'],
                'negative': counts['negative'],
                'neutral':  counts['neutral'],
                'total':    counts['total'],
                'positive_pct': round(100 * counts['positive'] / counts['total'], 1) if counts['total'] else 0,
                'negative_pct': round(100 * counts['negative'] / counts['total'], 1) if counts['total'] else 0,
            }
            for ym, counts in sorted_months
        ]
    })


@app.route('/api/tool_comparison', methods=['GET'])
def tool_comparison():
    """
    Compare sentiment distribution across top AI tools (Claude, Copilot, Cursor, etc.).
    For each tool, aggregates documents that mention it and returns label counts.
    """
    query = request.args.get('q', '')

    # Canonical mapping: lowercase aliases → display name
    canonical = {
        'claude': 'Claude',
        'copilot': 'Copilot',
        'github copilot': 'Copilot',
        'cursor': 'Cursor',
        'chatgpt': 'ChatGPT',
        'gemini': 'Gemini',
    }

    base_query: dict = {"match": {"text": query}} if query else {"match_all": {}}

    # Build a multi-filter aggregation: for each tool, filter docs mentioning it then bucket by label
    agg_body: dict = {
        "size": 0,
        "query": base_query,
        "aggs": {}
    }

    unique_tools = list(canonical.values())
    for tool_canonical in unique_tools:
        # Collect all alias keys for this tool
        aliases = [k for k, v in canonical.items() if v == tool_canonical]
        agg_body["aggs"][f"tool_{tool_canonical}"] = {
            "filter": {"terms": {"ai_tools": aliases}},
            "aggs": {
                "by_label": {"terms": {"field": "label", "size": 5}},
                "by_month": {
                    "terms": {"field": "date", "size": 200}
                }
            }
        }

    result = es.search(index=INDEX_NAME, body=agg_body)  # type: ignore
    aggs = result['aggregations']

    comparison = []
    for tool_canonical in unique_tools:
        agg_key = f"tool_{tool_canonical}"
        if agg_key not in aggs:
            continue
        tool_agg = aggs[agg_key]
        total = tool_agg['doc_count']
        if total == 0:
            continue

        labels = {b['key']: b['doc_count'] for b in tool_agg['by_label']['buckets']}
        pos = labels.get('positive', 0)
        neg = labels.get('negative', 0)
        neu = labels.get('neutral', 0)

        comparison.append({
            'tool':         tool_canonical,
            'total':        total,
            'positive':     pos,
            'negative':     neg,
            'neutral':      neu,
            'positive_pct': round(100 * pos / total, 1) if total else 0,
            'negative_pct': round(100 * neg / total, 1) if total else 0,
            'neutral_pct':  round(100 * neu / total, 1) if total else 0,
        })

    # Sort by total mentions descending
    comparison.sort(key=lambda x: x['total'], reverse=True)

    return jsonify({
        'query': query,
        'tools': comparison
    })


@app.route('/api/entity_search', methods=['GET'])
def entity_search():
    """Search documents by a specific named entity (e.g., a specific AI tool name)."""
    entity = request.args.get('entity', '')
    query = request.args.get('q', '')
    size = int(request.args.get('size', 20))

    if not entity:
        return jsonify({'error': 'entity parameter required'}), 400

    start = time.time()

    must = [{"match": {"text": query}}] if query else [{"match_all": {}}]
    body = {
        "query": {
            "bool": {
                "must": must,
                "filter": [{"term": {"ai_tools": entity.lower()}}]
            }
        },
        "size": size
    }

    result = es.search(index=INDEX_NAME, body=body)  # type: ignore
    search_time = (time.time() - start) * 1000

    return jsonify({
        'entity': entity,
        'query': query,
        'total_hits': result['hits']['total']['value'],
        'search_time_ms': round(search_time, 2),
        'results': [format_result(hit) for hit in result['hits']['hits']]
    })


@app.route('/api/top_entities', methods=['GET'])
def top_entities():
    """Return top AI tool entities mentioned in results for a query."""
    query = request.args.get('q', '')

    body: dict = {
        "size": 0,
        "aggs": {
            "top_tools": {"terms": {"field": "ai_tools", "size": 20}},
            "top_aspects": {"terms": {"field": "aspects", "size": 20}},
            "by_label": {"terms": {"field": "label", "size": 5}}
        }
    }
    if query:
        body["query"] = {"match": {"text": query}}

    result = es.search(index=INDEX_NAME, body=body)  # type: ignore
    aggs = result['aggregations']

    return jsonify({
        'query': query,
        'top_tools': [
            {'tool': b['key'], 'count': b['doc_count']}
            for b in aggs['top_tools']['buckets']
        ],
        'top_aspects': [
            {'aspect': b['key'], 'count': b['doc_count']}
            for b in aggs['top_aspects']['buckets']
        ],
        'label_distribution': [
            {'label': b['key'], 'count': b['doc_count']}
            for b in aggs['by_label']['buckets']
        ]
    })


@app.route('/api/doc/<doc_id>', methods=['GET'])
def get_doc(doc_id):
    """Return the full text and all fields for a single document by its ID."""
    try:
        hit = es.get(index=INDEX_NAME, id=doc_id)
        src = hit['_source']
        return jsonify({
            'id':                 src.get('id', ''),
            'text':               src.get('text', ''),   # full text, no truncation
            'source':             src.get('source', ''),
            'date':               src.get('date', ''),
            'author':             src.get('author', ''),
            'title':              src.get('title', ''),
            'type':               src.get('type', ''),
            'post_id':            src.get('post_id', ''),
            'score':              src.get('score', 0),
            'label':              src.get('label', ''),
            'subjectivity':       src.get('subjectivity', ''),
            'subjectivity_score': src.get('subjectivity_score', 0.0),
            'aspect_sentiments':  src.get('aspect_sentiments', ''),
            'ai_tools':           src.get('ai_tools', []),
            'has_sarcasm':        src.get('has_sarcasm', False),
        })
    except Exception:
        return jsonify({'error': f'Document {doc_id} not found'}), 404


@app.route('/api/doc/<doc_id>/siblings', methods=['GET'])
def get_siblings(doc_id):
    """
    Given a comment's doc_id, fetch the parent post and sibling comments.
    Uses the post_id field to find all documents sharing the same thread.
    """
    size = int(request.args.get('size', 20))
    try:
        hit = es.get(index=INDEX_NAME, id=doc_id)
        post_id = hit['_source'].get('post_id', '')
        if not post_id:
            return jsonify({'error': 'No post_id for this document'}), 404

        result = es.search(index=INDEX_NAME, body={  # type: ignore
            "query": {"term": {"post_id": post_id}},
            "size": size,
            "sort": [{"type": {"order": "asc"}}, {"score": {"order": "desc"}}]
        })

        docs = []
        for h in result['hits']['hits']:
            src = h['_source']
            docs.append({
                'id':      src.get('id', ''),
                'text':    src.get('text', ''),
                'type':    src.get('type', ''),
                'author':  src.get('author', ''),
                'date':    src.get('date', ''),
                'label':   src.get('label', ''),
                'is_current': src.get('id', '') == doc_id,
            })

        return jsonify({
            'post_id':   post_id,
            'total':     result['hits']['total']['value'],
            'documents': docs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/stats', methods=['GET'])
def stats():
    count = es.count(index=INDEX_NAME)['count']
    return jsonify({
        'total_documents': count,
        'index_name': INDEX_NAME,
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_dimensions': 384
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("ENHANCED SEARCH ENGINE WITH SENTIMENT ANALYSIS")
    print("=" * 70)
    print("\nInnovations:")
    print("  ⏰ Timeline Search     - Filter and visualize by date; click to drill down")
    print("  📊 Multifaceted Search - Breakdown by source/type/sentiment; click to drill down")
    print("  🧠 Sentiment Aware     - Labels, subjectivity, aspect sentiments in results")
    print("  📈 Enhanced Analytics  - 3D scatter, word clouds, score distributions")
    print("\nAPI Endpoints:")
    print("  /api/search/keyword?q=claude&date_from=2025-01-01")
    print("  /api/search/semantic?q=claude")
    print("  /api/search/hybrid?q=claude")
    print("  /api/search/by_source?q=claude&source=HackerNews")
    print("  /api/search/by_period?q=claude&year_month=2025-03")
    print("  /api/timeline?q=claude")
    print("  /api/facets?q=claude")
    print("  /api/stats")
    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop\n")
    app.run(port=5001, debug=True)
