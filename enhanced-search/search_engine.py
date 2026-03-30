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

def extract_year_month(date_str):
    """Extract year-month from various date formats"""
    if not date_str or date_str == '':
        return None
    # Try to extract YYYY-MM pattern
    match = re.search(r'(\d{4})-(\d{2})', str(date_str))
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    # Try YYYY/MM pattern
    match = re.search(r'(\d{4})/(\d{2})', str(date_str))
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None

@app.route('/api/search/keyword', methods=['GET'])
def keyword_search():
    query = request.args.get('q', '')
    size = int(request.args.get('size', 10))
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    start = time.time()
    
    query_body = {
        "query": {"match": {"text": query}},
        "size": size
    }
    
    # Add date filter if provided (works with keyword field)
    if date_from or date_to:
        bool_query = {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": []
            }
        }
        
        if date_from:
            bool_query["bool"]["filter"].append({
                "range": {"date": {"gte": date_from}}
            })
        if date_to:
            bool_query["bool"]["filter"].append({
                "range": {"date": {"lte": date_to}}
            })
        
        query_body["query"] = bool_query
    
    result = es.search(index=INDEX_NAME, body=query_body) # type: ignore
    search_time = (time.time() - start) * 1000
    
    return jsonify({
        'query': query,
        'method': 'keyword (BM25)',
        'date_from': date_from,
        'date_to': date_to,
        'total_hits': result['hits']['total']['value'],
        'search_time_ms': round(search_time, 2),
        'results': [{
            'text': hit['_source']['text'][:200] + '...',
            'score': hit['_score'],
            'source': hit['_source']['source'],
            'date': hit['_source'].get('date', ''),
            'author': hit['_source'].get('author', ''),
            'label': hit['_source'].get('label', 'unlabeled')
        } for hit in result['hits']['hits']]
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
    
    base_query = {"match_all": {}}
    
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
    
    result = es.search(
        index=INDEX_NAME,
        body={  # type: ignore
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
        }
    )
    
    search_time = (time.time() - start) * 1000
    
    return jsonify({
        'query': query,
        'method': 'semantic (embeddings)',
        'date_from': date_from,
        'date_to': date_to,
        'total_hits': result['hits']['total']['value'],
        'search_time_ms': round(search_time, 2),
        'results': [{
            'text': hit['_source']['text'][:200] + '...',
            'score': hit['_score'],
            'source': hit['_source']['source'],
            'date': hit['_source'].get('date', ''),
            'author': hit['_source'].get('author', ''),
            'label': hit['_source'].get('label', 'unlabeled')
        } for hit in result['hits']['hits']]
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
    
    base_query = {"match": {"text": query}}
    
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
    
    result = es.search(
        index=INDEX_NAME,
        body={  # type: ignore
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
        }
    )
    
    search_time = (time.time() - start) * 1000
    
    return jsonify({
        'query': query,
        'method': f'hybrid (α={alpha})',
        'date_from': date_from,
        'date_to': date_to,
        'total_hits': result['hits']['total']['value'],
        'search_time_ms': round(search_time, 2),
        'results': [{
            'text': hit['_source']['text'][:200] + '...',
            'score': hit['_score'],
            'source': hit['_source']['source'],
            'date': hit['_source'].get('date', ''),
            'author': hit['_source'].get('author', ''),
            'label': hit['_source'].get('label', 'unlabeled')
        } for hit in result['hits']['hits']]
    })

@app.route('/api/timeline', methods=['GET'])
def timeline():
    """Get timeline using client-side date extraction"""
    query = request.args.get('q', '')
    
    # Get all results (or large sample)
    body = {
        "size": 10000,  # Get large sample
        "_source": ["date"]
    }
    
    if query:
        body["query"] = {"match": {"text": query}}
    
    result = es.search(index=INDEX_NAME, body=body)  # type: ignore
    
    # Group by year-month on client side
    timeline_data = defaultdict(int)
    
    for hit in result['hits']['hits']:
        date_str = hit['_source'].get('date', '')
        year_month = extract_year_month(date_str)
        if year_month:
            timeline_data[year_month] += 1
    
    # Sort by date
    sorted_timeline = sorted(timeline_data.items())
    
    return jsonify({
        'query': query,
        'interval': 'month',
        'timeline': [{
            'date': date,
            'count': count
        } for date, count in sorted_timeline]
    })

@app.route('/api/facets', methods=['GET'])
def facets():
    """Get faceted breakdown"""
    query = request.args.get('q', '')
    
    facet_body = {
        "size": 0,
        "aggs": {
            "by_source": {
                "terms": {"field": "source", "size": 20}
            },
            "by_type": {
                "terms": {"field": "type"}
            },
            "top_authors": {
                "terms": {"field": "author.keyword", "size": 10}
            }
        }
    }
    
    if query:
        facet_body["query"] = {"match": {"text": query}}
    
    result = es.search(index=INDEX_NAME, body=facet_body)  # type: ignore
    
    return jsonify({
        'query': query,
        'facets': {
            'sources': [{
                'source': b['key'],
                'count': b['doc_count']
            } for b in result['aggregations']['by_source']['buckets']],
            'types': [{
                'type': b['key'],
                'count': b['doc_count']
            } for b in result['aggregations']['by_type']['buckets']],
            'top_authors': [{
                'author': b['key'] if b['key'] else 'Anonymous',
                'count': b['doc_count']
            } for b in result['aggregations']['top_authors']['buckets'][:10]]
        }
    })

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
    print("\n" + "="*70)
    print("🚀 ENHANCED SEARCH ENGINE WITH ALL FEATURES")
    print("="*70)
    print("\nInnovations:")
    print("  ⏰ Timeline Search - Filter and visualize by date")
    print("  📊 Multifaceted Search - Breakdown by source/type/author")
    print("  📈 Enhanced Visualizations - Charts and analytics")
    print("\nAPI Endpoints:")
    print("  📍 Search:   http://localhost:5001/api/search/keyword?q=claude&date_from=2025-01-01")
    print("  📍 Timeline: http://localhost:5001/api/timeline?q=claude")
    print("  📍 Facets:   http://localhost:5001/api/facets?q=claude")
    print("  📍 Stats:    http://localhost:5001/api/stats")
    print("\n" + "="*70)
    print("Press Ctrl+C to stop\n")
    app.run(port=5001, debug=True)
