#!/usr/bin/env python3
"""
HackerNews Elasticsearch Indexer
Indexes filtered corpus into Elasticsearch for search
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Elasticsearch imports (install with: pip install elasticsearch)
try:
    from elasticsearch import Elasticsearch, helpers
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("⚠️  Warning: elasticsearch package not installed")
    print("   Install with: pip install elasticsearch")


class HNIndexer:
    """Index HackerNews corpus into Elasticsearch"""
    
    def __init__(self, 
                 input_file: str,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 index_name: str = "hackernews_opinions"):
        
        self.input_file = Path(input_file)
        self.index_name = index_name
        
        if ELASTICSEARCH_AVAILABLE:
            self.es = Elasticsearch([f"http://{es_host}:{es_port}"])
        else:
            self.es = None
        
        self.stats = {
            "total_docs": 0,
            "indexed": 0,
            "failed": 0
        }
    
    def load_corpus(self) -> List[Dict]:
        """Load filtered corpus"""
        print(f"📂 Loading corpus from {self.input_file}")
        
        records = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        self.stats["total_docs"] = len(records)
        print(f"   Loaded {len(records)} records")
        
        return records
    
    def create_index_mapping(self):
        """Create Elasticsearch index with proper mapping"""
        mapping = {
            "mappings": {
                "properties": {
                    "source": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "item_id": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},
                    "story_id": {"type": "keyword"},
                    "author": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "created_at_i": {"type": "long"},
                    "url": {"type": "keyword"},
                    "story_title": {"type": "text"},
                    "title": {"type": "text"},
                    "text_raw": {"type": "text"},
                    "text_clean": {
                        "type": "text",
                        "analyzer": "english",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "points": {"type": "integer"},
                    "num_comments": {"type": "integer"},
                    "source_hash": {"type": "keyword"},
                    "text_hash": {"type": "keyword"},
                    "topic_tags": {"type": "keyword"},
                    "matched_categories": {"type": "keyword"},
                    "relevance_score": {"type": "float"},
                    "opinion_score": {"type": "float"},
                    "is_relevant": {"type": "boolean"},
                    
                    # Sentiment fields (to be added later)
                    "sentiment": {"type": "keyword"},
                    "sentiment_score": {"type": "float"},
                    "subjectivity": {"type": "keyword"},
                    "subjectivity_score": {"type": "float"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "english": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            }
        }
        
        # Delete index if exists
        if self.es.indices.exists(index=self.index_name):
            print(f"⚠️  Index '{self.index_name}' exists. Deleting...")
            self.es.indices.delete(index=self.index_name)
        
        # Create new index
        self.es.indices.create(index=self.index_name, body=mapping)
        print(f"✓ Created index '{self.index_name}'")
    
    def prepare_doc_for_indexing(self, record: Dict) -> Dict:
        """Prepare document for Elasticsearch"""
        # Create copy to avoid modifying original
        doc = record.copy()
        
        # Ensure created_at is in proper format
        if "created_at" in doc and doc["created_at"]:
            try:
                # Parse ISO format date
                dt = datetime.fromisoformat(doc["created_at"].replace('Z', '+00:00'))
                doc["created_at"] = dt.isoformat()
            except:
                # Remove if invalid
                doc.pop("created_at", None)
        
        return doc
    
    def bulk_index(self, records: List[Dict], batch_size: int = 500):
        """Index documents in bulk"""
        print(f"\n📤 Indexing {len(records)} documents...")
        
        def generate_actions():
            for record in records:
                doc = self.prepare_doc_for_indexing(record)
                yield {
                    "_index": self.index_name,
                    "_id": record["item_id"],
                    "_source": doc
                }
        
        # Use bulk helper
        success, failed = helpers.bulk(
            self.es,
            generate_actions(),
            chunk_size=batch_size,
            raise_on_error=False,
            stats_only=False
        )
        
        self.stats["indexed"] = success
        self.stats["failed"] = len(failed) if isinstance(failed, list) else 0
        
        print(f"   ✓ Indexed {success} documents")
        if self.stats["failed"] > 0:
            print(f"   ⚠️  Failed: {self.stats['failed']} documents")
    
    def verify_index(self):
        """Verify indexing was successful"""
        print(f"\n🔍 Verifying index...")
        
        # Refresh index
        self.es.indices.refresh(index=self.index_name)
        
        # Get count
        count = self.es.count(index=self.index_name)["count"]
        print(f"   Documents in index: {count}")
        
        # Sample query
        sample = self.es.search(
            index=self.index_name,
            body={
                "query": {"match_all": {}},
                "size": 1
            }
        )
        
        if sample["hits"]["total"]["value"] > 0:
            print(f"   ✓ Index is accessible")
            return True
        else:
            print(f"   ⚠️  No documents found")
            return False
    
    def print_statistics(self):
        """Print indexing statistics"""
        print("\n" + "=" * 70)
        print("📊 INDEXING STATISTICS")
        print("=" * 70)
        print(f"Total documents:          {self.stats['total_docs']}")
        print(f"Successfully indexed:     {self.stats['indexed']}")
        print(f"Failed:                   {self.stats['failed']}")
        print(f"Success rate:             {self.stats['indexed'] / self.stats['total_docs'] * 100:.1f}%")
        print("=" * 70)
    
    def run_sample_queries(self):
        """Run sample queries to test the index"""
        print("\n" + "=" * 70)
        print("🔍 SAMPLE QUERIES")
        print("=" * 70)
        
        queries = [
            ("copilot productivity", "Search for Copilot productivity opinions"),
            ("cursor vs copilot", "Compare Cursor and Copilot"),
            ("AI coding bugs", "Find opinions about AI coding bugs")
        ]
        
        for query_text, description in queries:
            print(f"\n{description}")
            print(f"Query: '{query_text}'")
            
            result = self.es.search(
                index=self.index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["text_clean^2", "story_title", "title"]
                        }
                    },
                    "size": 3
                }
            )
            
            hits = result["hits"]["hits"]
            print(f"Results: {len(hits)} (showing top 3)")
            
            for i, hit in enumerate(hits, 1):
                source = hit["_source"]
                print(f"\n  {i}. [{source.get('type')}] Score: {hit['_score']:.2f}")
                print(f"     Author: {source.get('author')}")
                print(f"     Opinion: {source.get('opinion_score', 0):.2f} | "
                      f"Relevance: {source.get('relevance_score', 0):.2f}")
                text = source.get('text_clean', '')[:150]
                print(f"     Text: {text}...")


def main():
    parser = argparse.ArgumentParser(
        description="Index HackerNews corpus into Elasticsearch"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file (filtered_corpus.jsonl)"
    )
    parser.add_argument(
        "--es-host",
        default="localhost",
        help="Elasticsearch host"
    )
    parser.add_argument(
        "--es-port",
        type=int,
        default=9200,
        help="Elasticsearch port"
    )
    parser.add_argument(
        "--index-name",
        default="hackernews_opinions",
        help="Elasticsearch index name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Bulk indexing batch size"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip index verification"
    )
    
    args = parser.parse_args()
    
    if not ELASTICSEARCH_AVAILABLE:
        print("\n❌ Error: Elasticsearch package not installed")
        print("   Install with: pip install elasticsearch")
        return
    
    # Initialize indexer
    indexer = HNIndexer(
        input_file=args.input,
        es_host=args.es_host,
        es_port=args.es_port,
        index_name=args.index_name
    )
    
    # Check Elasticsearch connection
    if not indexer.es.ping():
        print(f"\n❌ Error: Cannot connect to Elasticsearch at {args.es_host}:{args.es_port}")
        print("   Make sure Elasticsearch is running")
        print("   If using Docker: docker-compose up -d elasticsearch")
        return
    
    print(f"✓ Connected to Elasticsearch at {args.es_host}:{args.es_port}")
    
    # Load corpus
    records = indexer.load_corpus()
    
    # Create index
    indexer.create_index_mapping()
    
    # Index documents
    indexer.bulk_index(records, batch_size=args.batch_size)
    
    # Verify
    if not args.skip_verification:
        indexer.verify_index()
    
    # Print statistics
    indexer.print_statistics()
    
    # Run sample queries
    indexer.run_sample_queries()
    
    print("\n✅ Indexing complete!")
    print(f"   Index '{args.index_name}' is ready for search")


if __name__ == "__main__":
    main()