#!/usr/bin/env python3
"""
HackerNews Opinion Crawler for AI Coding Productivity Project
Uses Algolia HN Search API to collect opinionated comments and stories
"""

import requests
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Set
import argparse
from pathlib import Path

class HNScraper:
    """Crawler for HackerNews using Algolia Search API"""
    
    ALGOLIA_BASE = "https://hn.algolia.com/api/v1"
    
    # Keywords targeting AI coding productivity debate
    KEYWORDS = [
        # AI Tools
        "vibe coding",
        "copilot",
        "cursor",
        "github copilot",
        "codex",
        "codeium",
        "tabnine",
        "AI pair programming",
        "LLM coding",
        "AI code completion",
        "AI coding assistant",
        
        # Productivity terms
        "developer productivity",
        "coding productivity",
        "programming efficiency",
        "faster coding",
        "code faster",
        
        # Debate language (positive)
        "accelerate development",
        "boost productivity",
        "coding flow",
        "speeds up coding",
        
        # Debate language (negative)
        "AI hallucination",
        "buggy code",
        "slower coding",
        "learning curve",
        "context switching",
        "technical debt",
        
        # Related concepts
        "AI code review",
        "automated coding",
        "SWE agent",
        "agentic coding"
    ]
    
    def __init__(self, output_dir: str = "./dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seen_ids: Set[str] = set()
        self.stats = {
            "comments_fetched": 0,
            "stories_fetched": 0,
            "api_calls": 0,
            "duplicates_skipped": 0
        }
        
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with rate limiting"""
        url = f"{self.ALGOLIA_BASE}/{endpoint}"
        self.stats["api_calls"] += 1
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(0.5)  # Rate limiting
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return {}
    
    def search_comments(self, query: str, num_results: int = 1000) -> List[Dict]:
        """Search for comments matching a query"""
        print(f"\n📝 Searching comments for: '{query}'")
        
        all_hits = []
        page = 0
        hits_per_page = 100  # Algolia max
        
        while len(all_hits) < num_results:
            params = {
                "query": query,
                "tags": "comment",
                "hitsPerPage": hits_per_page,
                "page": page
            }
            
            data = self._make_request("search", params)
            hits = data.get("hits", [])
            
            if not hits:
                break
                
            all_hits.extend(hits)
            page += 1
            
            print(f"  Retrieved {len(all_hits)} comments so far...")
            
            if page >= data.get("nbPages", 1):
                break
        
        print(f"  ✓ Total: {len(all_hits)} comments")
        return all_hits[:num_results]
    
    def search_stories(self, query: str, num_results: int = 200) -> List[Dict]:
        """Search for stories (Ask HN, Show HN) matching a query"""
        print(f"\n📰 Searching stories for: '{query}'")
        
        all_hits = []
        page = 0
        hits_per_page = 100
        
        while len(all_hits) < num_results:
            params = {
                "query": query,
                "tags": "story",
                "hitsPerPage": hits_per_page,
                "page": page
            }
            
            data = self._make_request("search", params)
            hits = data.get("hits", [])
            
            if not hits:
                break
                
            all_hits.extend(hits)
            page += 1
            
            print(f"  Retrieved {len(all_hits)} stories so far...")
            
            if page >= data.get("nbPages", 1):
                break
        
        print(f"  ✓ Total: {len(all_hits)} stories")
        return all_hits[:num_results]
    
    def get_item_details(self, item_id: str) -> Dict:
        """Get full item details including text"""
        params = {}
        data = self._make_request(f"items/{item_id}", params)
        return data
    
    def normalize_comment(self, hit: Dict) -> Dict:
        """Convert Algolia comment hit to our schema"""
        item_id = str(hit.get("objectID", ""))
        
        # Skip if already seen
        if item_id in self.seen_ids:
            self.stats["duplicates_skipped"] += 1
            return None
        
        self.seen_ids.add(item_id)
        self.stats["comments_fetched"] += 1
        
        # Extract and clean text
        comment_text = hit.get("comment_text", "")
        if not comment_text:
            return None
        
        # Create unique hash
        source_hash = hashlib.sha1(
            f"hackernews_{item_id}".encode()
        ).hexdigest()
        
        record = {
            "source": "hackernews",
            "type": "comment",
            "item_id": item_id,
            "parent_id": str(hit.get("parent_id", "")),
            "story_id": str(hit.get("story_id", "")),
            "author": hit.get("author", "unknown"),
            "created_at": hit.get("created_at", ""),
            "created_at_i": hit.get("created_at_i", 0),
            "url": f"https://news.ycombinator.com/item?id={item_id}",
            "story_title": hit.get("story_title", ""),
            "story_url": hit.get("story_url", ""),
            "text_raw": comment_text,
            "text_clean": "",  # Will be filled by preprocessing
            "points": hit.get("points", 0),
            "source_hash": source_hash,
            "text_hash": "",  # Will be filled by preprocessing
            "topic_tags": [],  # Will be filled by classification
            "relevance_score": None,  # Will be filled by filter
            "is_relevant": None  # Will be filled by filter
        }
        
        return record
    
    def normalize_story(self, hit: Dict) -> Dict:
        """Convert Algolia story hit to our schema"""
        item_id = str(hit.get("objectID", ""))
        
        # Skip if already seen
        if item_id in self.seen_ids:
            self.stats["duplicates_skipped"] += 1
            return None
        
        self.seen_ids.add(item_id)
        self.stats["stories_fetched"] += 1
        
        # Extract text (for Ask HN, Show HN)
        story_text = hit.get("story_text", "")
        title = hit.get("title", "")
        
        # Create unique hash
        source_hash = hashlib.sha1(
            f"hackernews_{item_id}".encode()
        ).hexdigest()
        
        record = {
            "source": "hackernews",
            "type": "story",
            "item_id": item_id,
            "parent_id": "",
            "story_id": item_id,
            "author": hit.get("author", "unknown"),
            "created_at": hit.get("created_at", ""),
            "created_at_i": hit.get("created_at_i", 0),
            "url": f"https://news.ycombinator.com/item?id={item_id}",
            "story_title": title,
            "story_url": hit.get("url", ""),
            "title": title,
            "text_raw": story_text,
            "text_clean": "",
            "points": hit.get("points", 0),
            "num_comments": hit.get("num_comments", 0),
            "source_hash": source_hash,
            "text_hash": "",
            "topic_tags": [],
            "relevance_score": None,
            "is_relevant": None
        }
        
        return record
    
    def crawl_by_keywords(self, 
                         comments_per_keyword: int = 1000,
                         stories_per_keyword: int = 100,
                         keywords: List[str] = None) -> List[Dict]:
        """Main crawling method: fetch comments and stories for each keyword"""
        
        if keywords is None:
            keywords = self.KEYWORDS
        
        all_records = []
        
        print(f"\n🚀 Starting HackerNews crawl with {len(keywords)} keywords")
        print(f"   Target: {comments_per_keyword} comments + {stories_per_keyword} stories per keyword")
        print("=" * 70)
        
        for i, keyword in enumerate(keywords, 1):
            print(f"\n[{i}/{len(keywords)}] Processing: {keyword}")
            print("-" * 70)
            
            # Fetch comments
            comment_hits = self.search_comments(keyword, comments_per_keyword)
            for hit in comment_hits:
                record = self.normalize_comment(hit)
                if record:
                    all_records.append(record)
            
            # Fetch stories (Ask HN, Show HN with text)
            story_hits = self.search_stories(keyword, stories_per_keyword)
            for hit in story_hits:
                # Only keep stories with text content
                if hit.get("story_text"):
                    record = self.normalize_story(hit)
                    if record:
                        all_records.append(record)
            
            # Progress update
            print(f"\n📊 Progress: {len(all_records)} total records collected")
            print(f"   API calls: {self.stats['api_calls']}")
            print(f"   Duplicates skipped: {self.stats['duplicates_skipped']}")
        
        return all_records
    
    def save_raw_corpus(self, records: List[Dict], filename: str = "raw_corpus.jsonl"):
        """Save raw corpus to JSONL"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Saved {len(records)} records to {output_path}")
        return output_path
    
    def print_stats(self):
        """Print crawling statistics"""
        print("\n" + "=" * 70)
        print("📈 CRAWLING STATISTICS")
        print("=" * 70)
        print(f"Total API calls:          {self.stats['api_calls']}")
        print(f"Comments fetched:         {self.stats['comments_fetched']}")
        print(f"Stories fetched:          {self.stats['stories_fetched']}")
        print(f"Duplicates skipped:       {self.stats['duplicates_skipped']}")
        print(f"Total unique records:     {self.stats['comments_fetched'] + self.stats['stories_fetched']}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Crawl HackerNews for AI coding productivity opinions"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory for crawled data"
    )
    parser.add_argument(
        "--comments-per-keyword",
        type=int,
        default=1000,
        help="Number of comments to fetch per keyword"
    )
    parser.add_argument(
        "--stories-per-keyword",
        type=int,
        default=100,
        help="Number of stories to fetch per keyword"
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        help="Custom keywords (default: use built-in list)"
    )
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = HNScraper(output_dir=args.output_dir)
    
    # Crawl
    records = scraper.crawl_by_keywords(
        comments_per_keyword=args.comments_per_keyword,
        stories_per_keyword=args.stories_per_keyword,
        keywords=args.keywords
    )
    
    # Save raw corpus
    scraper.save_raw_corpus(records)
    
    # Print statistics
    scraper.print_stats()
    
    print("\n✅ Crawling complete!")
    print(f"   Next step: Run hn_preprocess.py to clean and filter the data")


if __name__ == "__main__":
    main()