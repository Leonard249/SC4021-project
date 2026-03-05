#!/usr/bin/env python3
"""
HackerNews Data Preprocessing and Cleaning
Handles text cleaning, deduplication, and relevance filtering
"""

import json
import hashlib
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set
from html import unescape
from collections import Counter

class HNPreprocessor:
    """Preprocess and clean HackerNews data"""
    
    def __init__(self, input_file: str, output_dir: str = "./dataset"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "total_input": 0,
            "empty_text_removed": 0,
            "duplicate_ids_removed": 0,
            "duplicate_text_removed": 0,
            "too_short_removed": 0,
            "total_output": 0
        }
    
    def load_corpus(self) -> List[Dict]:
        """Load raw corpus from JSONL"""
        records = []
        
        print(f"Loading corpus from {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        self.stats["total_input"] = len(records)
        print(f"   Loaded {len(records)} records")
        
        return records
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Unescape HTML entities
        text = unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs (keep the discussion, not the links)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def compute_text_hash(self, text: str) -> str:
        """Compute SHA-1 hash of cleaned text for deduplication"""
        normalized = text.lower().strip()
        return hashlib.sha1(normalized.encode()).hexdigest()
    
    def remove_duplicates(self, records: List[Dict]) -> List[Dict]:
        """Remove duplicate records by ID and text hash"""
        print("\nRemoving duplicates...")
        
        seen_ids: Set[str] = set()
        seen_text_hashes: Set[str] = set()
        unique_records = []
        
        for record in records:
            item_id = record.get("item_id")
            text_hash = record.get("text_hash")
            
            # Check ID duplication
            if item_id in seen_ids:
                self.stats["duplicate_ids_removed"] += 1
                continue
            
            # Check text duplication
            if text_hash in seen_text_hashes:
                self.stats["duplicate_text_removed"] += 1
                continue
            
            seen_ids.add(item_id) # type: ignore
            seen_text_hashes.add(text_hash) # type: ignore
            unique_records.append(record)
        
        print(f"   Removed {self.stats['duplicate_ids_removed']} duplicate IDs")
        print(f"   Removed {self.stats['duplicate_text_removed']} duplicate texts")
        print(f"   Remaining: {len(unique_records)} records")
        
        return unique_records
    
    def filter_empty_and_short(self, records: List[Dict], min_length: int = 20) -> List[Dict]:
        """Remove records with empty or very short text"""
        print(f"\n✂️  Filtering empty and short texts (min length: {min_length} chars)...")
        
        filtered = []
        
        for record in records:
            text_clean = record.get("text_clean", "")
            
            if not text_clean:
                self.stats["empty_text_removed"] += 1
                continue
            
            if len(text_clean) < min_length:
                self.stats["too_short_removed"] += 1
                continue
            
            filtered.append(record)
        
        print(f"   Removed {self.stats['empty_text_removed']} empty texts")
        print(f"   Removed {self.stats['too_short_removed']} texts < {min_length} chars")
        print(f"   Remaining: {len(filtered)} records")
        
        return filtered
    
    def add_basic_topics(self, record: Dict) -> List[str]:
        """Add basic topic tags based on keyword matching"""
        text = (record.get("text_clean", "") + " " + 
                record.get("story_title", "") + " " +
                record.get("title", "")).lower()
        
        topics = []
        
        # AI Tools
        if any(kw in text for kw in ["copilot", "github copilot"]):
            topics.append("copilot")
        if "cursor" in text:
            topics.append("cursor")
        if any(kw in text for kw in ["codex", "openai"]):
            topics.append("codex")
        if any(kw in text for kw in ["codeium", "tabnine"]):
            topics.append("ai-tools")
        
        # Concepts
        if any(kw in text for kw in ["productivity", "efficient", "faster"]):
            topics.append("productivity")
        if any(kw in text for kw in ["vibe", "flow", "experience"]):
            topics.append("developer-experience")
        if any(kw in text for kw in ["hallucination", "bug", "error", "wrong"]):
            topics.append("issues")
        if any(kw in text for kw in ["learn", "junior", "education"]):
            topics.append("learning")
        
        return topics
    
    def preprocess_records(self, records: List[Dict]) -> List[Dict]:
        """Main preprocessing pipeline"""
        print("\n🔧 Preprocessing records...")
        
        processed = []
        
        for i, record in enumerate(records, 1):
            if i % 1000 == 0:
                print(f"   Processed {i}/{len(records)} records...")
            
            # Clean text
            text_raw = record.get("text_raw", "")
            text_clean = self.clean_text(text_raw)
            
            # Update record
            record["text_clean"] = text_clean
            record["text_hash"] = self.compute_text_hash(text_clean)
            
            # Add basic topic tags
            record["topic_tags"] = self.add_basic_topics(record)
            
            processed.append(record)
        
        print(f"   ✓ Preprocessed {len(processed)} records")
        return processed
    
    def save_corpus(self, records: List[Dict], filename: str):
        """Save corpus to JSONL"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"\nSaved {len(records)} records to {output_path}")
        return output_path
    
    def generate_statistics(self, records: List[Dict]) -> Dict:
        """Generate corpus statistics"""
        stats = {
            "total_records": len(records),
            "comments": sum(1 for r in records if r.get("type") == "comment"),
            "stories": sum(1 for r in records if r.get("type") == "story"),
            "unique_authors": len(set(r.get("author") for r in records)),
            "avg_text_length": sum(len(r.get("text_clean", "")) for r in records) / len(records) if records else 0,
            "date_range": {
                "earliest": min((r.get("created_at_i", 0) for r in records), default=0),
                "latest": max((r.get("created_at_i", 0) for r in records), default=0)
            }
        }
        
        # Topic distribution
        all_topics = []
        for record in records:
            all_topics.extend(record.get("topic_tags", []))
        
        stats["topic_distribution"] = dict(Counter(all_topics).most_common(20))
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """Print corpus statistics"""
        print("\n" + "=" * 70)
        print("CORPUS STATISTICS")
        print("=" * 70)
        print(f"Total records:            {stats['total_records']}")
        print(f"  - Comments:             {stats['comments']}")
        print(f"  - Stories:              {stats['stories']}")
        print(f"Unique authors:           {stats['unique_authors']}")
        print(f"Average text length:      {stats['avg_text_length']:.1f} characters")
        
        print(f"\nTop topics:")
        for topic, count in list(stats['topic_distribution'].items())[:10]:
            print(f"  - {topic:25} {count:6,} occurrences")
        
        print("=" * 70)
    
    def print_processing_stats(self):
        """Print processing statistics"""
        print("\n" + "=" * 70)
        print("PROCESSING STATISTICS")
        print("=" * 70)
        print(f"Input records:            {self.stats['total_input']}")
        print(f"Empty text removed:       {self.stats['empty_text_removed']}")
        print(f"Too short removed:        {self.stats['too_short_removed']}")
        print(f"Duplicate IDs removed:    {self.stats['duplicate_ids_removed']}")
        print(f"Duplicate text removed:   {self.stats['duplicate_text_removed']}")
        print(f"Output records:           {self.stats['total_output']}")
        print(f"Retention rate:           {self.stats['total_output'] / self.stats['total_input'] * 100:.1f}%")
        print("=" * 70)
    
    def run_pipeline(self, min_text_length: int = 20) -> List[Dict]:
        """Run full preprocessing pipeline"""
        print("\n" + "=" * 70)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 70)
        
        # Step 1: Load
        records = self.load_corpus()
        
        # Step 2: Preprocess (clean text, compute hashes)
        records = self.preprocess_records(records)
        
        # Step 3: Remove duplicates
        records = self.remove_duplicates(records)
        
        # Step 4: Filter empty/short
        records = self.filter_empty_and_short(records, min_text_length)
        
        self.stats["total_output"] = len(records)
        
        return records


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and clean HackerNews corpus"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file (raw_corpus.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum text length in characters"
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = HNPreprocessor(
        input_file=args.input,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    records = preprocessor.run_pipeline(min_text_length=args.min_length)
    
    # Save cleaned corpus
    preprocessor.save_corpus(records, "cleaned_corpus.jsonl")
    
    # Generate and print statistics
    stats = preprocessor.generate_statistics(records)
    preprocessor.print_statistics(stats)
    preprocessor.print_processing_stats()
    
    # Save statistics
    stats_path = Path(args.output_dir) / "corpus_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to {stats_path}")
    print("\nPreprocessing complete!")
    print(f"   Next step: Run hn_relevance_filter.py to filter by relevance")


if __name__ == "__main__":
    main()