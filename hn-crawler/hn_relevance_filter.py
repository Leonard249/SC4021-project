#!/usr/bin/env python3
"""
HackerNews Relevance Filter
Filters corpus for opinion-rich, project-relevant content
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re

class HNRelevanceFilter:
    """Filter corpus by relevance to AI coding productivity"""
    
    # Strong opinion indicators
    OPINION_INDICATORS = [
        # First person
        r'\b(i|i\'m|i\'ve|i\'ll|my|mine|personally|imo|imho)\b',
        # Evaluative language
        r'\b(love|hate|great|terrible|awful|amazing|brilliant|stupid|useless|essential)\b',
        # Hedges and boosters
        r'\b(really|very|extremely|totally|absolutely|definitely|clearly|obviously)\b',
        # Recommendations
        r'\b(should|shouldn\'t|must|recommend|suggest|avoid|prefer)\b',
        # Experience
        r'\b(found|tried|tested|using|switched|experience|noticed)\b'
    ]
    
    # Project relevance keywords
    RELEVANCE_KEYWORDS = {
        "ai_tools": [
            "copilot", "cursor", "codex", "codeium", "tabnine", 
            "ai assistant", "llm", "gpt", "chatgpt", "claude"
        ],
        "productivity": [
            "productivity", "efficient", "faster", "slower", "speed",
            "accelerate", "boost", "improve", "workflow", "flow state"
        ],
        "vibe_coding": [
            "vibe", "vibe coding", "vibes", "feel", "experience",
            "intuitive", "natural", "seamless"
        ],
        "issues": [
            "hallucination", "bug", "error", "wrong", "mistake",
            "problem", "issue", "regression", "context"
        ],
        "comparison": [
            "better than", "worse than", "compared to", "vs",
            "prefer", "instead of", "replaced"
        ]
    }
    
    def __init__(self, input_file: str, output_dir: str = "./dataset"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "total_input": 0,
            "relevant": 0,
            "not_relevant": 0,
            "high_opinion": 0,
            "medium_opinion": 0,
            "low_opinion": 0
        }
    
    def load_corpus(self) -> List[Dict]:
        """Load cleaned corpus"""
        print(f"📂 Loading corpus from {self.input_file}")
        
        records = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        self.stats["total_input"] = len(records)
        print(f"   Loaded {len(records)} records")
        
        return records
    
    def compute_opinion_score(self, text: str) -> float:
        """Compute opinion score (0-1) based on opinion indicators"""
        text_lower = text.lower()
        
        matches = 0
        for pattern in self.OPINION_INDICATORS:
            matches += len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        # Normalize by text length (per 100 words)
        word_count = len(text_lower.split())
        if word_count == 0:
            return 0.0
        
        normalized_score = (matches / word_count) * 100
        
        # Cap at 1.0
        return min(normalized_score, 1.0)
    
    def compute_relevance_score(self, text: str, title: str = "") -> Tuple[float, List[str]]:
        """
        Compute relevance score (0-1) based on keyword matching
        Returns: (score, matched_categories)
        """
        combined_text = (text + " " + title).lower()
        
        category_matches = {}
        matched_categories = []
        
        for category, keywords in self.RELEVANCE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in combined_text)
            if matches > 0:
                category_matches[category] = matches
                matched_categories.append(category)
        
        # Score based on:
        # 1. Number of categories matched (diversity)
        # 2. Total number of keyword matches (depth)
        category_score = len(category_matches) / len(self.RELEVANCE_KEYWORDS)
        total_matches = sum(category_matches.values())
        match_score = min(total_matches / 5, 1.0)  # Cap at 5 matches
        
        # Weighted combination
        relevance_score = (category_score * 0.6) + (match_score * 0.4)
        
        return relevance_score, matched_categories
    
    def is_relevant(self, record: Dict, 
                   min_relevance: float = 0.15,
                   min_opinion: float = 0.05) -> bool:
        """
        Determine if record is relevant
        
        Criteria:
        1. Must have minimum relevance to AI coding productivity
        2. Should have some opinion signal
        """
        text = record.get("text_clean", "")
        title = record.get("story_title", "") or record.get("title", "")
        
        # Compute scores
        relevance_score, categories = self.compute_relevance_score(text, title)
        opinion_score = self.compute_opinion_score(text)
        
        # Update record
        record["relevance_score"] = round(relevance_score, 3)
        record["opinion_score"] = round(opinion_score, 3)
        record["matched_categories"] = categories
        
        # Decision
        is_relevant = (
            relevance_score >= min_relevance and 
            opinion_score >= min_opinion
        )
        
        record["is_relevant"] = is_relevant
        
        # Track opinion levels
        if opinion_score >= 0.3:
            self.stats["high_opinion"] += 1
        elif opinion_score >= 0.1:
            self.stats["medium_opinion"] += 1
        else:
            self.stats["low_opinion"] += 1
        
        return is_relevant
    
    def filter_corpus(self, records: List[Dict],
                     min_relevance: float = 0.15,
                     min_opinion: float = 0.05) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter corpus by relevance and opinion
        
        Returns: (relevant_records, all_scored_records)
        """
        print(f"\n🔍 Filtering corpus...")
        print(f"   Relevance threshold: {min_relevance}")
        print(f"   Opinion threshold: {min_opinion}")
        
        relevant = []
        
        for i, record in enumerate(records, 1):
            if i % 1000 == 0:
                print(f"   Processed {i}/{len(records)} records...")
            
            if self.is_relevant(record, min_relevance, min_opinion):
                relevant.append(record)
                self.stats["relevant"] += 1
            else:
                self.stats["not_relevant"] += 1
        
        print(f"\n   ✓ Filtered to {len(relevant)} relevant records")
        print(f"   Retention rate: {len(relevant) / len(records) * 100:.1f}%")
        
        return relevant, records
    
    def save_corpus(self, records: List[Dict], filename: str):
        """Save corpus to JSONL"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"\n💾 Saved {len(records)} records to {output_path}")
        return output_path
    
    def print_statistics(self):
        """Print filtering statistics"""
        print("\n" + "=" * 70)
        print("📊 FILTERING STATISTICS")
        print("=" * 70)
        print(f"Total input:              {self.stats['total_input']}")
        print(f"Relevant (kept):          {self.stats['relevant']}")
        print(f"Not relevant (removed):   {self.stats['not_relevant']}")
        print(f"Retention rate:           {self.stats['relevant'] / self.stats['total_input'] * 100:.1f}%")
        
        print(f"\nOpinion distribution:")
        print(f"  High opinion (>0.3):    {self.stats['high_opinion']}")
        print(f"  Medium opinion (0.1-0.3): {self.stats['medium_opinion']}")
        print(f"  Low opinion (<0.1):     {self.stats['low_opinion']}")
        print("=" * 70)
    
    def generate_sample_report(self, records: List[Dict], n: int = 100):
        """Generate sample report for manual validation"""
        print(f"\n📋 Generating sample validation report ({n} records)...")
        
        # Take random sample
        import random
        sample = random.sample(records, min(n, len(records)))
        
        report_path = self.output_dir / "validation_sample.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RELEVANCE FILTER VALIDATION SAMPLE\n")
            f.write("=" * 70 + "\n\n")
            f.write("Instructions: Review these samples to verify filter quality\n")
            f.write("Mark any false positives or false negatives\n\n")
            
            for i, record in enumerate(sample, 1):
                f.write(f"\n{'=' * 70}\n")
                f.write(f"SAMPLE {i}/{len(sample)}\n")
                f.write(f"{'=' * 70}\n")
                f.write(f"ID: {record.get('item_id')}\n")
                f.write(f"Type: {record.get('type')}\n")
                f.write(f"Author: {record.get('author')}\n")
                f.write(f"URL: {record.get('url')}\n")
                f.write(f"\nRelevance Score: {record.get('relevance_score', 0):.3f}\n")
                f.write(f"Opinion Score: {record.get('opinion_score', 0):.3f}\n")
                f.write(f"Categories: {', '.join(record.get('matched_categories', []))}\n")
                f.write(f"Is Relevant: {record.get('is_relevant', False)}\n")
                f.write(f"\nTitle: {record.get('story_title', '') or record.get('title', '')}\n")
                f.write(f"\nText (first 500 chars):\n{record.get('text_clean', '')[:500]}...\n")
                f.write(f"\n[ ] CORRECT  [ ] FALSE POSITIVE  [ ] FALSE NEGATIVE\n")
        
        print(f"   Saved validation report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter HackerNews corpus by relevance"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file (cleaned_corpus.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory"
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.15,
        help="Minimum relevance score (0-1)"
    )
    parser.add_argument(
        "--min-opinion",
        type=float,
        default=0.05,
        help="Minimum opinion score (0-1)"
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=100,
        help="Number of samples for validation report"
    )
    
    args = parser.parse_args()
    
    # Initialize filter
    filter_obj = HNRelevanceFilter(
        input_file=args.input,
        output_dir=args.output_dir
    )
    
    # Load corpus
    records = filter_obj.load_corpus()
    
    # Filter
    relevant_records, all_scored = filter_obj.filter_corpus(
        records,
        min_relevance=args.min_relevance,
        min_opinion=args.min_opinion
    )
    
    # Save both corpora
    filter_obj.save_corpus(all_scored, "scored_corpus.jsonl")
    filter_obj.save_corpus(relevant_records, "filtered_corpus.jsonl")
    
    # Print statistics
    filter_obj.print_statistics()
    
    # Generate validation sample
    if args.validation_samples > 0:
        filter_obj.generate_sample_report(relevant_records, args.validation_samples)
    
    print("\n✅ Filtering complete!")
    print(f"   Next step: Run hn_indexer.py to load into Elasticsearch")


if __name__ == "__main__":
    main()