#!/usr/bin/env python3
"""
Evaluation Dataset Generator
Creates balanced sample for manual annotation of sentiment and subjectivity
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict
import csv

class EvalDatasetGenerator:
    """Generate evaluation dataset for manual labeling"""
    
    def __init__(self, input_file: str, output_dir: str = "./dataset"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_corpus(self) -> List[Dict]:
        """Load filtered corpus"""
        print(f"📂 Loading corpus from {self.input_file}")
        
        records = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        print(f"   Loaded {len(records)} records")
        return records
    
    def stratified_sample(self, records: List[Dict], 
                         n: int = 1000) -> List[Dict]:
        """
        Create stratified sample balancing:
        - Opinion levels (high/medium/low)
        - Content types (comment/story)
        - Topics
        """
        print(f"\n🎯 Creating stratified sample of {n} records...")
        
        # Group by opinion level
        high_opinion = [r for r in records if r.get("opinion_score", 0) >= 0.3]
        medium_opinion = [r for r in records if 0.1 <= r.get("opinion_score", 0) < 0.3]
        low_opinion = [r for r in records if r.get("opinion_score", 0) < 0.1]
        
        print(f"   High opinion: {len(high_opinion)}")
        print(f"   Medium opinion: {len(medium_opinion)}")
        print(f"   Low opinion: {len(low_opinion)}")
        
        # Sample proportionally (but ensure minimum representation)
        target_high = max(int(n * 0.5), 100)  # 50% high opinion
        target_medium = max(int(n * 0.3), 100)  # 30% medium
        target_low = max(int(n * 0.2), 50)  # 20% low
        
        sample = []
        
        # Sample from each group
        sample.extend(random.sample(high_opinion, min(target_high, len(high_opinion))))
        sample.extend(random.sample(medium_opinion, min(target_medium, len(medium_opinion))))
        sample.extend(random.sample(low_opinion, min(target_low, len(low_opinion))))
        
        # Shuffle
        random.shuffle(sample)
        
        # Trim to exact size
        sample = sample[:n]
        
        print(f"   ✓ Created sample of {len(sample)} records")
        
        return sample
    
    def generate_excel_template(self, sample: List[Dict], 
                                filename: str = "eval_dataset.csv"):
        """
        Generate CSV template for manual annotation
        Format matches benchmark requirements
        """
        output_path = self.output_dir / filename
        
        print(f"\n📝 Generating evaluation template...")
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # Header matching benchmark format
            writer.writerow([
                "id",
                "source",
                "type",
                "author",
                "url",
                "story_title",
                "text",
                "text_length",
                "opinion_score_auto",
                "relevance_score_auto",
                "",
                "subjectivity",  # LABEL: neutral / opinionated
                "polarity",  # LABEL: positive / negative / neutral
                "notes",
                "annotator"
            ])
            
            # Data rows
            for record in sample:
                text = record.get("text_clean", "")
                
                writer.writerow([
                    record.get("item_id"),
                    record.get("source"),
                    record.get("type"),
                    record.get("author"),
                    record.get("url"),
                    record.get("story_title", ""),
                    text,
                    len(text),
                    record.get("opinion_score", ""),
                    record.get("relevance_score", ""),
                    "",
                    "",  # TO LABEL: subjectivity
                    "",  # TO LABEL: polarity
                    "",  # notes
                    ""   # annotator
                ])
        
        print(f"   ✓ Saved template to {output_path}")
        return output_path
    
    def generate_annotation_guidelines(self, filename: str = "annotation_guidelines.txt"):
        """Generate annotation guidelines for labelers"""
        output_path = self.output_dir / filename
        
        guidelines = """
ANNOTATION GUIDELINES FOR HACKERNEWS OPINION DATASET
=====================================================

OBJECTIVE:
Label 1,000+ HackerNews comments/stories for subjectivity and sentiment
to evaluate the opinion search engine's classification accuracy.

REQUIREMENTS:
- Minimum 1,000 labeled records
- ≥80% inter-annotator agreement
- Two annotators per record (recommended)

=====================================================
STEP 1: SUBJECTIVITY LABELING
=====================================================

Label each record as either:

NEUTRAL:
- Factual statements without personal opinion
- Technical descriptions
- Objective information
- Examples: "The API returns JSON", "It was released in 2023"

OPINIONATED:
- Personal experiences, feelings, judgments
- Recommendations, preferences
- Evaluative language (good/bad, love/hate)
- First-person statements (I think, I found)
- Examples: "Copilot is amazing", "I hate the autocomplete"

GUIDELINES:
- If uncertain, mark as NEUTRAL
- Mixed content: choose the dominant characteristic
- Sarcasm/irony: mark as OPINIONATED

=====================================================
STEP 2: POLARITY LABELING (OPINIONATED RECORDS ONLY)
=====================================================

For records labeled OPINIONATED, assign polarity:

POSITIVE:
- Favorable opinion about AI coding tools
- Benefits, advantages, improvements mentioned
- Examples: "productivity boost", "saves time", "amazing"

NEGATIVE:
- Unfavorable opinion about AI coding tools
- Problems, issues, criticisms mentioned
- Examples: "buggy", "slows me down", "hallucinations"

NEUTRAL:
- Mixed sentiment (both positive and negative)
- Factual personal experience without clear judgment
- Comparative statements without preference
- Examples: "different but not better", "works for some tasks"

GUIDELINES:
- Focus on the overall sentiment about AI coding tools
- If comparing (e.g., "X better than Y"), consider net sentiment
- Mixed reviews: mark as NEUTRAL if roughly balanced

=====================================================
ANNOTATION WORKFLOW
=====================================================

1. Open eval_dataset.csv in Excel or Google Sheets

2. For each row:
   a. Read the 'text' column
   b. Fill in 'subjectivity' column: neutral OR opinionated
   c. If opinionated, fill in 'polarity': positive OR negative OR neutral
   d. Add any notes to 'notes' column
   e. Put your name/ID in 'annotator' column

3. Use filters to spot-check consistency:
   - High opinion_score_auto → should mostly be opinionated
   - Low opinion_score_auto → check for false negatives

4. Save frequently to avoid losing work

=====================================================
INTER-ANNOTATOR AGREEMENT
=====================================================

To achieve ≥80% agreement:

1. Two annotators label the same records independently
2. Calculate agreement: (matching labels) / (total records)
3. Discuss disagreements and establish consensus
4. Document decision rules for edge cases

COMMON EDGE CASES:
- Sarcasm → Opinionated (even if factual sounding)
- Technical comparisons → Neutral (unless judgment expressed)
- "I use X" → Neutral (unless accompanied by evaluation)
- "I love/hate X" → Opinionated Positive/Negative

=====================================================
QUALITY CHECKS
=====================================================

Before submission:
✓ All records have subjectivity label
✓ All opinionated records have polarity label
✓ Annotator name filled in
✓ At least 1,000 records completed
✓ Second annotator has reviewed
✓ Agreement calculated (target: ≥80%)

=====================================================
QUESTIONS?
=====================================================

Refer to project documentation or contact team lead.

Remember: Consistency is more important than perfection.
When in doubt, document your reasoning in the notes column.
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(guidelines)
        
        print(f"\n📋 Generated annotation guidelines: {output_path}")
        return output_path
    
    def generate_quick_reference(self, sample: List[Dict],
                                filename: str = "annotation_quick_reference.txt"):
        """Generate quick reference with examples"""
        output_path = self.output_dir / filename
        
        # Select diverse examples
        examples = {
            "opinionated_positive": [],
            "opinionated_negative": [],
            "opinionated_neutral": [],
            "neutral": []
        }
        
        # Find examples (simplified - you'd want better selection)
        for record in sample[:100]:
            text = record.get("text_clean", "")[:200]
            opinion = record.get("opinion_score", 0)
            
            if opinion > 0.5 and "love" in text.lower() or "great" in text.lower():
                if len(examples["opinionated_positive"]) < 3:
                    examples["opinionated_positive"].append(text)
            elif opinion > 0.5 and any(w in text.lower() for w in ["hate", "terrible", "awful"]):
                if len(examples["opinionated_negative"]) < 3:
                    examples["opinionated_negative"].append(text)
            elif opinion < 0.1:
                if len(examples["neutral"]) < 3:
                    examples["neutral"].append(text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("QUICK REFERENCE: ANNOTATION EXAMPLES\n")
            f.write("=" * 70 + "\n\n")
            
            for category, texts in examples.items():
                f.write(f"\n{category.upper().replace('_', ' ')}:\n")
                f.write("-" * 70 + "\n")
                for i, text in enumerate(texts, 1):
                    f.write(f"{i}. {text}...\n\n")
        
        print(f"   Generated quick reference: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation dataset for manual annotation"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file (filtered_corpus.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Number of records to sample"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize generator
    generator = EvalDatasetGenerator(
        input_file=args.input,
        output_dir=args.output_dir
    )
    
    # Load corpus
    records = generator.load_corpus()
    
    # Create stratified sample
    sample = generator.stratified_sample(records, n=args.size)
    
    # Generate evaluation template
    generator.generate_excel_template(sample)
    
    # Generate annotation guidelines
    generator.generate_annotation_guidelines()
    
    # Generate quick reference
    generator.generate_quick_reference(sample)
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION DATASET GENERATED")
    print("=" * 70)
    print(f"Sample size: {len(sample)} records")
    print(f"\nNext steps:")
    print(f"1. Open {args.output_dir}/eval_dataset.csv in Excel/Google Sheets")
    print(f"2. Read annotation_guidelines.txt")
    print(f"3. Start labeling with 2+ annotators")
    print(f"4. Calculate inter-annotator agreement (target: ≥80%)")
    print(f"5. Save as eval.xls when complete")
    print("=" * 70)


if __name__ == "__main__":
    main()