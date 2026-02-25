#!/usr/bin/env python3
"""
HackerNews Opinion Crawler - Master Pipeline
Complete pipeline from crawling to indexing
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

class PipelineRunner:
    """Run the complete HN crawling pipeline"""
    
    def __init__(self, output_dir: str = "./dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.files = {
            "raw": self.output_dir / "raw_corpus.jsonl",
            "cleaned": self.output_dir / "cleaned_corpus.jsonl",
            "filtered": self.output_dir / "filtered_corpus.jsonl",
            "eval": self.output_dir / "eval_dataset.csv"
        }
    
    def run_command(self, cmd: list, description: str):
        """Run a command and handle errors"""
        print("\n" + "=" * 70)
        print(f"🚀 {description}")
        print("=" * 70)
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True, text=True)
            print(f"\n✅ {description} - COMPLETED")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {description} - FAILED")
            print(f"Error: {e}")
            return False
    
    def step_1_crawl(self, comments_per_kw: int, stories_per_kw: int):
        """Step 1: Crawl HackerNews"""
        cmd = [
            "python3", "hn_scraper.py",
            "--output-dir", str(self.output_dir),
            "--comments-per-keyword", str(comments_per_kw),
            "--stories-per-keyword", str(stories_per_kw)
        ]
        
        return self.run_command(cmd, "STEP 1: Crawling HackerNews")
    
    def step_2_preprocess(self, min_length: int):
        """Step 2: Preprocess and clean"""
        if not self.files["raw"].exists():
            print(f"\n❌ Error: Raw corpus not found at {self.files['raw']}")
            return False
        
        cmd = [
            "python3", "hn_preprocess.py",
            "--input", str(self.files["raw"]),
            "--output-dir", str(self.output_dir),
            "--min-length", str(min_length)
        ]
        
        return self.run_command(cmd, "STEP 2: Preprocessing and Cleaning")
    
    def step_3_filter(self, min_relevance: float, min_opinion: float):
        """Step 3: Filter by relevance"""
        if not self.files["cleaned"].exists():
            print(f"\n❌ Error: Cleaned corpus not found at {self.files['cleaned']}")
            return False
        
        cmd = [
            "python3", "hn_relevance_filter.py",
            "--input", str(self.files["cleaned"]),
            "--output-dir", str(self.output_dir),
            "--min-relevance", str(min_relevance),
            "--min-opinion", str(min_opinion),
            "--validation-samples", "100"
        ]
        
        return self.run_command(cmd, "STEP 3: Relevance Filtering")
    
    def step_4_index(self, es_host: str, es_port: int, index_name: str):
        """Step 4: Index into Elasticsearch"""
        if not self.files["filtered"].exists():
            print(f"\n❌ Error: Filtered corpus not found at {self.files['filtered']}")
            return False
        
        # Check if elasticsearch package is available
        try:
            import elasticsearch
        except ImportError:
            print("\n⚠️  Elasticsearch package not installed")
            print("   Install with: pip install elasticsearch")
            print("   Skipping indexing step")
            return True
        
        cmd = [
            "python3", "hn_indexer.py",
            "--input", str(self.files["filtered"]),
            "--es-host", es_host,
            "--es-port", str(es_port),
            "--index-name", index_name
        ]
        
        return self.run_command(cmd, "STEP 4: Elasticsearch Indexing")
    
    def step_5_eval_dataset(self, size: int):
        """Step 5: Generate evaluation dataset"""
        if not self.files["filtered"].exists():
            print(f"\n❌ Error: Filtered corpus not found at {self.files['filtered']}")
            return False
        
        cmd = [
            "python3", "hn_eval_generator.py",
            "--input", str(self.files["filtered"]),
            "--output-dir", str(self.output_dir),
            "--size", str(size)
        ]
        
        return self.run_command(cmd, "STEP 5: Evaluation Dataset Generation")
    
    def run_full_pipeline(self, config: dict):
        """Run the complete pipeline"""
        start_time = datetime.now()
        
        print("\n" + "=" * 70)
        print("🎯 HACKERNEWS OPINION CRAWLER - FULL PIPELINE")
        print("=" * 70)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
        
        steps = [
            ("Crawl", lambda: self.step_1_crawl(
                config["comments_per_keyword"],
                config["stories_per_keyword"]
            )),
            ("Preprocess", lambda: self.step_2_preprocess(
                config["min_text_length"]
            )),
            ("Filter", lambda: self.step_3_filter(
                config["min_relevance"],
                config["min_opinion"]
            )),
            ("Index", lambda: self.step_4_index(
                config["es_host"],
                config["es_port"],
                config["index_name"]
            )),
            ("Eval Dataset", lambda: self.step_5_eval_dataset(
                config["eval_size"]
            ))
        ]
        
        results = {}
        for step_name, step_func in steps:
            success = step_func()
            results[step_name] = success
            
            if not success and step_name != "Index":  # Index is optional
                print(f"\n❌ Pipeline failed at step: {step_name}")
                break
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Final report
        print("\n" + "=" * 70)
        print("📊 PIPELINE SUMMARY")
        print("=" * 70)
        print(f"Duration: {duration}")
        print(f"\nStep Results:")
        for step_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {step_name:20} {status}")
        
        print(f"\nOutput Files:")
        for name, path in self.files.items():
            exists = "✓" if path.exists() else "✗"
            print(f"  {exists} {name:15} {path}")
        
        print("=" * 70)
        
        all_success = all(results.values())
        if all_success or (not results.get("Index", True) and 
                          all(v for k, v in results.items() if k != "Index")):
            print("\n🎉 Pipeline completed successfully!")
            print("\nNext steps:")
            print("1. Review corpus statistics in dataset/corpus_stats.json")
            print("2. Check validation samples in dataset/validation_sample.txt")
            print("3. Start manual annotation with eval_dataset.csv")
            print("4. Run sentiment classification model")
            return True
        else:
            print("\n⚠️  Pipeline completed with errors")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete HackerNews crawling pipeline"
    )
    
    # Pipeline control
    parser.add_argument(
        "--output-dir",
        default="./dataset",
        help="Output directory for all files"
    )
    parser.add_argument(
        "--steps",
        choices=["all", "crawl", "preprocess", "filter", "index", "eval"],
        default="all",
        help="Which steps to run"
    )
    
    # Crawling parameters
    parser.add_argument(
        "--comments-per-keyword",
        type=int,
        default=1000,
        help="Comments to fetch per keyword"
    )
    parser.add_argument(
        "--stories-per-keyword",
        type=int,
        default=100,
        help="Stories to fetch per keyword"
    )
    
    # Processing parameters
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=20,
        help="Minimum text length (characters)"
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
    
    # Elasticsearch parameters
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
    
    # Evaluation parameters
    parser.add_argument(
        "--eval-size",
        type=int,
        default=1000,
        help="Evaluation dataset size"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = {
        "comments_per_keyword": args.comments_per_keyword,
        "stories_per_keyword": args.stories_per_keyword,
        "min_text_length": args.min_text_length,
        "min_relevance": args.min_relevance,
        "min_opinion": args.min_opinion,
        "es_host": args.es_host,
        "es_port": args.es_port,
        "index_name": args.index_name,
        "eval_size": args.eval_size
    }
    
    # Initialize pipeline
    pipeline = PipelineRunner(output_dir=args.output_dir)
    
    # Run pipeline
    if args.steps == "all":
        success = pipeline.run_full_pipeline(config)
        sys.exit(0 if success else 1)
    else:
        # Run individual step
        step_map = {
            "crawl": lambda: pipeline.step_1_crawl(
                config["comments_per_keyword"],
                config["stories_per_keyword"]
            ),
            "preprocess": lambda: pipeline.step_2_preprocess(
                config["min_text_length"]
            ),
            "filter": lambda: pipeline.step_3_filter(
                config["min_relevance"],
                config["min_opinion"]
            ),
            "index": lambda: pipeline.step_4_index(
                config["es_host"],
                config["es_port"],
                config["index_name"]
            ),
            "eval": lambda: pipeline.step_5_eval_dataset(
                config["eval_size"]
            )
        }
        
        success = step_map[args.steps]()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()