#!/usr/bin/env python3
"""
Test Script for HackerNews Opinion Crawler
Runs minimal tests to verify everything works
"""

import sys
import json
from pathlib import Path

def test_imports():
    """Test that required packages are available"""
    print("\n Testing imports...")
    
    tests = [
        ("requests", "Core HTTP library"),
        ("json", "JSON parsing (built-in)"),
        ("hashlib", "Hashing (built-in)"),
        ("csv", "CSV handling (built-in)"),
    ]
    
    failed = []
    for module, description in tests:
        try:
            __import__(module)
            print(f"  {module:20} {description}")
        except ImportError:
            print(f"  {module:20} {description} - MISSING")
            failed.append(module)
    
    # Optional packages
    optional = [
        ("elasticsearch", "For indexing (optional)"),
        ("pandas", "For data analysis (optional)")
    ]
    
    print("\n  Optional packages:")
    for module, description in optional:
        try:
            __import__(module)
            print(f" {module:20} {description}")
        except ImportError:
            print(f" {module:20} {description} - not installed")
    
    if failed:
        print(f"\nMissing required packages: {', '.join(failed)}")
        print("   Install with: pip install " + " ".join(failed))
        return False
    
    print("\nAll required imports available")
    return True


def test_algolia_api():
    """Test connection to Algolia HN API"""
    print("\nTesting Algolia HN API connection...")
    
    try:
        import requests # type: ignore
        
        url = "https://hn.algolia.com/api/v1/search"
        params = {"query": "test", "tags": "comment", "hitsPerPage": 1}
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        if "hits" in data:
            print(f"  ✓ API accessible")
            print(f"  ✓ Response format valid")
            print(f"  ✓ Sample hit count: {len(data['hits'])}")
            return True
        else:
            print(f"  ✗ Unexpected response format")
            return False
            
    except Exception as e:
        print(f"  ✗ API test failed: {e}")
        return False


def test_scripts_exist():
    """Test that all required scripts exist"""
    print("\nTesting script files...")
    
    scripts = [
        "hn_scraper.py",
        "hn_preprocess.py",
        "hn_relevance_filter.py",
        "hn_indexer.py",
        "hn_eval_generator.py",
        "hn_pipeline.py"
    ]
    
    failed = []
    for script in scripts:
        path = Path(script)
        if path.exists():
            print(f"  ✓ {script}")
        else:
            print(f"  ✗ {script} - MISSING")
            failed.append(script)
    
    if failed:
        print(f"\nMissing scripts: {', '.join(failed)}")
        return False
    
    print("\nAll scripts present")
    return True


def test_minimal_crawl():
    """Test a minimal crawl with one keyword"""
    print("\n Testing minimal crawl (this may take 30-60 seconds)...")
    
    try:
        from hn_scraper import HNScraper
        
        # Create temporary output directory
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Initialize scraper
        scraper = HNScraper(output_dir=str(output_dir))
        
        # Crawl with just one keyword, minimal results
        print("  → Fetching 10 comments for 'copilot'...")
        records = scraper.crawl_by_keywords(
            comments_per_keyword=10,
            stories_per_keyword=2,
            keywords=["copilot"]
        )
        
        if records and len(records) > 0:
            print(f"  ✓ Crawled {len(records)} records")
            
            # Save
            output_file = scraper.save_raw_corpus(records, "test_corpus.jsonl")
            
            # Verify file
            if output_file.exists():
                print(f"  ✓ Saved to {output_file}")
                
                # Verify content
                with open(output_file, 'r') as f:
                    first_line = f.readline()
                    if first_line:
                        record = json.loads(first_line)
                        required_fields = ["source", "type", "item_id", "text_clean"]
                        missing = [f for f in required_fields if f not in record]
                        
                        if missing:
                            print(f" Missing fields: {missing}")
                        else:
                            print(f"  ✓ Record schema valid")
                
                print("\n Minimal crawl successful!")
                
                # Cleanup
                import shutil
                shutil.rmtree(output_dir)
                print("  ✓ Cleaned up test files")
                
                return True
            else:
                print(f"  ✗ File not created")
                return False
        else:
            print(f"  ✗ No records crawled")
            return False
            
    except Exception as e:
        print(f"  ✗ Crawl test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("HackerNews Opinion Crawler - System Test")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("API Connection", test_algolia_api),
        ("Script Files", test_scripts_exist),
        ("Minimal Crawl", test_minimal_crawl)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\nTest '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20} {status}")
    
    print("=" * 70)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n All tests passed! System is ready to use.")
        print("\nRun the crawler with:")
        print("  python3 hn_pipeline.py --output-dir ./dataset")
        print("\nOr use the quick start:")
        print("  ./quickstart.sh")
        return 0
    else:
        print("\nSome tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
