import json
import random
from collections import defaultdict
from pathlib import Path

# Setup paths using the directory structure from your logs
PROJECT_ROOT = Path(r"C:\Users\ryanc\Downloads\NTU Stuff\SC4021\SC4021-project")
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "db_labelled.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "db_labelled_mini.json"

def create_stratified_mini_dataset(input_path: Path, output_path: Path, fraction: float = 1/3):
    if not input_path.exists():
        print(f"Error: Could not find {input_path}")
        return

    # Load the original data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Group records by their top-level label
    label_groups = defaultdict(list)
    for record in data:
        # Fallback to 'unknown' if missing, strip and lower to ensure clean matching
        label = record.get("label", "unknown").strip().lower()
        label_groups[label].append(record)
        
    mini_dataset = []
    
    print("--- Dataset Stratification Summary ---")
    print(f"Original total records: {len(data)}")
    
    # Sample proportionally from each group
    for label, records in label_groups.items():
        # Calculate 1/3, ensuring we always grab at least 1 record if the class exists
        sample_size = max(1, int(len(records) * fraction)) 
        
        # Randomly sample the records
        sampled_records = random.sample(records, sample_size)
        mini_dataset.extend(sampled_records)
        
        print(f"  - {label.capitalize():<10} : {len(records):>4} -> Sampled: {len(sampled_records):>3}")
        
    # Shuffle the final dataset so it isn't perfectly ordered by class
    random.shuffle(mini_dataset)
    
    # Save the new mini dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mini_dataset, f, indent=2, ensure_ascii=False)
        
    print("-" * 38)
    print(f"Mini dataset created successfully!")
    print(f"New total records: {len(mini_dataset)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # Optional: set a random seed so you get the exact same "random" 1/3 every time you run it
    random.seed(42) 
    
    create_stratified_mini_dataset(INPUT_FILE, OUTPUT_FILE)