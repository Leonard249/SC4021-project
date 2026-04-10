import json
import os
from collections import Counter

def get_unique_values(file_path):
    unique_types = Counter()
    unique_sources = Counter()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                iterator = data
            elif isinstance(data, dict):
                iterator = [data]
            else:
                print("Data format not recognized. Expected list or dict.")
                return
        except json.JSONDecodeError:
            # Try parsing as JSONL (JSON lines) if it's not a valid single JSON array/object
            f.seek(0)
            iterator = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        iterator.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                        
        for item in iterator:
            if isinstance(item, dict):
                # Using dict.get() handles cases where the key might be missing and sets it to None, 
                # but let's only add if the key actually exists to be safe
                if 'Type' in item:
                    val = item['Type']
                    # Handle mutable types that can't be used as dictionary keys
                    if isinstance(val, (list, dict)):
                        val = str(val)
                    unique_types[val] += 1
                    
                if 'Source' in item:
                    val = item['Source']
                    if isinstance(val, (list, dict)):
                        val = str(val)
                    unique_sources[val] += 1
                
    print("Content counts by Type:")
    for t, count in sorted(unique_types.items(), key=lambda x: str(x[0])):
        print(f" - {t}: {count}")
        
    print("\nContent counts by Source:")
    for s, count in sorted(unique_sources.items(), key=lambda x: str(x[0])):
        print(f" - {s}: {count}")

if __name__ == "__main__":
    # Ensure it works smoothly regardless of where script is run, assuming they are in same dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'raw_data.json')
    get_unique_values(data_path)
