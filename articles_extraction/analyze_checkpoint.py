import json
import os

def analyze_checkpoint(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    classified = data.get("classified", {})
    
    total_checked = len(classified)
    relevant_count = 0
    relevant_with_opinion_count = 0

    for entry in classified.values():
        is_relevant = entry.get("is_relevant") == "yes"
        has_opinion = entry.get("has_opinion") == "yes"

        if is_relevant:
            relevant_count += 1
            if has_opinion:
                relevant_with_opinion_count += 1

    print(f"1. Total number of websites checked: {total_checked}")
    print(f"2. Number of relevant websites: {relevant_count}")
    print(f"3. Number of relevant websites AND has opinion: {relevant_with_opinion_count}")

if __name__ == "__main__":
    checkpoint_path = "/Users/bryanatistakiely/Documents/Modules/Y4S2/SC4021/SC4021-project/articles_extraction/classification_checkpoint.json"
    analyze_checkpoint(checkpoint_path)
