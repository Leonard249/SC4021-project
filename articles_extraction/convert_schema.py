import os
import json

def main():
    base_path = "/Users/bryanatistakiely/Documents/Modules/Y4S2/SC4021/SC4021-project/articles_extraction"
    input_dir = os.path.join(base_path, "scraped_articles")
    output_dir = os.path.join(base_path, "articles_updated_schema")

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                old_data = json.load(f)
            
            file_id = filename.replace('.json', '')
            
            new_data = {
                "ID": f"article_{file_id}",
                "Source": old_data.get("source_type", ""),
                "Type": "Article",
                "Author": old_data.get("author", ""),
                "Title": old_data.get("title"),
                "Text": old_data.get("text", ""),
                "Score": 0,
                "Date": old_data.get("date", ""),
                "Word_Count": old_data.get("word_count", 0),
                "Comments": []
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Schema conversion completed successfully.")

if __name__ == "__main__":
    main()
