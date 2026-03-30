import os
import json

def check_schema(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        required_keys = {
            "ID": str,
            "Source": str,
            "Type": str,
            "Author": str,
            "Text": str,
            "Score": int,
            "Date": str,
            "Word_Count": int,
            "Comments": list
        }
        
        for key, expected_type in required_keys.items():
            if key not in data:
                return False, f"Missing key: {key}"
            if data[key] is not None and not isinstance(data[key], expected_type):
                return False, f"Invalid type for {key}: Expected {expected_type} or null, got {type(data[key])}"
                
        # Title can be string or null
        if "Title" not in data:
             return False, "Missing key: Title"
        if data["Title"] is not None and not isinstance(data["Title"], str):
             return False, f"Invalid type for Title: Expected str or null, got {type(data['Title'])}"
             
        return True, "Valid"
    except json.JSONDecodeError:
        return False, "Invalid JSON"
    except Exception as e:
        return False, str(e)

def main():
    base_path = "/Users/bryanatistakiely/Documents/Modules/Y4S2/SC4021/SC4021-project/articles_extraction"
    target_dir = os.path.join(base_path, "articles_updated_schema")
    
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found.")
        return
        
    files = [f for f in os.listdir(target_dir) if f.endswith('.json')]
    total_files = len(files)
    invalid_files = []
    
    for filename in files:
        file_path = os.path.join(target_dir, filename)
        is_valid, msg = check_schema(file_path)
        if not is_valid:
            invalid_files.append((filename, msg))
            
    if not invalid_files:
        print(f"Success! All {total_files} JSON files follow the correct schema.")
    else:
        print(f"Found {len(invalid_files)} invalid files out of {total_files}:")
        for filename, error_msg in invalid_files[:20]: # show first 20 errors
            print(f"- {filename}: {error_msg}")
        if len(invalid_files) > 20:
             print(f"... and {len(invalid_files) - 20} more.")

if __name__ == "__main__":
    main()
