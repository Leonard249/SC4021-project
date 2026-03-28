import json
from elasticsearch import Elasticsearch, helpers

# connect to elasticsearch
es = Elasticsearch("http://localhost:9200")

# load json
file_name = r'C:\Users\qteam\Documents\School\SC4021\SC4021-project\dataset\vibe_coding_transformer_processed.json'
print(f"Loading data from {file_name}...")

with open(file_name, 'r', encoding='utf-8') as file:
    crawled_data = json.load(file)

print(f"Loaded {len(crawled_data)} records. Preparing for upload...")

# format the data for the Bulk API
def generate_bulk_actions(data_list):
    for item in data_list:
        yield {
            "_index": "opinions",   # The name of your index
            "_id": item["ID"],      # Use your schema's "ID" as the official document ID
            "_source": item         # The actual JSON object (the document itself)
        }

try:
    # helpers.bulk takes care of batching the requests for you automatically
    success_count, failed_count = helpers.bulk(es, generate_bulk_actions(crawled_data))
    
    print(f"Successfully uploaded: {success_count} records.")
    
    if failed_count:
        print(f"Failed to upload: {len(failed_count)} records.")
        
except Exception as e:
    print(f"An error occurred during upload: {e}")