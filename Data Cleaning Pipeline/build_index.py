"""
Run once at startup to build index.json.
index.json: { source: [id, ...] } — IDs only, shuffled per source.
"""
import json, random, os

BASE       = os.path.dirname(os.path.abspath(__file__))
RAW_PATH   = os.path.join(BASE, "raw_data.json")
INDEX_PATH = os.path.join(BASE, "index.json")

with open(RAW_PATH) as f:
    data = json.load(f)

index = {}
for item in data:
    src = item["Source"]
    index.setdefault(src, []).append(item["ID"])

for src in index:
    random.shuffle(index[src])

with open(INDEX_PATH, "w") as f:
    json.dump(index, f)

print("Built index:")
for src, ids in index.items():
    print(f"  {src}: {len(ids)}")
print(f"Total: {sum(len(v) for v in index.values())}")
