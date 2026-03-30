### For macOS 14+ Users (Docker)
```bash
# 1. Start Elasticsearch
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:7.17.4

# 2. Wait 30 seconds, then verify
curl http://localhost:9200

# 3. Index data (first time only)
python prepare_raw_data.py  # ~20 mins
python index_data.py        # ~5 mins

# 4. Start search engine
python search_engine.py

# 5. Open UI
open search_ui_enhanced.html
```

### For macOS 13 Users (Homebrew)
```bash
# 1. Start Elasticsearch
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-11.jdk/Contents/Home
elasticsearch

# 2. In new terminal, index data (first time only)
python prepare_raw_data.py  # ~20 mins
python index_data.py        # ~5 mins

# 3. Start search engine
python search_engine.py

# 4. Open UI
open search_ui_enhanced.html
```

---
