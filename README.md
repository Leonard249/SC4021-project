Steps to use backend

1. Create and activate venv
   python -m venv venv
   source .venv/Bin/Activate

2. start the elasticsearch server
   docker run --name elastic_search -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.12.0

3. test elasticsearch server
   curl http://localhost:9200/

# my own steps

cd SC4021-project
source .venv/Scripts/Activate
cd backend
