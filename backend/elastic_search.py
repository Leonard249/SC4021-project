import requests

try:
    response = requests.get("http://127.0.0.1:9200")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Even requests failed: {e}")