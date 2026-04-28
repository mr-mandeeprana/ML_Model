import requests

ES_URL = "http://localhost:9200"

print("Checking Elasticsearch...")

try:
    r = requests.get(ES_URL)
    print("Elasticsearch status:", r.status_code)
except:
    print("Elasticsearch not reachable")