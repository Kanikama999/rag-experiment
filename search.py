import requests

URL = "http://localhost:9200/msmarco-v21-doc/_search"

query = {
    "size": 3,
    "query": {
        "match": {
            "body": "how does caffeine affect sleep"
        }            
    },
    "_source":["docid", "url", "title"]
}

response = requests.post(URL, json=query)
result = response.json()

print("total(approximate):", result["hits"]["total"]["value"])
print("-" * 20)

for hit in result["hits"]["hits"]:
    score = hit["_score"]
    src = hit["_source"]
    print(f"score: {score:.2f}")
    print(f"docid: {src['docid']}")
    print(f"url: {src['url']}")
    print(f"title: {src['title']}")
    print("-" * 20)