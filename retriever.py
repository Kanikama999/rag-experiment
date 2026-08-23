from opensearchpy import OpenSearch
from collections import defaultdict

client = OpenSearch("http://localhost:9200")
INDEX = "msmarco-v21-doc"

def bm25_body(query_text, k=100):
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "match": {
                "body": {
                    "query": query_text
                }
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]

def bm25_keyterms(query_text, k=100):
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["title^3", "headings^2", "body^1"]
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]

def rrf_fuse(rank_lists, k=60, top_n=100):
    fused = defaultdict(float)
    for lst in rank_lists:
        for rank, (docid, _) in enumerate(lst, start=1):
            fused[docid] += 1.0 /(k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:top_n]

def show_title_headings(fused, n=10):
    ids = [d for d, _ in fused[:n]]
    res = client.mget(index=INDEX,
                      body={
                          "ids": ids,
                      },
                      params={
                          "_source_includes": "title,headings,url"
                      },
    )
    meta = {d["_id"]: d.get("_source", {}) for d in res["docs"]}
    for docid, score in fused[:n]:
        m = meta.get(docid, {})
        title = m.get("title") or "(no title)"
        hs = [h.strip() for h in (m.get("headings") or "").split("\n") if h.strip()]
        print(f"{score:.5f}  title:{title[:70]}\n")
        print(f"          {' | '.join(hs)[:150]}\n")
        print(f"          {docid}")
        print("-" *60)
        
        
