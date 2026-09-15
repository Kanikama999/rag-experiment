from opensearchpy import OpenSearch
from collections import defaultdict

client = OpenSearch("http://localhost:9200", timeout=60, max_retries=2, retry_on_timeout=True)
INDEX = "msmarco-v21-doc"
QREL_POOL_INDEX = "qrel-segment-pool"

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

def bm25_title(query_text, k=100):
    """titleフィールドだけをmatchする（フィールド別寄与の切り分け用）。"""
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "match": {
                "title": {
                    "query": query_text
                }
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]

def bm25_headings(query_text, k=100):
    """headingsフィールドだけをmatchする（フィールド別寄与の切り分け用）。"""
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "match": {
                "headings": {
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

def bm25_fielded(title_text, headings_text, body_text, k=100,
                  title_boost=3, headings_boost=2, body_boost=1):
    """title/headings/bodyそれぞれ別のテキストを対応するフィールドにmatchし、
    bool/shouldで線形和（デフォルトはtitle^3 + headings^2 + body^1）にして検索する。
    bm25_keytermsが同じquery_textを3フィールドにぶつけてmax(best_fields)を取るのに対し、
    こちらはフィールドごとに別々のテキストを渡し、3フィールド分のスコアを全て合算する。
    ブースト値はtitle_boost/headings_boost/body_boostで変更できる（比率の見直し用）。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "bool": {
                "should": should
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]

def bm25_qrelpool(query_text, k=100):
    """qrelで正解判定されているセグメントだけのインデックス(10,281件)を対象にBM25検索する。"""
    res = client.search(index=QREL_POOL_INDEX, body={
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
        
        
