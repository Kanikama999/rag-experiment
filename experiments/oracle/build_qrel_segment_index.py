"""
qrel_segment_pool.py で使ったのと同じ抽出（2025-rag-qrels.txtに載っているsegment idを
/data/msmarco_v2.1_doc_segmented から拾う）を、今度はOpenSearchへの索引投入用にやり直す。

qrel_segment_pool.json は body(segment本文)しか保存していないので、title/url/headings込みで
もう一度シャードをスキャンし、そのままインデックスへbulk投入する。

インデックス設定はmsmarco-v21-docに合わせる（english analyzer, BM25 b=0.4/k1=0.9）。
1トピック内で複数回判定されたdocid以外は基本1トピックにしか出てこないが、
念のためqids/scoresは配列で持たせる（同じ位置がペア: qids[i]に対するscores[i]）。

使い方:
    python build_qrel_segment_index.py
"""

import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict

from opensearchpy import OpenSearch, helpers

DATA_DIR = os.path.expanduser("~/data")
QRELS_FILE = os.path.join(DATA_DIR, "2025-rag-qrels.txt")
SEG_DIR = "/data/msmarco_v2.1_doc_segmented"

INDEX = "qrel-segment-pool"

DOCID_RE = re.compile(rb'^\{"docid": "([^"]+)"')
SHARD_RE = re.compile(r"^msmarco_v2\.1_doc_(\d\d)_")

client = OpenSearch("http://localhost:9200")

MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "similarity": {"default": {"type": "BM25", "b": 0.4, "k1": 0.9}},
        "analysis": {"analyzer": {"english_search": {"type": "english"}}},
    },
    "mappings": {
        "properties": {
            "docid": {"type": "keyword"},
            "url": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "english_search"},
            "headings": {"type": "text", "analyzer": "english_search"},
            "body": {"type": "text", "analyzer": "english_search"},
            "qids": {"type": "keyword"},
            "scores": {"type": "integer"},
        }
    },
}


def load_qrels():
    """shard番号 -> {docid: [(qid, score), ...]} を返す"""
    by_shard = defaultdict(dict)
    n = 0
    with open(QRELS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid, _, docid, score = line.split()
            m = SHARD_RE.match(docid)
            if not m:
                print(f"  WARN 想定外のdocid形式: {docid}", file=sys.stderr)
                continue
            shard = m.group(1)
            by_shard[shard].setdefault(docid, []).append((qid, int(score)))
            n += 1
    print(f"qrels読み込み: {n}行 / {sum(len(d) for d in by_shard.values())} 種類のdocid / "
          f"{len(by_shard)}シャードに分散")
    return by_shard


def scan_shard(path, needed):
    """1シャードを読み、needed(docid集合)に一致する行だけ {docid: dict} で返す"""
    found = {}
    with gzip.open(path, "rb") as f:
        for raw in f:
            m = DOCID_RE.match(raw)
            if not m:
                continue
            docid = m.group(1).decode("utf-8")
            if docid in needed:
                found[docid] = json.loads(raw)
    return found


def ensure_index():
    if client.indices.exists(index=INDEX):
        raise SystemExit(
            f"インデックス {INDEX} は既に存在します。作り直す場合は先に手動でDELETEしてください。"
        )
    client.indices.create(index=INDEX, body=MAPPING)
    print(f"インデックス {INDEX} を作成しました。")


def main():
    t0 = time.time()
    by_shard = load_qrels()
    ensure_index()

    n_indexed, n_missing = 0, 0
    shards = sorted(by_shard.keys())
    for i, shard in enumerate(shards, 1):
        needed = by_shard[shard]
        path = os.path.join(SEG_DIR, f"msmarco_v2.1_doc_segmented_{shard}.json.gz")
        if not os.path.exists(path):
            print(f"  WARN シャードファイルが無い: {path}", file=sys.stderr)
            continue

        t1 = time.time()
        docs = scan_shard(path, set(needed.keys()))

        actions = []
        for docid, pairs in needed.items():
            d = docs.get(docid)
            if d is None:
                n_missing += 1
                continue
            actions.append({
                "_index": INDEX,
                "_id": docid,
                "_source": {
                    "docid": docid,
                    "url": d.get("url"),
                    "title": d.get("title"),
                    "headings": d.get("headings"),
                    "body": d.get("segment"),
                    "qids": [q for q, _ in pairs],
                    "scores": [s for _, s in pairs],
                },
            })
        if actions:
            helpers.bulk(client, actions)
            n_indexed += len(actions)

        print(f"[{i}/{len(shards)}] shard {shard}: {len(actions)}件投入 "
              f"({time.time() - t1:.0f}s)", flush=True)

    client.indices.refresh(index=INDEX)
    count = client.count(index=INDEX)["count"]
    print(f"\n投入完了: {n_indexed}件送信 / 見つからなかったdocid={n_missing} / "
          f"インデックス内docs.count={count}")
    print(f"所要時間={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
