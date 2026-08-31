"""
TREC 2025 RAG の生qrels（セグメント単位）に載っているsegment idだけを、
MS MARCO v2.1 segmented コーパス（/data/msmarco_v2.1_doc_segmented、60シャード・26GB）
から抜き出してpoolを作る。

narrative_expansion.py の出力（multi_query2doc_L200.json）と同じ構造
（config + results、results[qid].pool / .query2doc_results.query2doc_k / .word_counts）
にして、evaluate_rep*.py の QUERY2DOC_FILE をそのまま差し替えられるようにする。
pool は qrelスコア降順で並べる（query2doc_k = 上位k件の正解セグメント、という意味になる）。

対象は 2025-rag-qrels.txt に載っている22トピックのみ（105トピック全部ではない）。

使い方:
    python qrel_segment_pool.py
"""

import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict

DATA_DIR = os.path.expanduser("~/data")
QRELS_FILE = os.path.join(DATA_DIR, "2025-rag-qrels.txt")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
SEG_DIR = "/data/msmarco_v2.1_doc_segmented"

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(RAG_DIR, "qrel_segment_pool.json")

DOCID_RE = re.compile(rb'^\{"docid": "([^"]+)"')
SHARD_RE = re.compile(r"^msmarco_v2\.1_doc_(\d\d)_")

CONFIG = {
    "source": "2025-rag-qrels.txt (raw segment-level qrels)",
    "filter": "all judged segments (score>=0)",
    "pool_order": "qrel score desc",
}


def load_queries():
    queries = {}
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                queries[str(d["id"])] = d["title"]
    return queries


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
    """1シャードを読み、needed(docid集合)に一致する行だけ {docid: text} で返す"""
    found = {}
    with gzip.open(path, "rb") as f:
        for raw in f:
            m = DOCID_RE.match(raw)
            if not m:
                continue
            docid = m.group(1).decode("utf-8")
            if docid in needed:
                d = json.loads(raw)
                found[docid] = d["segment"]
    return found


def main():
    t0 = time.time()
    queries = load_queries()
    by_shard = load_qrels()

    # qid -> [(docid, score, text)]
    per_qid = defaultdict(list)
    n_missing = 0

    shards = sorted(by_shard.keys())
    for i, shard in enumerate(shards, 1):
        needed = by_shard[shard]
        path = os.path.join(SEG_DIR, f"msmarco_v2.1_doc_segmented_{shard}.json.gz")
        if not os.path.exists(path):
            print(f"  WARN シャードファイルが無い: {path}", file=sys.stderr)
            continue

        t1 = time.time()
        texts = scan_shard(path, set(needed.keys()))
        print(f"[{i}/{len(shards)}] shard {shard}: 対象{len(needed)}件中 {len(texts)}件ヒット "
              f"({time.time() - t1:.0f}s)", flush=True)

        for docid, pairs in needed.items():
            text = texts.get(docid)
            if text is None:
                n_missing += 1
                continue
            for qid, score in pairs:
                per_qid[qid].append((docid, score, text))

    K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30]
    results = {}
    for qid, items in per_qid.items():
        items.sort(key=lambda x: -x[1])  # スコア降順
        docids = [d for d, _, _ in items]
        scores = [s for _, s, _ in items]
        pool = [t for _, _, t in items]
        results[qid] = {
            "original_query": queries.get(qid),
            "pool": pool,
            "docids": docids,
            "scores": scores,
            "query2doc_results": {f"query2doc_{k}": pool[:k] for k in K_VALUES if k <= len(pool)},
            "word_counts": [len(t.split()) for t in pool],
        }

    json.dump({"config": CONFIG, "results": results},
              open(OUT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\nトピック数={len(results)}  見つからなかったdocid={n_missing}")
    for qid in sorted(results, key=lambda q: int(q)):
        print(f"  qid={qid}: {len(results[qid]['pool'])}件")
    print(f"\n所要時間={time.time() - t0:.0f}s")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    main()
