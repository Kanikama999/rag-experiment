"""
title/headings/bodyのフィールドブースト比を、実際のパイプライン（bm25_fielded + RRF融合）で
直接探索する。learn_field_boosts.py（生スコアを直接回帰）はクエリ間でBM25スコアのスケールが
バラバラなことを無視しており、しかも回帰の目的関数（relevanceの二乗誤差）と実際に見たい
評価指標（nDCG@10, recall@1000）がズレていた。こちらは評価指標そのものを直接最大化する。

recall@1000用とnDCG@10用で、それぞれ別々に最良のブースト比を探索する（両方をまとめた
複合指標は使わない。1指標につき1つの最良比率を出す）。
探索方法: title_boost×headings_boostの総当たりグリッドサーチ（bodyは常に1で固定、
相対比率なので固定して良い）。座標降下法（1軸ずつ探索）だとtitleとheadingsの間の
相互作用を見落とす可能性があるため、全16通り（title,headings ∈ {1,2,3,4}）を総当たりする。

探索フェーズは実行時間短縮のため全105トピックの1/3（約35トピック）だけを使う。
最後に見つかった最良の比率を全105トピックで確認評価する（別途 evaluate_decomposed_webstyle.py
にbm25_fielded_bestboostのような関数を追加して走らせる）。

使い方:
    python search_field_boosts.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
from retriever import bm25_fielded, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PARENT_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(PARENT_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 3000
QUERY_REPEAT = 5

TITLE_GRID = [1, 2, 3, 4]
HEADINGS_GRID = [1, 2, 3, 4]
BODY_BOOST = 1.0  # 相対比率なので固定

METRIC_SPECS = ["recall.1000", "ndcg_cut.10"]
METRICS = set(METRIC_SPECS)


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                queries[obj["id"]] = obj["title"]
    return queries


def dq_pairs_structured(entry):
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"],
                                          entry["query2doc_docs_structured"])
            if doc and doc.get("body")]


class ScoreCache:
    """(title_boost, headings_boost, body_boost, title_q, headings_q, body_q) -> 検索結果 のキャッシュ。
    同じテキストの組でも異なるブースト値では再検索が必要なので、ブースト込みでキーにする。"""

    def __init__(self):
        self._cache = {}
        self.calls = self.hits = 0

    def search(self, title_q, headings_q, body_q, title_boost, headings_boost, body_boost):
        key = (title_q, headings_q, body_q, title_boost, headings_boost, body_boost)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        r = bm25_fielded(title_q, headings_q, body_q, k=RETRIEVE_K,
                          title_boost=title_boost, headings_boost=headings_boost, body_boost=body_boost)
        self._cache[key] = r
        return r


def evaluate_boost(title_boost, headings_boost, body_boost, qids, queries, webstyle, qrels, cache):
    run = {}
    for qid in qids:
        q = queries[qid]
        entry = webstyle[qid]
        pairs = dq_pairs_structured(entry)
        repeated_q = " ".join([q] * QUERY_REPEAT)
        lists = []
        for dq, doc in pairs:
            title_q = f"{repeated_q} {dq} {doc['title']}"
            headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
            body_q = f"{repeated_q} {dq} {doc['body']}"
            lists.append(cache.search(title_q, headings_q, body_q, title_boost, headings_boost, body_boost))
        lists = [lst for lst in lists if lst]
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        run[qid] = {docid: float(score) for docid, score in fused}

    target = [q for q in qids if q in qrels]
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    recall_1000 = sum(r["recall_1000"] for r in results.values()) / n
    ndcg_10 = sum(r["ndcg_cut_10"] for r in results.values()) / n
    return recall_1000, ndcg_10, recall_1000 + ndcg_10


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and qid in qrels and webstyle[qid].get("decomposed_queries")
    )
    search_qids = valid_qids[::3]  # 探索フェーズ用の部分集合（約1/3）
    print(f"全トピック: {len(valid_qids)}   探索用サブセット: {len(search_qids)}")

    cache = ScoreCache()
    log = []

    def try_boost(title_boost, headings_boost, body_boost):
        t0 = time.time()
        recall_1000, ndcg_10, composite = evaluate_boost(
            title_boost, headings_boost, body_boost, search_qids, queries, webstyle, qrels, cache)
        dt = time.time() - t0
        print(f"  title={title_boost} headings={headings_boost} body={body_boost}  "
              f"recall@1000={recall_1000:.4f} nDCG@10={ndcg_10:.4f}  ({dt:.0f}s)")
        log.append({"title_boost": title_boost, "headings_boost": headings_boost, "body_boost": body_boost,
                     "recall_1000": recall_1000, "ndcg_10": ndcg_10})
        return recall_1000, ndcg_10

    print(f"\n=== グリッドサーチ: title×headings 全{len(TITLE_GRID) * len(HEADINGS_GRID)}通り（body=1固定） ===")
    for tb in TITLE_GRID:
        for hb in HEADINGS_GRID:
            try_boost(tb, hb, BODY_BOOST)

    results_by_metric = {}
    for metric_name, key in [("recall_1000", "recall_1000"), ("ndcg_10", "ndcg_10")]:
        best = max(log, key=lambda r: r[key])
        results_by_metric[metric_name] = {
            "title_boost": best["title_boost"], "headings_boost": best["headings_boost"],
            "body_boost": best["body_boost"], "score": best[key],
        }

    print(f"\n{'=' * 72}")
    print("探索結果まとめ:")
    for metric_name, r in results_by_metric.items():
        print(f"  {metric_name}: title={r['title_boost']}, headings={r['headings_boost']}, "
              f"body={r['body_boost']}  ({metric_name}={r['score']:.4f})")
    print(f"キャッシュ: 呼び出し {cache.calls} 回 / ヒット {cache.hits} 回")

    out_path = os.path.join(RAG_DIR, "field_boost_search_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "search_qids": search_qids,
            "grid_title": TITLE_GRID,
            "grid_headings": HEADINGS_GRID,
            "best_by_metric": results_by_metric,
            "log": log,
        }, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
