"""
クエリ長対照実験で判明した「固定span_boostは短いクエリでベーススコアを凌駕してランキング
を破壊する」問題を、boost_ratio（ベーススコアに対する相対比率）による正規化で解消できるか
検証する。Tier A（Subquery単体、短いクエリ）で、posboostなし（boost_ratio=0扱い）を
基準に、boost_ratioを走査してrecall@1000がposboostなし水準まで回復するか、
さらに上回るかを見る。

探索フェーズは全105トピックの1/3（約35トピック、valid_qids[::3]）だけを使う。

使い方:
    python search_normalized_posboost_ratio.py
"""

from __future__ import annotations

import json
import os
import time

import pytrec_eval

from retriever import bm25_body, bm25_body_posboost_normalized, rrf_fuse, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 1000

BOOST_RATIO_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

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


_analyze_cache = {}


def cached_span_terms(dq):
    if dq not in _analyze_cache:
        _analyze_cache[dq] = analyze_terms(dq, field="body")
    return _analyze_cache[dq]


class ScoreCache:
    def __init__(self):
        self._cache = {}
        self.calls = self.hits = 0

    def search(self, dq, span_terms, boost_ratio):
        key = (dq, tuple(span_terms), boost_ratio)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        if boost_ratio == 0.0:
            r = bm25_body(dq, k=RETRIEVE_K)
        else:
            r = bm25_body_posboost_normalized(dq, list(span_terms), k=RETRIEVE_K, boost_ratio=boost_ratio)
        self._cache[key] = r
        return r


def evaluate_ratio(boost_ratio, qids, webstyle, qrels, cache):
    run = {}
    for qid in qids:
        dqs = webstyle[qid]["decomposed_queries"]
        lists = []
        for dq in dqs:
            span_terms = cached_span_terms(dq)
            lists.append(cache.search(dq, span_terms, boost_ratio))
        lists = [lst for lst in lists if lst]
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        run[qid] = {docid: float(score) for docid, score in fused}

    target = [q for q in qids if q in qrels]
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    recall_1000 = sum(r["recall_1000"] for r in results.values()) / n
    ndcg_10 = sum(r["ndcg_cut_10"] for r in results.values()) / n
    return recall_1000, ndcg_10


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
    search_qids = valid_qids[::3]
    print(f"全トピック: {len(valid_qids)}   探索用サブセット: {len(search_qids)}")

    cache = ScoreCache()
    print("\n=== boost_ratioを走査（Tier A: Subquery単体） ===")
    results = []
    for ratio in BOOST_RATIO_GRID:
        t0 = time.time()
        r, n = evaluate_ratio(ratio, search_qids, webstyle, qrels, cache)
        dt = time.time() - t0
        tag = "(posboostなし)" if ratio == 0.0 else ""
        print(f"  boost_ratio={ratio:.3f} {tag}  recall@1000={r:.4f}  nDCG@10={n:.4f}  ({dt:.0f}s)")
        results.append({"boost_ratio": ratio, "recall_1000": r, "ndcg_10": n})

    print(f"\nキャッシュ: 呼び出し{cache.calls}回 / ヒット{cache.hits}回")
    out_path = os.path.join(RAG_DIR, "normalized_posboost_ratio_search_result.json")
    with open(out_path, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "results": results}, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
