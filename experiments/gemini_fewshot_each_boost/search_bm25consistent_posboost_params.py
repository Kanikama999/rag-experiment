"""
bm25_fielded_posboost_consistent（位置ボーナスをBM25の実効TFとして組み込む版、外付けの
boost正規化なし）のbonus_tfパラメータを、Tier A（Subquery単体、短いクエリ）で探索する。

理論的な狙い: BM25自身のIDF・TF飽和ロジックを通せば、クエリのスコアスケール
（クエリ長による違い）に関係なく、同じbonus_tfが機能するはず。固定span_boost=15が
短いクエリで-47%も悪化させた問題（クエリ長対照実験）を、外付けの正規化ではなく
定式化の変更で解消できるかを検証する第一段階。

探索フェーズは全105トピックの1/3（約35トピック、valid_qids[::3]）だけを使う。

使い方:
    python search_bm25consistent_posboost_params.py
"""

from __future__ import annotations

import json
import os
import time

import pytrec_eval

from retriever import bm25_fielded, bm25_fielded_posboost_consistent, rrf_fuse, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 1000
SPAN_END = 100
RERANK_N = 100

BONUS_TF_GRID = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

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

    def search(self, dq, span_terms, bonus_tf):
        key = (dq, tuple(span_terms), bonus_tf)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        if bonus_tf == 0.0:
            r = bm25_fielded(dq, "", "", k=RETRIEVE_K, title_boost=1, headings_boost=1, body_boost=1)
            # bm25_fielded(title,headings,body,...) を body専用として使う（title/headings空）
            r = bm25_fielded("", "", dq, k=RETRIEVE_K, title_boost=1, headings_boost=1, body_boost=1)
        else:
            r = bm25_fielded_posboost_consistent(
                "", "", dq, list(span_terms), k=RETRIEVE_K,
                title_boost=1, headings_boost=1, body_boost=1,
                span_end=SPAN_END, bonus_tf=bonus_tf, rerank_n=RERANK_N)
        self._cache[key] = r
        return r


def evaluate_bonus(bonus_tf, qids, webstyle, qrels, cache):
    run = {}
    for qid in qids:
        dqs = webstyle[qid]["decomposed_queries"]
        lists = []
        for dq in dqs:
            span_terms = cached_span_terms(dq)
            lists.append(cache.search(dq, span_terms, bonus_tf))
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
    print("\n=== bonus_tfを走査（Tier A: Subquery単体, BM25整合版） ===")
    results = []
    for bonus_tf in BONUS_TF_GRID:
        t0 = time.time()
        r, n = evaluate_bonus(bonus_tf, search_qids, webstyle, qrels, cache)
        dt = time.time() - t0
        tag = "(posboostなし)" if bonus_tf == 0.0 else ""
        print(f"  bonus_tf={bonus_tf:5.1f} {tag}  recall@1000={r:.4f}  nDCG@10={n:.4f}  ({dt:.0f}s)")
        results.append({"bonus_tf": bonus_tf, "recall_1000": r, "ndcg_10": n})

    print(f"\nキャッシュ: 呼び出し{cache.calls}回 / ヒット{cache.hits}回")
    out_path = os.path.join(RAG_DIR, "bm25consistent_posboost_param_search_result.json")
    with open(out_path, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "results": results}, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
