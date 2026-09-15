"""
search_normalized_posboost_ratio.py のTier C版（narrative×5+Subquery+疑似文書の
fielded検索、既存最良パイプライン）。Tier A/Bで正規化boost_ratioが有効だと確認できたので、
同じ発想がTier C（元々固定boostでも+9.8%改善していた設定）でどう働くかを見る。

クエリ構成はevaluate_querylength_confound.pyのbuild_tierCと同一
（title_boost=2, headings_boost=1, body_boost=1、narrative×5+Subquery+疑似文書）。

探索フェーズは全105トピックの1/3（約35トピック、valid_qids[::3]）だけを使う。

使い方:
    python search_normalized_posboost_ratio_tierC.py
"""

from __future__ import annotations

import json
import os
import time

import pytrec_eval

from retriever import bm25_fielded, bm25_fielded_posboost_normalized, rrf_fuse, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
TITLE_BOOST, HEADINGS_BOOST, BODY_BOOST = 2, 1, 1

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


def dq_pairs_structured(entry):
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"],
                                          entry["query2doc_docs_structured"])
            if doc and doc.get("body")]


_analyze_cache = {}


def cached_span_terms(dq):
    if dq not in _analyze_cache:
        _analyze_cache[dq] = analyze_terms(dq, field="body")
    return _analyze_cache[dq]


class ScoreCache:
    def __init__(self):
        self._cache = {}
        self.calls = self.hits = 0

    def search(self, title_q, headings_q, body_q, span_terms, boost_ratio):
        key = (title_q.strip(), headings_q.strip(), body_q.strip(), tuple(span_terms), boost_ratio)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        if boost_ratio == 0.0:
            r = bm25_fielded(title_q, headings_q, body_q, k=RETRIEVE_K,
                              title_boost=TITLE_BOOST, headings_boost=HEADINGS_BOOST, body_boost=BODY_BOOST)
        else:
            r = bm25_fielded_posboost_normalized(
                title_q, headings_q, body_q, list(span_terms), k=RETRIEVE_K,
                title_boost=TITLE_BOOST, headings_boost=HEADINGS_BOOST, body_boost=BODY_BOOST,
                boost_ratio=boost_ratio)
        self._cache[key] = r
        return r


def evaluate_ratio(boost_ratio, qids, queries, webstyle, qrels, cache):
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
            span_terms = cached_span_terms(dq)
            lists.append(cache.search(title_q, headings_q, body_q, span_terms, boost_ratio))
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
        if qid in queries and qid in qrels
        and webstyle[qid].get("decomposed_queries")
        and any(d and d.strip() for d in webstyle[qid].get("query2doc_docs", []))
    )
    search_qids = valid_qids[::3]
    print(f"全トピック: {len(valid_qids)}   探索用サブセット: {len(search_qids)}")

    cache = ScoreCache()
    print("\n=== boost_ratioを走査（Tier C: フルパイプライン） ===")
    results = []
    for ratio in BOOST_RATIO_GRID:
        t0 = time.time()
        r, n = evaluate_ratio(ratio, search_qids, queries, webstyle, qrels, cache)
        dt = time.time() - t0
        tag = "(posboostなし)" if ratio == 0.0 else ""
        print(f"  boost_ratio={ratio:.3f} {tag}  recall@1000={r:.4f}  nDCG@10={n:.4f}  ({dt:.0f}s)")
        results.append({"boost_ratio": ratio, "recall_1000": r, "ndcg_10": n})

    print(f"\nキャッシュ: 呼び出し{cache.calls}回 / ヒット{cache.hits}回")
    out_path = os.path.join(RAG_DIR, "normalized_posboost_ratio_search_result_tierC.json")
    with open(out_path, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "results": results}, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
