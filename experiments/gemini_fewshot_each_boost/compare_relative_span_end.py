"""
posboostの判定窓（文書先頭何語以内か）を、固定span_end=100と、文書長dlに応じた
相対窓 clamp(span_frac*dl, span_min, span_max) とで頭出し比較する。

事前のdl分布調査（ランダムサンプルn=300）: min=48, median=610, mean=1072.3,
p90=1993, p99=10102, max=19739。dl<100は2.3%とまれだが、dl>1000（先頭100語が
本文の10%未満）が32%あり、長い文書が不当に不利になっている可能性がある。

bm25_fielded_posboost_relative（事後リランク方式、boost_ratioは固定値の
一次スクリーニング用）で、use_relative=False（固定100）とTrue（相対窓、
span_frac=0.1, span_min=30, span_max=300）を同一boost_ratioで比較する。
recall@1000はアーキテクチャ上不変なので、nDCG@10のみで判定する。

探索フェーズは全105トピックの1/3（約35トピック、valid_qids[::3]）だけを使う。

使い方:
    python compare_relative_span_end.py
"""

from __future__ import annotations

import json
import os
import time

import pytrec_eval

from opensearchpy.exceptions import ConnectionError as OSConnectionError, ConnectionTimeout

from retriever import bm25_fielded, bm25_fielded_posboost_relative, rrf_fuse, analyze_terms


def _with_retry(fn, *args, retries=4, backoff=3, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except (OSConnectionError, ConnectionTimeout) as e:
            if attempt == retries - 1:
                raise
            wait = backoff * (attempt + 1)
            print(f"    [接続エラー、{wait}秒後に再試行 {attempt+1}/{retries}] {type(e).__name__}", flush=True)
            time.sleep(wait)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 1000
BOOST_RATIO = 0.05  # Tier AのnDCG@10のピーク付近（search_normalized_posboost_ratio.pyより）
RERANK_N = 100

CONDITIONS = [
    ("posboostなし", None),
    ("固定 span_end=100", dict(use_relative=False, fixed_span_end=100)),
    # 第1回の結果、span_min=30(<100)が中央値dl=610の文書で窓を61語まで狭めてしまい、
    # 大半の文書で固定100語より厳しい条件になっていたと判明。span_min=100を「常に
    # 固定版以上」の下限フロアとして再設計（長い文書だけをより緩くする）。
    ("相対窓 10%dl [100,300]", dict(use_relative=True, span_frac=0.10, span_min=100, span_max=300)),
    ("相対窓 15%dl [100,400]", dict(use_relative=True, span_frac=0.15, span_min=100, span_max=400)),
    ("相対窓 20%dl [100,600]", dict(use_relative=True, span_frac=0.20, span_min=100, span_max=600)),
]

METRIC_SPECS = ["recall.1000", "ndcg_cut.10", "P.10"]
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

    def search(self, dq, span_terms, label, params):
        key = (dq, tuple(span_terms), label)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        if params is None:
            r = _with_retry(bm25_fielded, "", "", dq, k=RETRIEVE_K,
                             title_boost=1, headings_boost=1, body_boost=1)
        else:
            r = _with_retry(
                bm25_fielded_posboost_relative,
                "", "", dq, list(span_terms), k=RETRIEVE_K,
                title_boost=1, headings_boost=1, body_boost=1,
                boost_ratio=BOOST_RATIO, rerank_n=RERANK_N, **params)
        self._cache[key] = r
        return r


def evaluate(label, params, qids, webstyle, qrels, cache):
    run = {}
    for qid in qids:
        dqs = webstyle[qid]["decomposed_queries"]
        lists = []
        for dq in dqs:
            span_terms = cached_span_terms(dq)
            lists.append(cache.search(dq, span_terms, label, params))
        lists = [lst for lst in lists if lst]
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        run[qid] = {docid: float(score) for docid, score in fused}

    target = [q for q in qids if q in qrels]
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    return {
        "recall_1000": sum(r["recall_1000"] for r in results.values()) / n,
        "ndcg_10": sum(r["ndcg_cut_10"] for r in results.values()) / n,
        "p_10": sum(r["P_10"] for r in results.values()) / n,
    }


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
    print(f"\n=== 固定 vs 相対 span_end 比較（Tier A, boost_ratio={BOOST_RATIO}固定） ===")
    results = []
    for label, params in CONDITIONS:
        t0 = time.time()
        m = evaluate(label, params, search_qids, webstyle, qrels, cache)
        dt = time.time() - t0
        print(f"  {label:24s}  recall@1000={m['recall_1000']:.4f}  nDCG@10={m['ndcg_10']:.4f}  P@10={m['p_10']:.4f}  ({dt:.0f}s)")
        results.append({"label": label, **m})

    print(f"\nキャッシュ: 呼び出し{cache.calls}回 / ヒット{cache.hits}回")
    out_path = os.path.join(RAG_DIR, "relative_span_end_compare_result_v2.json")
    with open(out_path, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "boost_ratio": BOOST_RATIO, "results": results}, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
