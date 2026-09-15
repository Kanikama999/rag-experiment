"""
equalweight posboost_onlyのspan_end/span_boostについて、現行値(100, 15.0)と探索で見つかった
候補(200, 30.0)の2点だけを、全4指標（recall@100, recall@1000, nDCG@10, precision@100）で
評価する。search_equalweight_posboost_params.pyのグリッド探索ではrecall.1000/ndcg_cut.10の
2指標しか計算していなかったため、そのギャップを埋める。35トピックのサブセットで実施。

使い方:
    python eval_equalweight_posboost_two_points.py
"""

from __future__ import annotations

import json
import os
import time

import pytrec_eval

from retriever import bm25_equalweight_posboost_discourseboost, rrf_fuse, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5

POINTS = [
    ("現行値", 100, 15.0),
    ("候補", 200, 30.0),
]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


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


def evaluate_params(span_end, span_boost, qids, queries, webstyle, qrels):
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
            r = bm25_equalweight_posboost_discourseboost(
                title_q, headings_q, body_q, list(span_terms), k=RETRIEVE_K,
                span_end=span_end, span_boost=span_boost, markers=[])
            lists.append(r)
        lists = [lst for lst in lists if lst]
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        run[qid] = {docid: float(score) for docid, score in fused}

    target = [q for q in qids if q in qrels]
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    return agg, n


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

    results = {}
    for label, span_end, span_boost in POINTS:
        t0 = time.time()
        agg, n = evaluate_params(span_end, span_boost, search_qids, queries, webstyle, qrels)
        dt = time.time() - t0
        print(f"[{label}] span_end={span_end} span_boost={span_boost}  n={n}  "
              + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS)
              + f"  ({dt:.0f}s)")
        results[label] = {"span_end": span_end, "span_boost": span_boost, **agg}

    out_path = os.path.join(RAG_DIR, "equalweight_posboost_two_points_full_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "results": results}, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
