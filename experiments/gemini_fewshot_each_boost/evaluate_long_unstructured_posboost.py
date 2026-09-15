"""
「長いが構造化されていない」クエリでposboostがどう効くかを検証する、querylength_confound
のTier A/B/Cに対する第4の条件（Tier D）。

Tier C（narrative×5+Subquery+LLM生成疑似文書のfielded検索）はposboostが有意にプラスだったが、
これは「クエリが長い」ことと「疑似文書で構造化されている」ことが同時に変わっているため、
どちらが効いているのか切り分けられなかった。

本スクリプトは narrative×5 + Subquery だけ（疑似文書のtitle/headings/bodyは一切使わない）を
単一のbodyフィールドへ投げ、posboost（span_first, Subqueryの語, span_end=100, span_boost=15.0）
の有無を比較する。「長いが構造化なし」条件で、Tier Cと同じ傾向（posboostが有意にプラス）が
出るなら「長さ」が効いている、出ないなら「構造化」も必要、という切り分けができる。

retriever.pyのbm25_bodyonly_posboost_discourseboost（title/headingsを使わずbodyのみ+posboost）を
markers=[]（discourseboost無効）で使う。

35トピックのサブセットで実施（querylength_confound.py本体は105トピックだが、本スクリプトは
追加の確認用として速度を優先）。

使い方:
    python evaluate_long_unstructured_posboost.py
"""

from __future__ import annotations

import json
import os
import time

import pytrec_eval

from retriever import bm25_body, bm25_bodyonly_posboost_discourseboost, rrf_fuse, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
SPAN_END, SPAN_BOOST = 100, 15.0

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


def build_run(method, qids, queries, webstyle):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        entry = webstyle[qid]
        pairs = dq_pairs_structured(entry)
        repeated_q = " ".join([q] * QUERY_REPEAT)
        lists = []
        for dq, doc in pairs:
            body_text = f"{repeated_q} {dq}"  # 疑似文書は一切使わない
            if method == "posboost_off":
                lists.append(bm25_body(body_text, k=RETRIEVE_K))
            else:
                span_terms = cached_span_terms(dq)
                lists.append(bm25_bodyonly_posboost_discourseboost(
                    body_text, span_terms, k=RETRIEVE_K,
                    span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[]))
        lists = [lst for lst in lists if lst]
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        run[qid] = {docid: float(score) for docid, score in fused}
        if i % 10 == 0 or i == len(qids):
            print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return run


def evaluate(run, qrels, eval_qids, label):
    target = [q for q in eval_qids if q in qrels]
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
        and any(d and d.strip() for d in webstyle[qid].get("query2doc_docs", []))
    )
    search_qids = valid_qids[::3]
    print(f"全トピック: {len(valid_qids)}   探索用サブセット: {len(search_qids)}")
    print(f"body_text = narrative×{QUERY_REPEAT} + Subquery（疑似文書は含まない）")

    methods = ["posboost_off", "posboost_on"]
    runs = {m: build_run(m, search_qids, queries, webstyle) for m in methods}

    summary = {}
    for name in QREL_SETS:
        print(f"\n[{name}]")
        for method in methods:
            summary.setdefault(name, {})[method] = evaluate(runs[method], qrels[name], search_qids, f"{method}/{name}")

    print("\n" + "=" * 72)
    for name in QREL_SETS:
        off, on = summary[name]["posboost_off"], summary[name]["posboost_on"]
        print(f"[{name}] posboost off->on: " +
              "  ".join(f"{m}: {off[m]:.4f}->{on[m]:.4f} ({on[m]-off[m]:+.4f})" for m in METRIC_KEYS))

    out_path = os.path.join(RAG_DIR, "long_unstructured_posboost_result.json")
    with open(out_path, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "summary": summary}, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
