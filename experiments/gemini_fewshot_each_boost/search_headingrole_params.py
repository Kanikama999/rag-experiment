"""
見出し役割ブースト（bm25_equalweight_posboost_headingroleboost）のパラメータ探索。

heading_slop（headingsフィールド上で役割語とクエリ語をどこまで近接とみなすか）と
heading_role_boost（そのブーストの重み）を、35トピックのサブセット上で座標降下的に
振る。既存のsearch_*_params.pyと同じ流儀（consensus qrels、rep5、RRF融合）。

並列で別の実験が走っているとOpenSearch側が飽和して1設定あたり数十分かかるため、
--stride でサブセットを粗くできるようにしてある（既定6 = 18トピック。既存の
search_*_params.pyと揃えたい場合は --stride 3 = 35トピック）。設定が1つ終わるたびに
結果JSONを上書き保存するので、途中で止めてもそこまでの結果は残る。

使い方:
    python search_headingrole_params.py [--stride N]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pytrec_eval

from retriever import (bm25_equalweight_posboost_headingroleboost, rrf_fuse, analyze_terms,
                       heading_roles, role_markers_for, HEADING_ROLE_MARKERS)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")
OUT_FILE = os.path.join(RAG_DIR, "headingrole_param_search_result.json")

TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
SPAN_END = 100          # チャンピオンの現行値
SPAN_BOOST = 15.0       # 同上

SLOP_GRID = [4, 8, 15]
BOOST_GRID = [2.0, 10.0]

METRIC_SPECS = ["recall.1000", "ndcg_cut.10"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]

# 全役割ぶんをそのまま使うと markers x span_terms が OpenSearch の
# maxClauseCount(1024) を超えるため、役割あたり2語までに絞る。
ALL_ROLE_MARKERS = role_markers_for(list(HEADING_ROLE_MARKERS))[:18]


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return qrels


def load_queries(path):
    q = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                q[o["id"]] = o["title"]
    return q


_term_cache, _role_cache = {}, {}


def span_terms_for(dq):
    if dq not in _term_cache:
        _term_cache[dq] = list(analyze_terms(dq, field="body"))
    return _term_cache[dq]


def aligned_markers(dq):
    """Subqueryの意図に一致する役割のマーカーだけを返す（提案手法の core）。"""
    if dq not in _role_cache:
        _role_cache[dq] = role_markers_for(heading_roles(dq))
    return _role_cache[dq]


def run_config(qids, queries, webstyle, marker_mode, slop, boost, label=""):
    run, t0, n_search = {}, time.time(), 0
    for i, qid in enumerate(qids, 1):
        entry = webstyle[qid]
        repeated_q = " ".join([queries[qid]] * QUERY_REPEAT)
        lists = []
        for dq, doc in zip(entry["decomposed_queries"], entry["query2doc_docs_structured"]):
            if not doc or not doc.get("body"):
                continue
            if marker_mode == "aligned":
                markers = aligned_markers(dq)
            elif marker_mode == "all":
                markers = ALL_ROLE_MARKERS
            else:
                markers = []
            r = bm25_equalweight_posboost_headingroleboost(
                f"{repeated_q} {dq} {doc['title']}",
                f"{repeated_q} {dq} {' '.join(doc['headings'])}",
                f"{repeated_q} {dq} {doc['body']}",
                span_terms_for(dq), markers, k=RETRIEVE_K,
                span_end=SPAN_END, span_boost=SPAN_BOOST,
                heading_slop=slop, heading_role_boost=boost)
            n_search += 1
            if r:
                lists.append(r)
        print(f"\r    {label}: {i}/{len(qids)} topics, {n_search} searches "
              f"({time.time()-t0:.0f}s)", end="", flush=True)
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        run[qid] = {d: float(s) for d, s in fused}
    print()
    return run


def score(run, qrels, qids):
    target = [q for q in qids if q in qrels]
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
    n = len(res)
    return {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=6,
                    help="トピックサブセットの間引き幅（6=18トピック, 3=35トピック）")
    args = ap.parse_args()

    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)
    valid = sorted(q for q in webstyle
                   if q in queries and q in qrels and webstyle[q].get("decomposed_queries"))
    qids = valid[::args.stride]
    print(f"全{len(valid)}トピック中、探索用サブセット{len(qids)}トピック"
          f"（stride={args.stride}）", flush=True)

    results = []
    state = {"search_qids_n": len(qids), "stride": args.stride, "span_end": SPAN_END,
             "span_boost": SPAN_BOOST, "results": results}

    def save():
        with open(OUT_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def measure(label, mode, slop, boost):
        t0 = time.time()
        agg, n = score(run_config(qids, queries, webstyle, mode, slop, boost, label), qrels, qids)
        row = {"label": label, "mode": mode, "slop": slop, "boost": boost, "n": n, **agg}
        results.append(row)
        print(f"  {label:34s} " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS)
              + f"   ({time.time()-t0:.0f}s)", flush=True)
        save()
        return agg

    print("\n[0] チャンピオン参照値（役割ブースト無し = posboost_only）")
    champ = measure("champion(posboost_only)", "none", 0, 0.0)
    state["champion"] = champ

    print("\n[1] slop掃引（aligned, boost=5.0）")
    best_slop, best_ndcg = None, -1
    for slop in SLOP_GRID:
        agg = measure(f"aligned slop={slop} boost=5.0", "aligned", slop, 5.0)
        if agg["ndcg_cut_10"] > best_ndcg:
            best_slop, best_ndcg = slop, agg["ndcg_cut_10"]
    print(f"  -> best_slop={best_slop}")
    state["best_slop"] = best_slop

    print(f"\n[2] boost掃引（aligned, slop={best_slop}）")
    best_boost, best_ndcg2 = 5.0, best_ndcg
    for b in BOOST_GRID:
        if b == 5.0:
            continue
        agg = measure(f"aligned slop={best_slop} boost={b}", "aligned", best_slop, b)
        if agg["ndcg_cut_10"] > best_ndcg2:
            best_boost, best_ndcg2 = b, agg["ndcg_cut_10"]
    print(f"  -> best_boost={best_boost}")
    state["best_boost"] = best_boost

    print(f"\n[3] 対照条件: 役割非依存（全役割のマーカーを常に使う, slop={best_slop}, boost={best_boost}）")
    measure(f"all-roles slop={best_slop} boost={best_boost}", "all", best_slop, best_boost)

    save()
    print(f"\n-> {OUT_FILE}")


if __name__ == "__main__":
    main()
