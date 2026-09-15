"""
見出し役割（heading role）ブーストの105トピック評価。

背景: body中の談話標識（"in conclusion"等）を使うdiscourseboostは、チャンピオンに
足しても効果がほぼ無かった（consensus nDCG@10 0.5251 -> 0.5225）。事前分析
（measure_heading_role_conditional.py / _alignment.py）で分かったのは

  - 要約系マーカーは relevant/nonrelevant の分離力が最も弱い部類（lift 1.19〜2.12）
  - 一方 definition / cause / comparison の見出しは lift 2.5〜3.3 と明確に強い
  - さらに「Subqueryの意図」と「見出しの役割」が一致した場合が最も強い
    （plain 1.84 < crossed 2.20 < aligned 3.06）

そこで着目点を body から headings フィールドへ移し、Subqueryの意図に一致する役割語と
クエリ語が同一見出し内で近接している文書を加点する。

条件:
  baseline                                  narrativeそのままのBM25
  ..._equalweight_posboost_only             現行チャンピオン（役割ブースト無し）
  ..._equalweight_headingrole_aligned       提案手法（Subquery意図に一致する役割のみ）
  ..._equalweight_headingrole_all           対照（役割を問わず全マーカーを常に使う）

alignedがallを上回るかどうかが、「役割という軸そのもの」に意味があるかの判定になる。
allの方が良ければ、効いているのは意図との対応ではなく単なる見出し語の近接である。

使い方:
    python evaluate_decomposed_webstyle_headingrole.py [SLOP] [BOOST] [QUERY_REPEAT]
    （SLOP/BOOSTの既定値は search_headingrole_params.py の探索結果に合わせる）
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, rrf_fuse, analyze_terms,
                       bm25_equalweight_posboost_headingroleboost,
                       heading_roles, role_markers_for, HEADING_ROLE_MARKERS)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
PARAM_FILE = os.path.join(RAG_DIR, "headingrole_param_search_result.json")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
SPAN_END = 100
SPAN_BOOST = 15.0

_defaults = {"slop": 6, "boost": 5.0}
if os.path.exists(PARAM_FILE):
    try:
        _p = json.load(open(PARAM_FILE))
        _defaults = {"slop": _p["best_slop"], "boost": _p["best_boost"]}
    except Exception:
        pass

HEADING_SLOP = int(sys.argv[1]) if len(sys.argv) > 1 else _defaults["slop"]
HEADING_BOOST = float(sys.argv[2]) if len(sys.argv) > 2 else _defaults["boost"]
QUERY_REPEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 5

CHAMPION = "webstyle_narrative_equalweight_posboost_only"
ALIGNED = "webstyle_narrative_equalweight_headingrole_aligned"
ALLROLES = "webstyle_narrative_equalweight_headingrole_all"
METHODS = ["baseline", CHAMPION, ALIGNED, ALLROLES]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]

ALL_ROLE_MARKERS = role_markers_for(list(HEADING_ROLE_MARKERS))


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


def markers_for(method, dq):
    if method == ALIGNED:
        if dq not in _role_cache:
            _role_cache[dq] = role_markers_for(heading_roles(dq))
        return _role_cache[dq]
    if method == ALLROLES:
        return ALL_ROLE_MARKERS
    return []


def fuse(lists):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {d: float(s) for d, s in rrf_fuse(lists, top_n=TOPK)}


def build_run(method, qids, queries, webstyle):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        if method == "baseline":
            run[qid] = fuse([bm25_body(q, k=RETRIEVE_K)])
        else:
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = []
            for dq, doc in zip(webstyle[qid]["decomposed_queries"],
                               webstyle[qid]["query2doc_docs_structured"]):
                if not doc or not doc.get("body"):
                    continue
                lists.append(bm25_equalweight_posboost_headingroleboost(
                    f"{repeated_q} {dq} {doc['title']}",
                    f"{repeated_q} {dq} {' '.join(doc['headings'])}",
                    f"{repeated_q} {dq} {doc['body']}",
                    span_terms_for(dq), markers_for(method, dq), k=RETRIEVE_K,
                    span_end=SPAN_END, span_boost=SPAN_BOOST,
                    heading_slop=HEADING_SLOP, heading_role_boost=HEADING_BOOST))
            run[qid] = fuse(lists)
        if i % 10 == 0 or i == len(qids):
            print(f"\r  {method}: {i}/{len(qids)} ({time.time()-t0:.0f}s)", end="", flush=True)
    print()
    return run


def evaluate(run, qrels, eval_qids):
    target = [q for q in eval_qids if q in qrels]
    if not target:
        return None, 0
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
    n = len(res)
    return {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}, n


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}

    valid_qids = sorted(q for q in webstyle
                        if q in queries and webstyle[q].get("decomposed_queries"))
    n_aligned_sq = sum(1 for q in valid_qids
                       for dq in webstyle[q]["decomposed_queries"] if heading_roles(dq))
    n_sq = sum(len(webstyle[q]["decomposed_queries"]) for q in valid_qids)
    print(f"{len(valid_qids)} クエリ / Subquery {n_sq} 本（うち意図ラベルが付いたもの "
          f"{n_aligned_sq} 本 = {100*n_aligned_sq/max(n_sq,1):.1f}%）")
    print(f"heading_slop={HEADING_SLOP}  heading_role_boost={HEADING_BOOST}  "
          f"span_end={SPAN_END}  span_boost={SPAN_BOOST}  QUERY_REPEAT={QUERY_REPEAT}")
    print("=" * 72)

    runs = {m: build_run(m, valid_qids, queries, webstyle) for m in METHODS}
    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in METHODS)]
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    summary = {}
    for name in QREL_SETS:
        for method in METHODS:
            agg, n = evaluate(runs[method], qrels[name], eval_qids)
            summary.setdefault(name, {})[method] = agg

    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "P@100"}
    for name in QREL_SETS:
        print(f"\n[{name}]")
        header = "method".ljust(54) + "".join(labels[k].ljust(18) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        champ = summary[name].get(CHAMPION)
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(54)
            if agg is None:
                print(row + "-")
                continue
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method not in ("baseline", CHAMPION) and champ:
                    cell += f" ({agg[key]-champ[key]:+.4f})"
                row += cell.ljust(18)
            print(row)
    print("\n※ 括弧内はチャンピオン（posboost_only）との差")

    out = os.path.join(RAG_DIR, "decomposed_query2doc_webstyle_eval_summary_headingrole_rep5.json")
    with open(out, "w") as f:
        json.dump({"n_queries": len(eval_qids), "heading_slop": HEADING_SLOP,
                   "heading_role_boost": HEADING_BOOST, "span_end": SPAN_END,
                   "span_boost": SPAN_BOOST, "query_repeat": QUERY_REPEAT,
                   "qids": eval_qids, "summary": summary}, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
