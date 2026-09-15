"""
見出し役割シグナルを「検索式への加点」ではなく「上位N件の再ランキング特徴量」として使う版。

search_headingrole_params.py（span_nearによる加点）はチャンピオンを一貫して下回った。
原因は、事前分析で識別力を確認した条件と、検索式が表現している条件がずれていること:

  分析側 : 同一見出しの中に Subquery の内容語が MIN_OVERLAP(=2) 語以上あり、
           かつその見出しの役割が Subquery の意図と一致（lift 4.90）
  検索式 : headings フィールドのどこかで役割語とクエリ語1語が slop 以内に近接
           （＝はるかに緩い。しかも headings への match 節と重複している）

ここでは分析側とまったく同じ条件を、チャンピオンの融合済み上位RERANK_N件に対して
クライアント側で計算し、RRFスコアへ加点する:

    score' = rrf_score + alpha * feature

feature は aligned / crossed / plain / any の4種を用意し、aligned が他を上回るかで
「役割と意図の対応」に意味があるかを判定する。alphaは0を含めて掃引するので、
alpha=0 の行がそのままチャンピオンの再現値になる。

チャンピオンの検索結果はJSONにキャッシュするので、2回目以降のalpha掃引は検索なしで済む。

使い方:
    python evaluate_headingrole_rerank.py [--stride N] [--rerank-n N] [--refresh]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pytrec_eval

from retriever import (rrf_fuse, analyze_terms, heading_roles,
                       bm25_equalweight_posboost_headingroleboost)
from measure_heading_role_conditional import doc_heading_units
from measure_heading_role_distribution import fetch_headings

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
RUN_CACHE = os.path.join(RAG_DIR, "_cache_champion_run_rep5.json")
OUT_FILE = os.path.join(RAG_DIR, "headingrole_rerank_eval_summary_rep5.json")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
SPAN_END = 100
SPAN_BOOST = 15.0
MIN_OVERLAP = 2          # measure_heading_role_conditional.py と同じ

FEATURES = ["aligned", "crossed", "plain", "any"]
# RRFスコアは1/(60+rank)の和なので、上位付近の隣接ランク差は1e-4程度しかない。
# 最初 0.001 刻みで掃引したところ全featureが単調に悪化したが、それは
# 「シグナルが無い」のではなく1段の加点が10ランク分に相当していたため。
ALPHA_GRID = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]

METRIC_SPECS = ["recall.100", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]

ROLE_TERMS_TEXT = (
    "what is are definition defined meaning mean how it works explained detail details "
    "overview about history origin origins etymology background faq frequently asked "
    "questions summary conclusion key takeaways short bottom line recap warning caution "
    "risk risks side effects precautions important note vs versus difference between "
    "compared comparison steps guide instructions tutorial why cause causes reason reasons")


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


def champion_run(qids, queries, webstyle, refresh=False):
    """チャンピオン（equalweight + posboost、役割ブースト無し）の融合済みランキング。"""
    cache = {}
    if os.path.exists(RUN_CACHE) and not refresh:
        cache = json.load(open(RUN_CACHE))
    todo = [q for q in qids if q not in cache]
    if todo:
        print(f"チャンピオン検索: {len(todo)} トピック（キャッシュ済み {len(qids)-len(todo)}）")
        t0 = time.time()
        for i, qid in enumerate(todo, 1):
            repeated_q = " ".join([queries[qid]] * QUERY_REPEAT)
            lists = []
            for dq, doc in zip(webstyle[qid]["decomposed_queries"],
                               webstyle[qid]["query2doc_docs_structured"]):
                if not doc or not doc.get("body"):
                    continue
                r = bm25_equalweight_posboost_headingroleboost(
                    f"{repeated_q} {dq} {doc['title']}",
                    f"{repeated_q} {dq} {' '.join(doc['headings'])}",
                    f"{repeated_q} {dq} {doc['body']}",
                    list(analyze_terms(dq, field="body")), [], k=RETRIEVE_K,
                    span_end=SPAN_END, span_boost=SPAN_BOOST)
                if r:
                    lists.append(r)
            cache[qid] = [[d, float(s)] for d, s in (rrf_fuse(lists, top_n=TOPK) if lists else [])]
            print(f"\r  {i}/{len(todo)} ({time.time()-t0:.0f}s)", end="", flush=True)
        print()
        with open(RUN_CACHE, "w") as f:
            json.dump(cache, f)
    return {q: cache[q] for q in qids}


def doc_features(headings_text, sq_units, role_terms):
    """分析スクリプトと同一条件で aligned/crossed/plain/any を数える。
    値は『その条件を満たす見出しを持つSubqueryの本数』。"""
    units = doc_heading_units(headings_text)
    feat = dict.fromkeys(FEATURES, 0)
    if not units:
        return feat
    for terms, intents in sq_units:
        hit = {k: False for k in FEATURES}
        for raw, stems in units:
            if len((stems - role_terms) & terms) < MIN_OVERLAP:
                continue
            hit["any"] = True
            hroles = heading_roles(raw)
            if not hroles:
                hit["plain"] = True
            elif hroles & intents:
                hit["aligned"] = True
            else:
                hit["crossed"] = True
        for k, v in hit.items():
            feat[k] += int(v)
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--rerank-n", type=int, default=100)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}

    valid = sorted(q for q in webstyle
                   if q in queries and q in qrels["consensus"]
                   and webstyle[q].get("decomposed_queries"))
    qids = valid[::args.stride]
    print(f"全{len(valid)}トピック中 {len(qids)}トピック（stride={args.stride}）  "
          f"rerank_n={args.rerank_n}  MIN_OVERLAP={MIN_OVERLAP}")

    runs = champion_run(qids, queries, webstyle, args.refresh)
    role_terms = set(analyze_terms(ROLE_TERMS_TEXT, field="headings"))

    print("見出し特徴量を計算中...")
    feats, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        sq_units = []
        for dq in webstyle[qid]["decomposed_queries"]:
            terms = set(analyze_terms(dq, field="headings")) - role_terms
            if terms:
                sq_units.append((terms, heading_roles(dq)))
        top = [d for d, _ in runs[qid][:args.rerank_n]]
        heads = fetch_headings(top)
        feats[qid] = {d: doc_features(heads.get(d, ""), sq_units, role_terms) for d in top}
        print(f"\r  {i}/{len(qids)} ({time.time()-t0:.0f}s)", end="", flush=True)
    print()

    summary = {}
    for name in QREL_SETS:
        target = [q for q in qids if q in qrels[name]]
        ev = pytrec_eval.RelevanceEvaluator({q: qrels[name][q] for q in target}, METRICS)
        for feature in FEATURES:
            for alpha in ALPHA_GRID:
                run = {}
                for qid in target:
                    scored = {d: s for d, s in runs[qid]}
                    for d, f in feats[qid].items():
                        scored[d] = scored[d] + alpha * f[feature]
                    run[qid] = scored
                res = ev.evaluate(run)
                n = len(res)
                agg = {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}
                summary.setdefault(name, {}).setdefault(feature, {})[str(alpha)] = agg
                if alpha == 0.0 and feature != FEATURES[0]:
                    continue  # alpha=0は全featureで同じ値なので1回だけ出す

    labels = {"recall_100": "recall@100", "ndcg_cut_10": "nDCG@10", "P_100": "P@100"}
    for name in QREL_SETS:
        print(f"\n[{name}]  (alpha=0 がチャンピオン)")
        header = "feature/alpha".ljust(22) + "".join(labels[k].ljust(14) for k in METRIC_KEYS)
        print(header); print("-" * len(header))
        base = summary[name][FEATURES[0]]["0.0"]
        print("champion (alpha=0)".ljust(22)
              + "".join(f"{base[k]:.4f}".ljust(14) for k in METRIC_KEYS))
        for feature in FEATURES:
            for alpha in ALPHA_GRID:
                if alpha == 0.0:
                    continue
                agg = summary[name][feature][str(alpha)]
                row = f"{feature} a={alpha}".ljust(22)
                for k in METRIC_KEYS:
                    row += f"{agg[k]:.4f} ({agg[k]-base[k]:+.4f})".ljust(14)
                print(row)

    with open(OUT_FILE, "w") as f:
        json.dump({"n_topics": len(qids), "stride": args.stride, "rerank_n": args.rerank_n,
                   "min_overlap": MIN_OVERLAP, "alpha_grid": ALPHA_GRID,
                   "features": FEATURES, "summary": summary}, f, indent=2)
    print(f"\n-> {OUT_FILE}")


if __name__ == "__main__":
    main()
