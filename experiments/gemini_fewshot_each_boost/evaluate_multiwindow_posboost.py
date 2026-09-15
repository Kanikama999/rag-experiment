"""
位置ブーストを「単一固定窓」から「多段窓」に置き換えて比較する。

## 問題

現行の位置ブーストは span_first(end=100, boost=15) を bool/should に1節足すだけである。
2026-09-09 に実装と挙動を実測して、次の2つの制約が確定した。

1. **位置を連続値として使えていない。**
   span_first は Lucene の SpanFirstQuery（SpanPositionRangeQuery(0, end)）で、
   スコアは「窓 [0,end) 内の span 頻度」を tf とした BM25 である。explain の実測:

       end= 50 → score(freq= 15.0) = boost * idf * tf
       end=100 → score(freq= 28.0)
       end=400 → score(freq=145.0)

   窓内であればどこでマッチしても同じ扱いで、「どれだけ早く出たか」は見ていない。
   また **窓を広げるほど freq が増えてスコアが上がる**方向に働くため、
   段ごとの寄与は boost の定数ではない（後述のスケジュール設計はこれを前提にする）。
   最初のマッチ位置を数値として取るには _script_score / rescore が要るが、
   現状のパイプラインは使っていない。

2. **固定窓は全文包含バイアスを生む。**
   無作為サンプル300文書の body 語数は中央値938語。span_end 以下（＝窓が全文を覆い、
   位置情報が消える）文書の割合は end=100 で 2.0%、200 で 6.0%、400 で 16.0%、
   800 で 42.7%、1600 で 75.0%。窓を広げるほど「位置ブースト」は
   「語を含むかどうか」に退化していく。

## 比較する案

  single  : 現行。span_first(end=100, boost=15) 1節。
  cascade : 入れ子の累積窓。[(50,b1),(100,b2),(200,b3),(400,b4)] を全部 should に足す。
            先頭で当たった語ほど多くの段に入り加点が積み上がるので、位置減衰関数の
            階段近似になる。Lucene 標準クエリだけで実装できるのが利点。
            ただし窓が入れ子なので、**全文が窓に収まる短い文書は全段を総取りする**。
            つまり cascade は上記の問題2を解決しない（段が増えるぶん増幅しうる）。
  ring    : span_not で作る排他リング [0,50),[50,100),[100,200),[200,400)。
            80語の文書は [200,400) に原理的にマッチできないため総取りが起きない。
            **問題2に直接対処するのはこちら。**
            2026-09-09 実測で ring[200,400) にマッチした文書の body 語数は最小350語、
            200語未満は0件であることを確認済み。

## 段ごとの boost をどう決めるか

freq が窓幅とともに増えるため、理論的に決められない。スケジュールを振って実測する。

  flat    : 全段同じ重み（位置勾配なし。対照）
  geom0.5 : 8:4:2:1（窓幅が倍々なので boost×end 一定＝「面積一定」と数学的に同一）
  geom0.25: 64:16:4:1（急減衰。冒頭を強く優遇）

いずれも合計が SCALE になるよう正規化するので、スケジュール間で「配る加点の総量」は
揃う。SCALE は現行 champion と同じ 15 と、その倍の 30 を見る。

検索構成は champion に固定（title/headings/body 均等重み、疑似文書あり、
QUERY_REPEAT=5、RETRIEVE_K=1000、RRF k=60）。変えるのは位置ブースト節だけ。
トピックは valid_qids[::3] の35件サブセット（既存のパラメータ探索と同じ取り方）。

使い方:
    python evaluate_multiwindow_posboost.py
    python evaluate_multiwindow_posboost.py --limit 3
    python evaluate_multiwindow_posboost.py --full     # 105トピック
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pytrec_eval

from retriever import (bm25_equalweight_multiwindow,
                       bm25_equalweight_posboost_discourseboost,
                       analyze_terms, rrf_fuse)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5

ENDS = [50, 100, 200, 400]
SCHEDULES = {
    "flat":     [1.0, 1.0, 1.0, 1.0],
    "geom0.5":  [8.0, 4.0, 2.0, 1.0],
    "geom0.25": [64.0, 16.0, 4.0, 1.0],
}
SCALES = [15.0, 30.0]
MODES = ["cascade", "ring"]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


def make_tiers(schedule, scale):
    """合計が scale になるよう正規化した [(end, boost), ...] を返す。"""
    w = SCHEDULES[schedule]
    total = sum(w)
    return [(e, scale * x / total) for e, x in zip(ENDS, w)]


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
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


def prepare(webstyle, qids):
    out = {}
    for qid in qids:
        entry = webstyle[qid]
        pairs = []
        for dq, doc in zip(entry.get("decomposed_queries") or [],
                           entry.get("query2doc_docs_structured") or []):
            if not doc or not doc.get("body"):
                continue
            pairs.append((dq, doc, analyze_terms(dq, field="body")))
        out[qid] = pairs
    return out


def run_condition(qids, queries, prepared, mode, tiers):
    run = {}
    for qid in qids:
        repeated_q = " ".join([queries[qid]] * QUERY_REPEAT)
        lists = []
        for dq, doc, span_terms in prepared[qid]:
            t_q = f"{repeated_q} {dq} {doc['title']}"
            h_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
            b_q = f"{repeated_q} {dq} {doc['body']}"
            if mode == "champion":
                res = bm25_equalweight_posboost_discourseboost(
                    t_q, h_q, b_q, span_terms, k=RETRIEVE_K,
                    span_end=100, span_boost=15.0, markers=[])
            else:
                res = bm25_equalweight_multiwindow(
                    t_q, h_q, b_q, span_terms, k=RETRIEVE_K, tiers=tiers, mode=mode)
            if res:
                lists.append(res)
        run[qid] = ({docid: float(sc) for docid, sc in rrf_fuse(lists, top_n=TOPK)}
                    if lists else {})
    return run


def score(run, qrels, qids):
    target = [q for q in qids if q in qrels and run.get(q)]
    if not target:
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
    n = len(res)
    return {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
        and any(d and d.get("body") for d in webstyle[qid].get("query2doc_docs_structured", []))
    )
    qids = valid_qids if args.full else valid_qids[::3]
    if args.limit:
        qids = qids[:args.limit]
    prepared = prepare(webstyle, qids)
    qids = [q for q in qids if prepared[q]]

    conditions = [("champion", None, None, None)]
    for mode in MODES:
        for sch in SCHEDULES:
            for sc in SCALES:
                conditions.append((mode, sch, sc, make_tiers(sch, sc)))

    print(f"全トピック {len(valid_qids)} / 対象 {len(qids)}")
    print(f"窓: {ENDS}   スケジュール: {list(SCHEDULES)}   スケール: {SCALES}")
    print(f"= {len(conditions)} 条件")
    print("=" * 72)

    cells, t_all = [], time.time()
    for mode, sch, sc, tiers in conditions:
        label = "champion" if mode == "champion" else f"{mode}_{sch}_s{sc:g}"
        t0 = time.time()
        run = run_condition(qids, queries, prepared, mode, tiers)
        rec = {"label": label, "mode": mode, "schedule": sch, "scale": sc,
               "tiers": tiers, "seconds": round(time.time() - t0, 1)}
        for name in QREL_SETS:
            rec[name] = score(run, qrels[name], qids)
        cells.append(rec)
        c = rec.get("consensus") or {}
        print(f"  {label:22s} R@1000={c.get('recall_1000', float('nan')):.4f}  "
              f"nDCG@10={c.get('ndcg_cut_10', float('nan')):.4f}  "
              f"({rec['seconds']:.0f}s)", flush=True)
    print(f"\n全 {len(cells)} 条件完了 ({time.time() - t_all:.0f}s)")

    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    base = next((c for c in cells if c["label"] == "champion"), None)
    for name in QREL_SETS:
        print(f"\n[{name}]  (括弧内は champion との差)")
        header = "condition".ljust(24) + "".join(labels[k].ljust(18) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        for c in cells:
            agg = c.get(name)
            row = c["label"].ljust(24)
            if not agg:
                print(row + "-")
                continue
            for k in METRIC_KEYS:
                cell = f"{agg[k]:.4f}"
                if base and base.get(name) and c["label"] != "champion":
                    cell += f" ({agg[k] - base[name][k]:+.4f})"
                row += cell.ljust(18)
            print(row)

        ok = [c for c in cells if c.get(name)]
        if not ok:
            print(f"\n  [{name}] 採点対象なし（この qrels に含まれるトピックが無い）")
            continue
        print(f"\n  [{name}] 指標ごとの最良")
        for k in METRIC_KEYS:
            best = max(ok, key=lambda c: c[name][k])
            print(f"    {labels[k]:14s} {best['label']:22s} {best[name][k]:.4f}")

    print("\n" + "=" * 72)
    print("検索コスト（1条件あたりの総秒数）")
    for c in cells:
        rel = c["seconds"] / base["seconds"] if base and base["seconds"] else float("nan")
        print(f"  {c['label']:22s} {c['seconds']:7.1f}s  (champion比 {rel:.2f}倍)")

    out_path = os.path.join(
        RAG_DIR, f"multiwindow_posboost_result{'_full' if args.full else ''}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(qids), "full": args.full, "ends": ENDS,
                   "schedules": SCHEDULES, "scales": SCALES,
                   "retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT,
                   "qids": qids, "cells": cells}, f, ensure_ascii=False, indent=2)
    print(f"\n集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
