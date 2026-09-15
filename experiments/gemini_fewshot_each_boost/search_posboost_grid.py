"""
位置ブーストの (span_end, span_boost) を同時グリッドで探索する。

なぜやるか:
  既存の探索（equalweight_posboost_param_search_result.json など）は座標降下で、
  しかも span_end の探索範囲が {10, 30, 50, 100, 200} と 200 で打ち切られていた。
  そこでの結果は:
    - span_boost は内点に最適がある（35トピック: recall@1000 は 15、nDCG@10 は 8〜10 で最大、
      50 まで上げると明確に悪化）→ この軸は探索済みと見てよい
    - span_end は recall@1000 が 200 まで単調増加（0.2823 → 0.3008）で**頭打ちしていない**
  さらに別途 (span_end=200, span_boost=30) が現行 (100, 15) を上回ることも確認されており
  （equalweight_posboost_two_points_full_metrics.json）、2軸に交互作用がある可能性が高い。
  座標降下は交互作用があると最適点を外すので、ここは同時グリッドで見る。

span_end を大きく広げる根拠（2026-09-09 実測、116文書）:
  - msmarco-v21-doc の body は中央値 3,620 語
  - 1文書あたりのユニーク見出しは平均 21.8 個
  - **body 先頭100語に含まれる見出しは平均 29.2%（中央値 18.8%）**、
    先頭200語でも平均 37.2% しか覆えない。全見出しを覆えた文書は 10〜12% のみ
  つまり現行の span_end=100 は body の 2.8% しか見ておらず、見出しの約7割が
  位置ブーストの射程外にある。窓を広げれば後半の見出しを拾える可能性がある。

  一方 span_end を極端に大きくすると span_first は「body のどこかに出現」と実質同じになり、
  語を含む全候補に一律の加点をするだけになって選別力を失うはず。どこかに最適点があるか、
  それとも文書全体まで単調に伸びるのかを確かめる。

探索条件:
  検索構成は champion に固定（bm25_equalweight_posboost_discourseboost(markers=[])、
  title/headings/body 均等重み、疑似文書あり、QUERY_REPEAT=5、RETRIEVE_K=1000、RRF k=60）。
  変えるのは span_end と span_boost だけ。

  トピックは valid_qids[::3] の35件サブセット（既存のパラメータ探索と同じ取り方なので、
  過去の探索結果と直接比較できる）。見つかった最良点は別途105トピックで確認評価すること。

  recall@1000 と nDCG@10 は最適点が異なる（既存探索でも recall は 15、nDCG は 8〜10）ため、
  両方を全セルについて表示する。

使い方:
    python search_posboost_grid.py
    python search_posboost_grid.py --span-ends 100,200,400 --span-boosts 8,15 --limit 5
    python search_posboost_grid.py --full   # 35件サブセットではなく105トピック全部
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pytrec_eval

from retriever import bm25_equalweight_posboost_discourseboost, analyze_terms, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5

# span_end は現行100/既存上端200の先を見る。3200 は body 中央値3,620語のほぼ全体。
DEFAULT_SPAN_ENDS = [100, 200, 400, 800, 1600]
# span_boost は既存探索で内点最適が見えている範囲を、span_end との交互作用込みで再確認する。
DEFAULT_SPAN_BOOSTS = [8.0, 15.0, 30.0, 60.0]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


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


def run_cell(qids, queries, prepared, span_end, span_boost):
    run = {}
    for qid in qids:
        repeated_q = " ".join([queries[qid]] * QUERY_REPEAT)
        lists = []
        for dq, doc, span_terms in prepared[qid]:
            res = bm25_equalweight_posboost_discourseboost(
                f"{repeated_q} {dq} {doc['title']}",
                f"{repeated_q} {dq} {' '.join(doc['headings'])}",
                f"{repeated_q} {dq} {doc['body']}",
                span_terms, k=RETRIEVE_K,
                span_end=span_end, span_boost=span_boost, markers=[])
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
    ap.add_argument("--span-ends", default="")
    ap.add_argument("--span-boosts", default="")
    ap.add_argument("--full", action="store_true", help="35件サブセットでなく全トピックで回す")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    span_ends = ([int(x) for x in args.span_ends.split(",") if x.strip()]
                 if args.span_ends else list(DEFAULT_SPAN_ENDS))
    span_boosts = ([float(x) for x in args.span_boosts.split(",") if x.strip()]
                   if args.span_boosts else list(DEFAULT_SPAN_BOOSTS))

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
    search_qids = valid_qids if args.full else valid_qids[::3]
    if args.limit:
        search_qids = search_qids[:args.limit]

    prepared = prepare(webstyle, search_qids)
    search_qids = [q for q in search_qids if prepared[q]]
    print(f"全トピック {len(valid_qids)} / 探索対象 {len(search_qids)}")
    print(f"span_end   : {span_ends}")
    print(f"span_boost : {span_boosts}")
    print(f"= {len(span_ends) * len(span_boosts)} セル")
    print("=" * 72)

    cells, t_all = [], time.time()
    for se in span_ends:
        for sb in span_boosts:
            t0 = time.time()
            run = run_cell(search_qids, queries, prepared, se, sb)
            rec = {"span_end": se, "span_boost": sb, "seconds": round(time.time() - t0, 1)}
            for name in QREL_SETS:
                agg = score(run, qrels[name], search_qids)
                rec[name] = agg
            cells.append(rec)
            c = rec.get("consensus") or {}
            print(f"  span_end={se:5d} span_boost={sb:5.1f}  "
                  f"R@1000={c.get('recall_1000', float('nan')):.4f}  "
                  f"nDCG@10={c.get('ndcg_cut_10', float('nan')):.4f}  "
                  f"({rec['seconds']:.0f}s)", flush=True)
    print(f"\n全 {len(cells)} セル完了 ({time.time() - t_all:.0f}s)")

    # グリッド表（consensus）
    for name in QREL_SETS:
        for key, label in (("recall_1000", "recall@1000"), ("ndcg_cut_10", "nDCG@10")):
            print(f"\n[{name}] {label}   行=span_end 列=span_boost")
            print("span_end".ljust(10) + "".join(f"{sb:>10.1f}" for sb in span_boosts))
            print("-" * (10 + 10 * len(span_boosts)))
            for se in span_ends:
                row = f"{se}".ljust(10)
                for sb in span_boosts:
                    c = next((x for x in cells
                              if x["span_end"] == se and x["span_boost"] == sb), None)
                    v = (c.get(name) or {}).get(key) if c else None
                    row += f"{v:>10.4f}" if v is not None else " " * 10
                print(row)

    print("\n" + "=" * 72)
    print("指標ごとの最良セル")
    for name in QREL_SETS:
        for key, label in (("recall_1000", "recall@1000"), ("ndcg_cut_10", "nDCG@10"),
                           ("recall_100", "recall@100"), ("P_100", "precision@100")):
            ok = [c for c in cells if c.get(name) and c[name].get(key) is not None]
            if not ok:
                continue
            best = max(ok, key=lambda c: c[name][key])
            edge = ""
            if best["span_end"] in (span_ends[0], span_ends[-1]) or \
               best["span_boost"] in (span_boosts[0], span_boosts[-1]):
                edge = "  ← グリッドの端（範囲を広げる必要あり）"
            print(f"  [{name}] {label:14s} span_end={best['span_end']:5d} "
                  f"span_boost={best['span_boost']:5.1f}  {best[name][key]:.4f}{edge}")

    out_path = os.path.join(
        RAG_DIR, f"posboost_grid_result{'_full' if args.full else ''}.json")
    with open(out_path, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "full": args.full,
                   "span_ends": span_ends, "span_boosts": span_boosts,
                   "retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT,
                   "qids": search_qids, "cells": cells}, f, ensure_ascii=False, indent=2)
    print(f"\n集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
