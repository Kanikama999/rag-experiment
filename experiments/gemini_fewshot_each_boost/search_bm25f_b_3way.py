"""
BM25F の b_title/b_headings/b_body を3軸同時グリッドで探索する（avgdl バグ修正後）。

search_bm25f_field_b.py は座標降下（段階1で b_body=0.4 固定 → 段階2で body のみ）だったが、
2026-09-10 の avgdl バグ修正後に回したところ b_body がグリッド上端 0.8 で単調増加のまま
打ち切られ、b_title も下端 0.0 だった。端で止まる問題が2軸で同時に起きているので、
交互作用込みで3軸を同時に振り直す。b は BM25 の定義上 [0,1] なので 1.0 を含める。

注意: recall@1000 は candidate_k 件の候補プールの並べ替えしか起きないため b に依存せず
一定になる（探索できるのは nDCG@10 / P@100 / recall@100 のみ）。
"""
from __future__ import annotations
import itertools, json, os, time
import pytrec_eval
from retriever import bm25f_prepare, bm25f_score, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK, CANDIDATE_K, QUERY_REPEAT = 1000, 200, 5
TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
BM25_K1 = 0.9
B_GRID = [0.0, 0.4, 0.8, 1.0]
METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS); KEYS = [m.replace(".", "_") for m in METRIC_SPECS]

def load_qrels(p):
    q = {}
    for line in open(p):
        f = line.strip().split()
        if len(f) == 4: q.setdefault(f[0], {})[f[2]] = int(f[3])
    return q

def main():
    web = json.load(open(WEBSTYLE_FILE))["results"]
    queries = {json.loads(l)["id"]: json.loads(l)["title"] for l in open(QUERIES_FILE) if l.strip()}
    qrels = load_qrels(QRELS_FILE)
    qids = sorted(q for q in web if q in queries and web[q].get("decomposed_queries"))[::3]
    print(f"探索対象 {len(qids)} トピック / grid {B_GRID} の3軸 = {len(B_GRID)**3} セル")

    print("候補プール・IDF・mtermvectors を取得中（b 非依存、1回だけ）...")
    raw_lists, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        rep = " ".join([queries[qid]] * QUERY_REPEAT)
        lst = []
        for dq, doc in zip(web[qid]["decomposed_queries"], web[qid]["query2doc_docs_structured"]):
            if not doc or not doc.get("body"): continue
            lst.append(bm25f_prepare(f"{rep} {dq} {doc['title']}",
                                     f"{rep} {dq} {' '.join(doc['headings'])}",
                                     f"{rep} {dq} {doc['body']}",
                                     title_weight=TITLE_W, headings_weight=HEADINGS_W,
                                     body_weight=BODY_W, candidate_k=CANDIDATE_K))
        raw_lists[qid] = lst
        if i % 5 == 0 or i == len(qids):
            print(f"\r  {i}/{len(qids)} ({time.time()-t0:.0f}s)", end="", flush=True)
    print()

    cells = []
    for bt, bh, bb in itertools.product(B_GRID, repeat=3):
        run = {}
        for qid in qids:
            lists = [bm25f_score(raw, k=TOPK, title_weight=TITLE_W, headings_weight=HEADINGS_W,
                                 body_weight=BODY_W, b_title=bt, b_headings=bh, b_body=bb,
                                 bm25_k1=BM25_K1) for raw in raw_lists[qid]]
            lists = [x for x in lists if x]
            run[qid] = {d: float(s) for d, s in rrf_fuse(lists, top_n=TOPK)} if lists else {}
        tgt = [q for q in qids if q in qrels and run.get(q)]
        ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in tgt}, METRICS)
        res = ev.evaluate({q: run[q] for q in tgt})
        agg = {m: sum(r[m] for r in res.values()) / len(res) for m in KEYS}
        cells.append({"b_title": bt, "b_headings": bh, "b_body": bb, **agg})
        print(f"  b=({bt},{bh},{bb})  nDCG@10={agg['ndcg_cut_10']:.4f}  "
              f"R@100={agg['recall_100']:.4f}  P@100={agg['P_100']:.4f}", flush=True)

    print("\n" + "=" * 60)
    for key in ("ndcg_cut_10", "recall_100", "P_100"):
        best = max(cells, key=lambda c: c[key])
        edge = [n for n, v in (("b_title", best["b_title"]), ("b_headings", best["b_headings"]),
                               ("b_body", best["b_body"])) if v in (B_GRID[0], B_GRID[-1])]
        print(f"{key:12s} 最良 b=({best['b_title']},{best['b_headings']},{best['b_body']}) "
              f"{best[key]:.4f}" + (f"   ← 端: {','.join(edge)}" if edge else ""))
    out = os.path.join(RAG_DIR, "bm25f_b_3way_grid_result.json")
    json.dump({"n_queries": len(qids), "grid": B_GRID, "candidate_k": CANDIDATE_K,
               "weights": [TITLE_W, HEADINGS_W, BODY_W], "k1": BM25_K1, "cells": cells},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"-> {out}")

if __name__ == "__main__":
    main()
