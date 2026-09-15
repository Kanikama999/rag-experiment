"""
主要手法を「正解とみなす grade の閾値」を変えて測り直す（doc版）。

背景:
  consensus qrels は1トピックあたり関連文書(grade>=1)が平均1,334件あり、105トピック
  すべてが100件を超える。そのため recall@100 の理論上限が 0.0854 しかなく、champion の
  0.0689 は既に上限の81%に達している。「recall@100 が baseline の2倍」という報告は、
  実質「天井の38%→81%」を意味しており、指標としての識別力をほぼ失っている。
  grade>=1 の内訳は grade1 が96.6%（135,342/140,091件）で、この物量が分母を作っている。

  grade>=2 に絞ると1トピック45件（中央値47）に落ち着き、recall@100 の上限は 1.0 になる。
  そこで主要手法を relevance_level = 1 / 2 / 3 の3通りで測り直し、どの閾値で手法間の差が
  最もよく見えるかを確認する。

注意:
  pytrec_eval の relevance_level は二値指標（recall / P / map）にのみ効く。nDCG は
  常に段階的利得（qrelsのgradeそのもの）で計算されるため閾値を変えても値は変わらない。
  よって nDCG は1度だけ報告する。

手法:
  baseline           narrative そのまま bm25_body
  fielded_recallopt  フィールド線形和 (title=2, headings=1, body=1)、位置ブーストなし
  champion           均等重み + 位置ブースト (span_end=100, span_boost=15)
  idf_posboost_s5    champion + IDF項重み付け + Subquery×5（既存キャッシュから融合、検索不要）
  bm25f_qtf          クエリ項頻度を尊重した BM25F（既存 run を読み込むだけ、検索不要）

検索した run は runs ファイルに保存するので、以後どの閾値・どの指標でも再検索なしで
測り直せる（既存の評価スクリプトは集計値しか残さないため、毎回再検索が必要だった）。

使い方:
    python evaluate_relevance_level.py              # 105トピック全部
    python evaluate_relevance_level.py --limit 3    # 動作確認
    python evaluate_relevance_level.py --reuse-runs # 保存済み run だけで再集計（検索なし）
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pytrec_eval

from retriever import (bm25_body, bm25_fielded, bm25_equalweight_posboost_discourseboost,
                       analyze_terms, rrf_fuse)
from term_weights import idf_term_weights

HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "data")
HERE = os.path.dirname(os.path.abspath(__file__))

QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
WEBSTYLE_FILE = os.path.join(HERE, "multi_query2doc_decomposed_webstyle_L200.json")
IDFW_CACHE = os.path.join(HERE, "_cache_lists_idfweighted_n5_s5_gemini.json")
BM25F_RUNS = os.path.join(HERE, "bm25f_narrative_rep5_runs.json")
QRELS = {"coverage": os.path.join(DATA_DIR, "keystone_qrels_coverage.txt"),
         "consensus": os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")}

RUNS_OUT = os.path.join(HERE, "relevance_level_runs.json")
RESULT_OUT = os.path.join(HERE, "relevance_level_result.json")

TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
SPAN_END = 100
SPAN_BOOST = 15.0
LEVELS = [1, 2, 3]

BINARY_METRICS = {"recall.100", "recall.1000", "P.10", "P.100", "map"}
BINARY_KEYS = ["recall_100", "recall_1000", "P_10", "P_100", "map"]
NDCG_METRICS = {"ndcg_cut.10", "ndcg_cut.100"}
NDCG_KEYS = ["ndcg_cut_10", "ndcg_cut_100"]

SEARCHED = ["baseline", "fielded_recallopt", "champion"]
FREE = ["idf_posboost_s5", "bm25f_qtf"]


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4 or line.startswith("#"):
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


def fuse(lists, topk=TOPK):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


def build_searched_runs(qids, queries, entries):
    """検索が必要な3手法の run を作る。"""
    runs = {m: {} for m in SEARCHED}
    for method in SEARCHED:
        t0 = time.time()
        for i, qid in enumerate(qids, 1):
            q = queries[qid]
            repeated_q = " ".join([q] * QUERY_REPEAT)
            if method == "baseline":
                runs[method][qid] = fuse([bm25_body(q.strip(), k=RETRIEVE_K)])
            else:
                lists = []
                for dq, doc, span_terms in entries[qid]:
                    title_q = f"{repeated_q} {dq} {doc['title']}"
                    headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
                    body_q = f"{repeated_q} {dq} {doc['body']}"
                    if method == "fielded_recallopt":
                        lists.append(bm25_fielded(title_q, headings_q, body_q,
                                                  k=RETRIEVE_K, title_boost=2,
                                                  headings_boost=1, body_boost=1))
                    else:  # champion
                        lists.append(bm25_equalweight_posboost_discourseboost(
                            title_q, headings_q, body_q, span_terms, k=RETRIEVE_K,
                            span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[]))
                runs[method][qid] = fuse(lists)
            if i % 10 == 0 or i == len(qids):
                print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s)",
                      end="", flush=True)
        print()
    return runs


def load_free_runs(qids):
    """検索済みのキャッシュから作れる2手法。"""
    runs = {}
    if os.path.exists(IDFW_CACHE):
        with open(IDFW_CACHE) as f:
            lists_by_qid = json.load(f)
        runs["idf_posboost_s5"] = {
            qid: fuse([[(d, s) for d, s in lst] for lst in lists_by_qid[qid]])
            for qid in qids if qid in lists_by_qid}
        print(f"  idf_posboost_s5: キャッシュから融合 ({len(runs['idf_posboost_s5'])}トピック)")
    if os.path.exists(BM25F_RUNS):
        with open(BM25F_RUNS) as f:
            bm25f = json.load(f)
        if "bm25f_qtf" in bm25f:
            runs["bm25f_qtf"] = {qid: bm25f["bm25f_qtf"][qid]
                                 for qid in qids if qid in bm25f["bm25f_qtf"]}
            print(f"  bm25f_qtf: 既存runを読み込み ({len(runs['bm25f_qtf'])}トピック)")
    return runs


def score(run, qrels, level):
    target = [q for q in run if q in qrels]
    if not target:
        return None, 0
    sub = {q: qrels[q] for q in target}
    sub_run = {q: run[q] for q in target}
    ev = pytrec_eval.RelevanceEvaluator(sub, BINARY_METRICS, relevance_level=level)
    res = ev.evaluate(sub_run)
    n = len(res)
    agg = {m: sum(r[m] for r in res.values()) / n for m in BINARY_KEYS}
    if level == 1:
        ev2 = pytrec_eval.RelevanceEvaluator(sub, NDCG_METRICS)
        res2 = ev2.evaluate(sub_run)
        for m in NDCG_KEYS:
            agg[m] = sum(r[m] for r in res2.values()) / n
    return agg, n


def ceiling(qrels, qids, level, cut):
    vals = []
    for q in qids:
        if q not in qrels:
            continue
        R = sum(1 for g in qrels[q].values() if g >= level)
        if R:
            vals.append(min(cut, R) / R)
    return sum(vals) / len(vals) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reuse-runs", action="store_true",
                    help="保存済み run だけで再集計する（検索を一切しない）")
    args = ap.parse_args()

    print("読み込み中...")
    queries = load_queries(QUERIES_FILE)
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    qrels = {name: load_qrels(path) for name, path in QRELS.items()}

    qids = sorted(q for q in webstyle if q in queries)
    if args.limit:
        qids = qids[:args.limit]
    print(f"対象 {len(qids)} トピック")

    if args.reuse_runs and os.path.exists(RUNS_OUT):
        with open(RUNS_OUT) as f:
            runs = json.load(f)
        print(f"保存済み run を読み込み: {os.path.basename(RUNS_OUT)} ({list(runs)})")
    else:
        t0 = time.time()
        entries = {}
        for qid in qids:
            e = webstyle[qid]
            docs = e.get("query2doc_docs_structured") or []
            dqs = e.get("decomposed_queries") or []
            entries[qid] = [(dq, doc, analyze_terms(dq, field="body"))
                            for dq, doc in zip(dqs, docs)
                            if doc and doc.get("body")]
        n_sub = sum(len(v) for v in entries.values())
        print(f"  Subquery {n_sub} 本を準備 ({time.time() - t0:.0f}s)")
        print(f"RETRIEVE_K={RETRIEVE_K} QUERY_REPEAT={QUERY_REPEAT} "
              f"SPAN_END={SPAN_END} SPAN_BOOST={SPAN_BOOST}")
        runs = build_searched_runs(qids, queries, entries)
        runs.update(load_free_runs(qids))
        with open(RUNS_OUT, "w") as f:
            json.dump(runs, f)
        print(f"run を保存: {os.path.basename(RUNS_OUT)} "
              f"({os.path.getsize(RUNS_OUT) / 1e6:.1f} MB)")

    out = {"n_queries": len(qids), "topk": TOPK, "retrieve_k": RETRIEVE_K,
           "query_repeat": QUERY_REPEAT, "span_end": SPAN_END,
           "span_boost": SPAN_BOOST, "levels": LEVELS, "qids": qids,
           "ceilings": {}, "summary": {}}

    for qname, qr in qrels.items():
        out["ceilings"][qname] = {
            f"level{lv}": {f"recall_{c}": ceiling(qr, qids, lv, c) for c in (100, 1000)}
            for lv in LEVELS}
        out["summary"][qname] = {}
        for lv in LEVELS:
            print(f"\n{'=' * 78}\n[{qname}] grade>={lv} を正解とみなす"
                  f"  (recall@100の上限={out['ceilings'][qname][f'level{lv}']['recall_100']:.4f})")
            out["summary"][qname][f"level{lv}"] = {}
            for method in SEARCHED + FREE:
                if method not in runs:
                    continue
                agg, n = score(runs[method], qr, lv)
                if agg is None:
                    continue
                out["summary"][qname][f"level{lv}"][method] = agg
                cols = "  ".join(f"{m}={agg[m]:.4f}" for m in BINARY_KEYS)
                nd = ("  " + "  ".join(f"{m}={agg[m]:.4f}" for m in NDCG_KEYS)
                      if lv == 1 else "")
                print(f"  {method:20s} n={n:3d}  {cols}{nd}")

    with open(RESULT_OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {os.path.basename(RESULT_OUT)}")


if __name__ == "__main__":
    main()
