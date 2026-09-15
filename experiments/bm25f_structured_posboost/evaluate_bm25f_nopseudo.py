"""
BM25F（bm25f_prepare/bm25f_score, k1_length_sweepと同じweights/b）を、
LLM生成の構造化疑似文書を使わない「素の」クエリで測る。

背景:
  これまでのBM25F実験（evaluate_bm25f_k1_length_sweep.py 等）は、champion系で
  既に確立していたクエリ構築（narrative×5 + Subquery + LLM生成のtitle/headings/body）を
  そのまま流用していた。しかしBM25Fは Robertson/Zaragoza/Taylor (2004) の
  「文書側の複数フィールドをどう合算してスコアにするか」だけを定める式であり、
  クエリが疑似文書かどうかとは無関係（bm25f_prepareの引数title_text/headings_text/
  body_textは「各フィールドにマッチさせたい任意のテキスト」を受け取るだけ）。
  疑似文書による改善とBM25Fによる改善が一度も切り分けられていなかったため、
  「本来の」BM25F単体の効果を、疑似文書という交絡を抜いて測る。

クエリ構築:
  evaluate_oracle_current_pipeline.py の long_pos_fielded 条件
  （"LLMもオラクルも使わない対照"）と同じ規約を採用する:
    narrative×5 + Subquery のみを、title/headings/bodyの3フィールド全てへ
    同じテキストとしてmatchする（lt = narrative×5 + Subquery、fielded(lt, lt, lt)）。
  疑似文書のtitle/headings/body別テキストは一切使わない。

測る条件（bm25f_prepare()を1回呼べば3条件とも同時に出せる。追加I/Oなし）:
  norerank_nopseudo : bm25f_prepare()の候補プール（Stage 1、ネイティブBM25の
                       ヒット順）をそのまま使う。evaluate_bm25f_norerank.pyの
                       疑似文書なし版に相当。
  k1b0.9_nopseudo   : bm25f_score()でk1=0.9（k1_length_sweepのk1_base最小値）
  k1b3_nopseudo     : bm25f_score()でk1=3.0（k1_length_sweepの最良点）

weights=[2,1,1]・b=[0,0.8,1.0]（avgdlバグ修正後の値）・RETRIEVE_K=3000・
QUERY_REPEAT=5・RRF(k=60)はk1_length_sweep/norerankと完全に揃えてある。

比較の読み方:
  norerank_nopseudo vs k1b3_nopseudo の差 → 疑似文書なしでも BM25F rerank は効くか
  k1b3_nopseudo vs k1b3_a0（疑似文書あり） の差 → BM25Fの改善は疑似文書に依存するか
  norerank_nopseudo vs norerank（疑似文書あり） の差 → 疑似文書自体の効果（rerank抜き）

使い方:
    python3 evaluate_bm25f_nopseudo.py [LIMIT]
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import bm25f_prepare, bm25f_score, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000
QUERY_REPEAT = 5
TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
B_TITLE, B_HEADINGS, B_BODY = 0.0, 0.8, 1.0
K1_VALUES = [0.9, 3.0]
METHODS = ["norerank_nopseudo"] + [f"k1b{k1:g}_nopseudo" for k1 in K1_VALUES]
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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


def dq_pairs_structured(entry):
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"],
                                         entry["query2doc_docs_structured"])
            if doc and doc.get("body")]


def evaluate(run, qrels, qids, label):
    target = [q for q in qids if q in qrels and q in run]
    if not target:
        print(f"[{label}] 採点対象なし")
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = ev.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    agg["n"] = n
    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt"))
             for n in QREL_SETS}

    # 疑似文書ファイルはSubquery一覧（decomposed_queries）を得るためだけに使う。
    # 疑似文書の中身（query2doc_docs_structured）自体はクエリ構築に一切使わない。
    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
        and any(d and d.get("body") for d in webstyle[qid].get("query2doc_docs_structured", []))
    )
    if LIMIT:
        valid_qids = valid_qids[:LIMIT]
    print(f"対象 {len(valid_qids)} クエリ")
    print(f"RETRIEVE_K={RETRIEVE_K} QUERY_REPEAT={QUERY_REPEAT} "
          f"weights=({TITLE_W},{HEADINGS_W},{BODY_W}) b=({B_TITLE},{B_HEADINGS},{B_BODY}) "
          f"k1={K1_VALUES}（疑似文書なし: narrative×5+Subqueryのみを3フィールド共通で使用）")

    runs = {m: {} for m in METHODS}
    t0 = time.time()
    for i, qid in enumerate(valid_qids, 1):
        q = queries[qid]
        repeated_q = " ".join([q] * QUERY_REPEAT)
        lists = {m: [] for m in METHODS}
        # Subquery一覧だけをwebstyleから取り出す（doc本体は使わない）
        dqs = [dq for dq, doc in dq_pairs_structured(webstyle[qid])]
        for dq in dqs:
            lt = f"{repeated_q} {dq}"   # long_pos_fielded と同じ規約
            raw = bm25f_prepare(lt, lt, lt,
                                 title_weight=TITLE_W, headings_weight=HEADINGS_W,
                                 body_weight=BODY_W,
                                 candidate_k=RETRIEVE_K, rerank_n=RETRIEVE_K)
            if raw is None:
                continue
            # Stage 1のみ（ネイティブBM25ヒット順、再スコアリングなし）
            norerank_lst = [(d, 0.0) for d in raw["candidates"][:RETRIEVE_K]]
            if norerank_lst:
                lists["norerank_nopseudo"].append(norerank_lst)
            for k1 in K1_VALUES:
                ranked = bm25f_score(raw, k=RETRIEVE_K,
                                     title_weight=TITLE_W, headings_weight=HEADINGS_W,
                                     body_weight=BODY_W,
                                     b_title=B_TITLE, b_headings=B_HEADINGS, b_body=B_BODY,
                                     bm25_k1=k1)
                if ranked:
                    lists[f"k1b{k1:g}_nopseudo"].append(ranked)

        for m in METHODS:
            fused = rrf_fuse(lists[m], top_n=TOPK) if lists[m] else []
            runs[m][qid] = {d: float(s) for d, s in fused}

        if i % 10 == 0 or i == len(valid_qids):
            el = time.time() - t0
            eta = el / i * (len(valid_qids) - i)
            print(f"\r  {i}/{len(valid_qids)}  経過{el/60:.1f}分  残り約{eta/60:.0f}分",
                  end="", flush=True)
    print()

    summary = {}
    for m in METHODS:
        for name in QREL_SETS:
            agg = evaluate(runs[m], qrels[name], valid_qids, f"{m} / {name}")
            if agg:
                summary.setdefault(name, {})[m] = agg

    out_path = os.path.join(RAG_DIR, "bm25f_nopseudo_result.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(valid_qids), "retrieve_k": RETRIEVE_K,
                   "weights": [TITLE_W, HEADINGS_W, BODY_W],
                   "b": [B_TITLE, B_HEADINGS, B_BODY], "k1_values": K1_VALUES,
                   "summary": summary}, f, ensure_ascii=False, indent=1)
    print(f"\n結果を保存: {out_path}")


if __name__ == "__main__":
    main()
