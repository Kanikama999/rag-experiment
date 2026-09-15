"""
BM25Fを、構造化疑似文書（title/headings/bodyへ別々にルーティング）ではなく
フラット疑似文書（同じ生成内容をtitle+headings+bodyに連結した1本のテキスト）で測る。

背景:
  evaluate_bm25f_nopseudo.py は「疑似文書を使うかどうか」と「疑似文書を構造化して
  フィールドごとに振り分けるかどうか」の2つの軸を同時に消していた。しかし疑似文書
  そのものは有効な手法として使ってよく、問題視されているのは後者
  （構造化ルーティングが効果の由来を分かりにくくする）の方である。
  この実験ではその2軸を切り分ける：疑似文書の生成内容（LLM出力）はそのまま使うが、
  title生成テキストをtitleフィールドへ、headings生成テキストをheadingsフィールドへ、
  という個別ルーティングをやめ、title+headings+bodyを連結したフラットテキスト
  （webstyleファイルの query2doc_docs、構造化版 query2doc_docs_structured と
  生成内容は同一）を、long_pos_fielded/nopseudoと同じ規約で3フィールド共通のテキストとして
  match する。

  これで以下の3つが同一の基盤（weights=[2,1,1], b=[0,0.8,1.0], RETRIEVE_K=3000,
  QUERY_REPEAT=5, RRF k=60）の上で比較できる:
    nopseudo      : narrative×5 + Subquery                       （疑似文書なし）
    flatpseudo    : narrative×5 + Subquery + フラット疑似文書      （疑似文書あり・構造化なし）
    k1b3_a0等     : narrative×5 + Subquery + 構造化疑似文書        （疑似文書あり・構造化あり）
  nopseudo→flatpseudo の差が「疑似文書の内容そのものの効果」、
  flatpseudo→structured の差が「構造化ルーティングそのものの効果」。

測る条件（bm25f_prepare()を1回呼べば3条件とも同時に出せる）:
  norerank_flatpseudo : Stage 1のネイティブBM25ヒット順のみ（再スコアリングなし）
  k1b0.9_flatpseudo   : bm25f_score()でk1=0.9
  k1b3_flatpseudo     : bm25f_score()でk1=3.0（k1_length_sweepの最良点）

使い方:
    python3 evaluate_bm25f_flatpseudo.py [LIMIT]
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import bm25f_prepare, bm25f_score, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000
QUERY_REPEAT = 5
TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
B_TITLE, B_HEADINGS, B_BODY = 0.0, 0.8, 1.0
K1_VALUES = [0.9, 3.0]
METHODS = ["norerank_flatpseudo"] + [f"k1b{k1:g}_flatpseudo" for k1 in K1_VALUES]
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


def dq_flat_pairs(entry):
    """(Subquery, フラット疑似文書テキスト) のペア。構造化版と同じ行だけを対象にする
    （query2doc_docs_structured が空/bodyなしの行はnopseudo/k1b3_a0側でも除外されて
    いるため、比較対象を完全に揃えるために同じフィルタをかける）。"""
    return [(dq, flat) for dq, flat, doc in zip(
                entry["decomposed_queries"], entry["query2doc_docs"],
                entry["query2doc_docs_structured"])
            if doc and doc.get("body") and flat and flat.strip()]


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
          f"k1={K1_VALUES}（フラット疑似文書: 構造化せず3フィールド共通で使用）")

    runs = {m: {} for m in METHODS}
    t0 = time.time()
    for i, qid in enumerate(valid_qids, 1):
        q = queries[qid]
        repeated_q = " ".join([q] * QUERY_REPEAT)
        lists = {m: [] for m in METHODS}
        for dq, flat in dq_flat_pairs(webstyle[qid]):
            lt = f"{repeated_q} {dq} {flat}"
            raw = bm25f_prepare(lt, lt, lt,
                                 title_weight=TITLE_W, headings_weight=HEADINGS_W,
                                 body_weight=BODY_W,
                                 candidate_k=RETRIEVE_K, rerank_n=RETRIEVE_K)
            if raw is None:
                continue
            norerank_lst = [(d, 0.0) for d in raw["candidates"][:RETRIEVE_K]]
            if norerank_lst:
                lists["norerank_flatpseudo"].append(norerank_lst)
            for k1 in K1_VALUES:
                ranked = bm25f_score(raw, k=RETRIEVE_K,
                                     title_weight=TITLE_W, headings_weight=HEADINGS_W,
                                     body_weight=BODY_W,
                                     b_title=B_TITLE, b_headings=B_HEADINGS, b_body=B_BODY,
                                     bm25_k1=k1)
                if ranked:
                    lists[f"k1b{k1:g}_flatpseudo"].append(ranked)

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

    out_path = os.path.join(RAG_DIR, "bm25f_flatpseudo_result.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(valid_qids), "retrieve_k": RETRIEVE_K,
                   "weights": [TITLE_W, HEADINGS_W, BODY_W],
                   "b": [B_TITLE, B_HEADINGS, B_BODY], "k1_values": K1_VALUES,
                   "summary": summary}, f, ensure_ascii=False, indent=1)
    print(f"\n結果を保存: {out_path}")


if __name__ == "__main__":
    main()
