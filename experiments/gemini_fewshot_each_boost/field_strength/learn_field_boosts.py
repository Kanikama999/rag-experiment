"""
title/headings/bodyそれぞれのBM25スコアを特徴量にした線形回帰で、フィールドブースト
（title^?+headings^?+body^?の?の部分）を手作業で決め打ちせず、qrelsから推定する。

手順:
1. 各トピックのSubquery毎に、narrative×QUERY_REPEAT + Subquery +
   生成されたtitle/headings/bodyそれぞれをtitleフィールド/headingsフィールド/body
   フィールドへ単独match（bm25_title/bm25_headings/bm25_body）し、上位POOL_K件を取る。
2. 3つの結果の和集合を候補プールとし、候補文書ごとに
   (title_score, headings_score, body_score) を特徴量、qrels（consensus）の
   relevance grade（未収録なら0）を目的変数にした学習データを作る。
   スコアが無い（＝その検索の上位POOL_K件に入らなかった）フィールドは0とする。
3. 最小二乗法（numpy.linalg.lstsq、切片あり）で y ≈ w0 + w1*title + w2*headings + w3*body
   を推定する。切片はランキングに影響しないので無視し、w1/w2/w3の比率を新しいブースト値
   の候補として使う。

使い方:
    python learn_field_boosts.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
from retriever import bm25_body, bm25_title, bm25_headings

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PARENT_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(PARENT_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

POOL_K = 1000
QUERY_REPEAT = 5
OUT_FILE = os.path.join(RAG_DIR, "field_boost_regression.json")


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


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
    )
    print(f"対象トピック数: {len(valid_qids)}")

    rows_title, rows_headings, rows_body, rows_y = [], [], [], []
    n_dq = 0
    t0 = time.time()
    for i, qid in enumerate(valid_qids, 1):
        q = queries[qid]
        entry = webstyle[qid]
        pairs = dq_pairs_structured(entry)
        repeated_q = " ".join([q] * QUERY_REPEAT)
        qrel_for_topic = qrels.get(qid, {})

        for dq, doc in pairs:
            n_dq += 1
            title_q = f"{repeated_q} {dq} {doc['title']}"
            headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
            body_q = f"{repeated_q} {dq} {doc['body']}"

            title_hits = dict(bm25_title(title_q, k=POOL_K))
            headings_hits = dict(bm25_headings(headings_q, k=POOL_K))
            body_hits = dict(bm25_body(body_q, k=POOL_K))

            candidates = set(title_hits) | set(headings_hits) | set(body_hits)
            for docid in candidates:
                rows_title.append(title_hits.get(docid, 0.0))
                rows_headings.append(headings_hits.get(docid, 0.0))
                rows_body.append(body_hits.get(docid, 0.0))
                rows_y.append(qrel_for_topic.get(docid, 0))

        if i % 10 == 0 or i == len(valid_qids):
            print(f"\r  {i}/{len(valid_qids)} topics, {n_dq} DQ, "
                  f"{len(rows_y)} rows ({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    X = np.column_stack([
        np.ones(len(rows_y)),
        np.array(rows_title),
        np.array(rows_headings),
        np.array(rows_body),
    ])
    y = np.array(rows_y, dtype=float)

    print(f"\n学習データ: {X.shape[0]} 行 (topics={len(valid_qids)}, DQ={n_dq})")
    print(f"  y の分布: 0={np.sum(y == 0)}, >0={np.sum(y > 0)}, 平均relevance={y.mean():.4f}")

    coef, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    intercept, w_title, w_headings, w_body = coef
    print("\n=== 回帰係数（切片はランキングに無関係なので無視） ===")
    print(f"  intercept     = {intercept:.6f}")
    print(f"  w_title       = {w_title:.6f}")
    print(f"  w_headings    = {w_headings:.6f}")
    print(f"  w_body        = {w_body:.6f}")

    # body基準で正規化した比率（bm25_fieldedのboost引数に使う想定）
    if abs(w_body) > 1e-12:
        ratio = (w_title / w_body, w_headings / w_body, 1.0)
    else:
        ratio = (w_title, w_headings, w_body)
    print(f"\nbody=1に正規化した比率: title={ratio[0]:.4f}  headings={ratio[1]:.4f}  body={ratio[2]:.4f}")

    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"R^2 (参考、あくまで線形回帰としての当てはまり): {r2:.4f}")

    with open(OUT_FILE, "w") as f:
        json.dump({
            "n_rows": int(X.shape[0]),
            "n_topics": len(valid_qids),
            "n_dq": n_dq,
            "coefficients": {"intercept": float(intercept), "title": float(w_title),
                              "headings": float(w_headings), "body": float(w_body)},
            "ratio_body1": {"title": float(ratio[0]), "headings": float(ratio[1]), "body": float(ratio[2])},
            "r2": float(r2),
        }, f, indent=2)
    print(f"\n-> {OUT_FILE}")


if __name__ == "__main__":
    main()
