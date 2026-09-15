"""
BM25Fの「narrative×5版」— クエリ側の語の出現回数（query term frequency, qtf）を
スコアに反映したBM25Fを、既存のn=105条件でそのまま測り直す。

背景（EXPERIMENT_SUMMARY.md §3.3 / §8-3）:
retriever.py の bm25f_prepare() は title/headings/body の3つのクエリ文字列から
「重複を除いた語の集合」を作り、bm25f_score() は各語を1回ずつしか回さない。そのため
narrative×5（QUERY_REPEAT=5）の繰り返しがBM25Fのスコアリングに一切反映されていなかった。
一方 match クエリを使う他手法（baseline・フィールド線形和・champion）は、クエリ文字列内で
同じ語が5回出れば5つのBooleanQuery節ができ、スコア寄与が5倍になる。つまり
**BM25Fだけが narrative×5 の恩恵を受けていない**状態で「BM25Fは負ける」と報告していた。
このスクリプトはその非対称を取り除いた数値を出し、論文の比較表を差し替え可能にする。

条件（すべて既存の webstyle_bm25f_tuned_eval_summary_rep5.json と同一。n=105、
RETRIEVE_K=3000、QUERY_REPEAT=5、weights title=2/headings=1/body=1、
b_title=0.6/b_headings=0.2/b_body=0.2、k1=0.9、Subquery単位検索→RRF(k=60)）:

- bm25f_dedup   : 現行実装（qtf無視）。既存n=105行の再現。run間ノイズの確認も兼ねる。
- bm25f_qtf     : qtfを線形に掛ける版。Luceneが同一語の複数節を単純加算するのと同じ扱い。
                    score += qtf(t) * idf(t) * pseudo_tf/(k1 + pseudo_tf)
- bm25f_qtf_k3  : Robertson流のクエリ側飽和 ((k3+1)*qtf)/(k3+qtf) を掛ける版（k3=8）。
                    narrative×5が線形だと効きすぎる場合の対照。

3条件はすべて**同一の候補プール・同一の_mtermvectors結果**から計算する（bm25f_prepare()を
1回呼び、bm25f_score相当を3通り回す）。ネットワークI/Oは1条件ぶんしかかからず、
条件間の差は純粋にスコア計算式の違いだけになる。

qtfは「narrative×5 + Subquery + 疑似文書のtitle + headings + body」を1本に連結した
論理的なクエリ本文から数える（3つのフィールドクエリで共通prefixを3回数える不自然さを
避けるため。evaluate_bm25f_qtf.py と同じ定義）。

使い方:
    python3 evaluate_bm25f_narrative_rep5.py [SUBSET_STRIDE] [LIMIT]
    # SUBSET_STRIDE 省略時は1（全105トピック）。LIMIT はパイロット実行用の先頭N件。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter

import pytrec_eval

from retriever import (client, INDEX, bm25f_prepare, rrf_fuse)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000          # webstyle_bm25f_tuned_eval_summary_rep5.json と同じ候補プール
QUERY_REPEAT = 5
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
B_TITLE, B_HEADINGS, B_BODY = 0.6, 0.2, 0.2   # bm25f_field_b_search_result.json の最良点
BM25_K1 = 0.9
QTF_K3 = 8.0

METHODS = ["bm25f_dedup", "bm25f_qtf", "bm25f_qtf_k3"]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]

CKPT = os.path.join(RAG_DIR, "_cache_bm25f_narrative_rep5_runs.json")


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
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


def dq_pairs_structured(entry):
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"],
                                          entry["query2doc_docs_structured"])
            if doc and doc.get("body")]


def qtf_counts(logical_query):
    """論理クエリ本文をbodyのanalyzerでトークン化し、語ごとの出現回数を数える
    （analyze_terms()は重複除去してしまうので生トークン列から数える）。"""
    res = client.indices.analyze(index=INDEX, body={"field": "body", "text": logical_query})
    return Counter(t["token"] for t in res.get("tokens", []))


def score_variants(raw, qtf, k):
    """bm25f_prepare()の生データから3条件のランキングを同時に作る（ネットワークI/Oなし）。

    3条件はqtfの掛け方だけが違う:
      dedup  -> 1.0（現行実装。narrative×5が無視される）
      qtf    -> qtf(t)（線形。Luceneの複数節加算と等価）
      qtf_k3 -> (k3+1)*qtf/(k3+qtf)（Robertson流のクエリ側飽和）
    """
    if raw is None:
        return {m: [] for m in METHODS}
    if not raw["query_terms"]:
        flat = [(d, 0.0) for d in raw["candidates"][:k]]
        return {m: list(flat) for m in METHODS}

    idf, avgdl, docs = raw["idf"], raw["avgdl"], raw["docs"]
    weights = {"title": TITLE_W, "headings": HEADINGS_W, "body": BODY_W}
    bs = {"title": B_TITLE, "headings": B_HEADINGS, "body": B_BODY}

    # 語ごとの倍率は文書に依らないので先に作る
    mult = {}
    for t in raw["query_terms"]:
        n = qtf.get(t, 1)
        mult[t] = (1.0, float(n), (QTF_K3 + 1.0) * n / (QTF_K3 + n))

    scores = {m: {} for m in METHODS}
    for docid, doc in docs.items():
        field_terms, field_len = doc["field_terms"], doc["field_len"]
        acc = [0.0, 0.0, 0.0]
        for t in raw["query_terms"]:
            t_idf = idf.get(t, 0.0)
            if t_idf <= 0:
                continue
            pseudo_tf = 0.0
            for field in ("title", "headings", "body"):
                info = field_terms[field].get(t)
                if not info:
                    continue
                dl = field_len[field]
                B = (1 - bs[field]) + bs[field] * (dl / avgdl[field]) if avgdl[field] else 1.0
                pseudo_tf += weights[field] * info["term_freq"] / B
            if pseudo_tf > 0:
                contrib = t_idf * pseudo_tf / (BM25_K1 + pseudo_tf)
                m = mult[t]
                acc[0] += contrib * m[0]
                acc[1] += contrib * m[1]
                acc[2] += contrib * m[2]
        for i, name in enumerate(METHODS):
            scores[name][docid] = acc[i]

    return {name: sorted(scores[name].items(), key=lambda x: -x[1])[:k] for name in METHODS}


def build_runs(qids, queries, webstyle):
    runs = {m: {} for m in METHODS}
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            cached = json.load(f)
        if cached.get("config") == run_config():
            runs = {m: cached["runs"].get(m, {}) for m in METHODS}
            print(f"チェックポイント復帰: {len(runs[METHODS[0]])} トピック済み")
        else:
            print("チェックポイントは条件が違うため破棄")

    t0, done = time.time(), 0
    todo = [q for q in qids if q not in runs[METHODS[0]]]
    for qid in todo:
        q = queries[qid]
        repeated_q = " ".join([q] * QUERY_REPEAT)
        lists = {m: [] for m in METHODS}
        for dq, doc in dq_pairs_structured(webstyle[qid]):
            headings_txt = " ".join(doc["headings"])
            title_q = f"{repeated_q} {dq} {doc['title']}"
            headings_q = f"{repeated_q} {dq} {headings_txt}"
            body_q = f"{repeated_q} {dq} {doc['body']}"
            logical_q = f"{repeated_q} {dq} {doc['title']} {headings_txt} {doc['body']}"

            raw = bm25f_prepare(title_q, headings_q, body_q,
                                 title_weight=TITLE_W, headings_weight=HEADINGS_W,
                                 body_weight=BODY_W,
                                 candidate_k=RETRIEVE_K, rerank_n=RETRIEVE_K)
            ranked = score_variants(raw, qtf_counts(logical_q), RETRIEVE_K)
            for m in METHODS:
                if ranked[m]:
                    lists[m].append(ranked[m])

        for m in METHODS:
            fused = rrf_fuse(lists[m], top_n=TOPK) if lists[m] else []
            runs[m][qid] = {d: float(s) for d, s in fused}

        done += 1
        el = time.time() - t0
        eta = el / done * (len(todo) - done)
        print(f"  {done}/{len(todo)}  qid={qid}  経過{el/60:.1f}分  残り約{eta/60:.0f}分", flush=True)
        with open(CKPT, "w") as f:
            json.dump({"config": run_config(), "runs": runs}, f)
    return runs


def run_config():
    return {"retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT, "topk": TOPK,
            "weights": [TITLE_W, HEADINGS_W, BODY_W], "b": [B_TITLE, B_HEADINGS, B_BODY],
            "k1": BM25_K1, "k3": QTF_K3, "stride": SUBSET_STRIDE, "limit": LIMIT}


def evaluate(run, qrels, qids, label):
    target = [q for q in qids if q in qrels and q in run]
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = ev.evaluate({q: run[q] for q in target})
    n = len(results)
    if n == 0:
        print(f"[{label}] このqrelsに含まれるトピックが0件（スキップ）")
        return None
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    agg["n"] = n
    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}

    valid_qids = sorted(qid for qid in webstyle
                        if qid in queries and webstyle[qid].get("decomposed_queries"))
    search_qids = valid_qids[::SUBSET_STRIDE]
    if LIMIT:
        search_qids = search_qids[:LIMIT]
    print(f"全トピック: {len(valid_qids)}   対象: {len(search_qids)}   "
          f"RETRIEVE_K={RETRIEVE_K}   QUERY_REPEAT={QUERY_REPEAT}")

    runs = build_runs(search_qids, queries, webstyle)

    summary = {}
    for name in QREL_SETS:
        print(f"\n[{name}]")
        for m in METHODS:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], search_qids, f"{m}/{name}")

    suffix = "" if SUBSET_STRIDE == 1 and not LIMIT else f"_stride{SUBSET_STRIDE}_n{len(search_qids)}"
    out = os.path.join(RAG_DIR, f"bm25f_narrative_rep5_result{suffix}.json")
    with open(out, "w") as f:
        json.dump({"n_queries": len(search_qids), "qids": search_qids,
                    "config": run_config(), "summary": summary}, f, indent=2)
    print(f"\n-> {out}")

    run_out = os.path.join(RAG_DIR, f"bm25f_narrative_rep5_runs{suffix}.json")
    with open(run_out, "w") as f:
        json.dump(runs, f)
    print(f"-> {run_out}")


if __name__ == "__main__":
    main()
