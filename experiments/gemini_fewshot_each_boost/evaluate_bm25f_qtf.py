"""
BM25Fのスコア計算がクエリ側の語の出現回数（query term frequency, qtf）を無視している問題を
修正した版と、現行版のA/B比較。

問題: retriever.pyのbm25f_prepare()はtitle/headings/bodyの3つのクエリ文字列から
「重複を除いた語の集合」を作り、bm25f_score()は各語を1回ずつしか回さない。そのため
narrative×5（QUERY_REPEAT=5）の繰り返しがBM25Fのスコアリングに一切反映されない。
一方でmatchクエリを使う他手法（baseline・個別フィールド検索・提案手法）は、
クエリ文字列内で同じ語が5回出れば5つのBooleanQuery節ができてスコアが5倍寄与する。
つまりBM25Fだけがnarrative×5の恩恵を受けられておらず、比較が非対称だった。

修正: クエリ側の出現回数qtf(t)を数え、各語のスコア寄与に線形に掛ける
（Lucene が同一語の複数節を単純加算するのと同じ扱い）。
    score += qtf(t) * idf(t) * pseudo_tf/(k1 + pseudo_tf)
qtfは「narrative×5 + Subquery + 疑似文書のtitle + headings + body」を1本に連結した
論理的なクエリ本文から数える（共通prefixを3回数える不自然さを避けるため）。

条件:
- fielded_linear : 個別フィールド検索（線形和、title=2/headings=1/body=1）参照用
- bm25f_dedup    : 現行のBM25F（クエリ側の重複除去、qtf無視）
- bm25f_qtf      : qtfを反映したBM25F（本スクリプトの修正版）

35トピックのサブセット、candidate_k=1000。

使い方:
    python evaluate_bm25f_qtf.py
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter

import pytrec_eval

from retriever import (client, INDEX, bm25_fielded, rrf_fuse, analyze_terms,
                        _bm25f_avgdl, _bm25f_total_docs, _bm25f_combined_df)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5

TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
B_TITLE, B_HEADINGS, B_BODY = 0.6, 0.2, 0.2   # bm25f_field_b_search_resultの最良点
BM25_K1 = 0.9

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


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


def bm25f_run(title_text, headings_text, body_text, logical_query, k, use_qtf):
    """bm25f_prepare + bm25f_score 相当。use_qtf=Trueならクエリ側の出現回数を反映する。"""
    should = []
    for field, text, w in (("title", title_text, TITLE_W),
                            ("headings", headings_text, HEADINGS_W),
                            ("body", body_text, BODY_W)):
        if text and text.strip():
            should.append({"match": {field: {"query": text, "boost": w}}})
    if not should:
        return []
    res = client.search(index=INDEX, body={
        "size": k, "_source": False, "query": {"bool": {"should": should}}})
    candidates = [h["_id"] for h in res["hits"]["hits"]]
    if not candidates:
        return []

    # クエリ語（use_qtf=Trueなら出現回数付き、Falseなら重複除去して1回ずつ）
    # analyze_terms()は重複除去してしまうので、qtfはanalyze APIの生トークン列から数える
    qtf = None
    if use_qtf:
        res_an = client.indices.analyze(index=INDEX, body={"field": "body", "text": logical_query})
        qtf = Counter(t["token"] for t in res_an.get("tokens", []))
    query_terms, seen = [], set()
    for field, text in (("title", title_text), ("headings", headings_text), ("body", body_text)):
        if not text or not text.strip():
            continue
        for t in analyze_terms(text, field=field):
            if t not in seen:
                seen.add(t)
                query_terms.append(t)
    if not query_terms:
        return [(d, 0.0) for d in candidates[:k]]

    avgdl = {f: _bm25f_avgdl(f) for f in ("title", "headings", "body")}
    N = _bm25f_total_docs()
    idf = {}
    for t in query_terms:
        df = _bm25f_combined_df(t)
        idf[t] = math.log(1 + (N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0

    tv_docs = client.mtermvectors(index=INDEX, body={
        "ids": candidates,
        "parameters": {"fields": ["title", "headings", "body"],
                        "term_statistics": False, "field_statistics": False,
                        "positions": False, "offsets": False, "payloads": False}})

    weights = {"title": TITLE_W, "headings": HEADINGS_W, "body": BODY_W}
    bs = {"title": B_TITLE, "headings": B_HEADINGS, "body": B_BODY}
    scores = {}
    for doc in tv_docs.get("docs", []):
        docid = doc.get("_id")
        tvs = doc.get("term_vectors", {})
        field_terms, field_len = {}, {}
        for field in ("title", "headings", "body"):
            ti = tvs.get(field, {}).get("terms", {})
            field_terms[field] = ti
            field_len[field] = sum(v["term_freq"] for v in ti.values())
        score = 0.0
        for t in query_terms:
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
                score += contrib * (qtf.get(t, 1) if use_qtf else 1)
        scores[docid] = score
    return sorted(scores.items(), key=lambda x: -x[1])[:k]


def build_run(method, qids, queries, webstyle):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        pairs = dq_pairs_structured(webstyle[qid])
        repeated_q = " ".join([q] * QUERY_REPEAT)
        lists = []
        for dq, doc in pairs:
            headings_txt = " ".join(doc["headings"])
            title_q = f"{repeated_q} {dq} {doc['title']}"
            headings_q = f"{repeated_q} {dq} {headings_txt}"
            body_q = f"{repeated_q} {dq} {doc['body']}"
            if method == "fielded_linear":
                lists.append(bm25_fielded(title_q, headings_q, body_q, k=RETRIEVE_K,
                                           title_boost=TITLE_W, headings_boost=HEADINGS_W,
                                           body_boost=BODY_W))
            else:
                logical_q = f"{repeated_q} {dq} {doc['title']} {headings_txt} {doc['body']}"
                lists.append(bm25f_run(title_q, headings_q, body_q, logical_q,
                                        RETRIEVE_K, use_qtf=(method == "bm25f_qtf")))
        lists = [lst for lst in lists if lst]
        run[qid] = {d: float(s) for d, s in (rrf_fuse(lists, top_n=TOPK) if lists else [])}
        if i % 5 == 0 or i == len(qids):
            print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return run


def evaluate(run, qrels, qids, label):
    target = [q for q in qids if q in qrels]
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = ev.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
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
    search_qids = valid_qids[::3]
    print(f"全トピック: {len(valid_qids)}   サブセット: {len(search_qids)}   candidate_k={RETRIEVE_K}")

    methods = ["bm25f_qtf"]  # dedup版・線形和版は前回の実行で測定済み
    runs = {m: build_run(m, search_qids, queries, webstyle) for m in methods}

    summary = {}
    for name in QREL_SETS:
        print(f"\n[{name}]")
        for m in methods:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], search_qids, f"{m}/{name}")

    out = os.path.join(RAG_DIR, "bm25f_qtf_compare_result.json")
    with open(out, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "summary": summary}, f, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
