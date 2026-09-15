"""
「TFは残すが、出現の1回1回に減衰する重みを与えて足し上げる」を検証する。

背景:
evaluate_k1_decay_sweep.py（k1定数）と evaluate_bm25f_k1_length_sweep.py（k1を文書長の
関数に）は、どちらも「TFの効き方を弱めると一貫して悪化する」という結果だった。
指摘: TFを捨てる／弱めるのではなく、**出現回数nを i=1..n の個々の出現に分解し、
各出現に減衰する重みf(i)を与えて合計する**べきではないか。

    score_t(D) = idf(t) × Σ_{i=1}^{n} f(i)      (n = tf(t,D))

標準BM25の k1 は「合計の出現回数」に対して1つの飽和曲線
（tf(k1+1)/(tf+k1B)）を当てはめる。これは実は暗黙に「i番目の出現の限界寄与」
（g(i)-g(i-1)、常に減少列）を積分した形になっているが、**カーブの形が固定**されている
（一定のk1で決まるMichaelis-Menten型の一形）。f(i)を陽に選べば、k1では表現できない
減衰パターン（無限に伸び続けるが対数的に遅い、など）を試せる。

f(i) = i^(-p) というべき乗族を使う:
    p=0   : f(i)=1 で全て等しい → Σf(i)=tf（単純加算、減衰なし。k1→∞の極限に相当）
    p=1   : 調和級数的減衰。Σは対数的に増え続け、上限を持たない
    p>1   : Σ_{i=1}^{n} i^(-p) は n→∞ で有限値に収束する（一般化調和数）
    p→∞  : f(1)=1, f(i>1)≈0 に近づく（案1の binary coverage に接近）

フィールドごとに Σf(i) を「減衰後のtf」として使い、bm25f_score() と同じ構造で
フィールド長正規化・フィールド間重み付け・idf適用まで行う（k1による最終飽和は
不要——減衰は既に出現ごとに起きているため）:

    pseudo_tf = Σ_field weight_field × DecayedTF_field(t) / B_field
    contrib   = idf(t) × pseudo_tf

qtf（クエリ側の重み）は既に線形が最良と判明しているのでlinear固定。
b は avgdlバグ修正後の正しい値（0.0/0.8/1.0）を使う。

pの値ごとに Σ_{i=1}^{n} i^(-p) を都度計算するだけで追加のI/Oは不要
（bm25f_prepare()の生データ1回ぶんから全p値のスコアを計算する）。

使い方:
    python3 evaluate_bm25f_occurrence_decay_sweep.py [SUBSET_STRIDE] [LIMIT]
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

import pytrec_eval

from retriever import client, INDEX, bm25f_prepare, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
CKPT = os.path.join(RAG_DIR, "bm25f_occurrence_decay_ckpt.json")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000
QUERY_REPEAT = 5
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
B_TITLE, B_HEADINGS, B_BODY = 0.0, 0.8, 1.0   # avgdlバグ修正後の正しい再探索値

P_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
METHODS = [f"p{p:g}" for p in P_VALUES]
MAX_TF_CACHE = 500   # Σi^(-p) の累積和をtf=1..500まで前計算してキャッシュ

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


def qtf_counts(logical_query):
    res = client.indices.analyze(index=INDEX, body={"field": "body", "text": logical_query})
    return Counter(t["token"] for t in res.get("tokens", []))


# Σ_{i=1}^{n} i^(-p) の累積和テーブル（p値ごとに1回だけ計算、tf=1..MAX_TF_CACHEまで）
_CUMSUM_CACHE = {}


def cumsum_table(p):
    if p not in _CUMSUM_CACHE:
        table = [0.0]
        s = 0.0
        for i in range(1, MAX_TF_CACHE + 1):
            s += i ** (-p)
            table.append(s)
        _CUMSUM_CACHE[p] = table
    return _CUMSUM_CACHE[p]


def decayed_tf(n, p):
    """Σ_{i=1}^{n} i^(-p)。nがキャッシュ範囲を超えたら末尾の値で近似
    （p>0なら増分は無視できるほど小さいので十分実用的、p=0なら末尾+はみ出し分を線形加算）。"""
    table = cumsum_table(p)
    if n <= 0:
        return 0.0
    if n <= MAX_TF_CACHE:
        return table[n]
    if p == 0:
        return table[MAX_TF_CACHE] + (n - MAX_TF_CACHE) * 1.0
    return table[MAX_TF_CACHE]   # p>0の裾は無視できるほど小さい


def score_variants(raw, qtf, k):
    """bm25f_prepare()の生データから全p値のランキングを同時に作る（I/Oなし）。"""
    if raw is None:
        return {m: [] for m in METHODS}
    if not raw["query_terms"]:
        flat = [(d, 0.0) for d in raw["candidates"][:k]]
        return {m: list(flat) for m in METHODS}

    idf, avgdl, docs = raw["idf"], raw["avgdl"], raw["docs"]
    weights = {"title": TITLE_W, "headings": HEADINGS_W, "body": BODY_W}
    bs = {"title": B_TITLE, "headings": B_HEADINGS, "body": B_BODY}

    n_cond = len(METHODS)
    scores = {m: {} for m in METHODS}
    for docid, doc in docs.items():
        field_terms, field_len = doc["field_terms"], doc["field_len"]
        acc = [0.0] * n_cond
        for t in raw["query_terms"]:
            t_idf = idf.get(t, 0.0)
            if t_idf <= 0:
                continue
            n_qtf = float(qtf.get(t, 1))
            pseudo_tf_per_p = [0.0] * n_cond
            for field in ("title", "headings", "body"):
                info = field_terms[field].get(t)
                if not info:
                    continue
                dl = field_len[field]
                B = (1 - bs[field]) + bs[field] * (dl / avgdl[field]) if avgdl[field] else 1.0
                raw_tf = info["term_freq"]
                for i, p in enumerate(P_VALUES):
                    pseudo_tf_per_p[i] += weights[field] * decayed_tf(raw_tf, p) / B
            for i in range(n_cond):
                if pseudo_tf_per_p[i] > 0:
                    acc[i] += t_idf * pseudo_tf_per_p[i] * n_qtf
        for i, name in enumerate(METHODS):
            scores[name][docid] = acc[i]

    return {name: sorted(scores[name].items(), key=lambda x: -x[1])[:k] for name in METHODS}


def run_config():
    return {"retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT, "topk": TOPK,
            "weights": [TITLE_W, HEADINGS_W, BODY_W], "b": [B_TITLE, B_HEADINGS, B_BODY],
            "p_values": P_VALUES, "stride": SUBSET_STRIDE, "limit": LIMIT}


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

    todo = [q for q in qids if q not in runs[METHODS[0]]]
    t0 = time.time()
    for n, qid in enumerate(todo, 1):
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

        el = time.time() - t0
        eta = el / n * (len(todo) - n)
        print(f"  {n}/{len(todo)}  qid={qid}  経過{el/60:.1f}分  残り約{eta/60:.0f}分", flush=True)
        if n % 5 == 0 or n == len(todo):
            with open(CKPT, "w") as f:
                json.dump({"config": run_config(), "runs": runs}, f)
    return runs


def evaluate(run, qrels, qids, label):
    target = [q for q in qids if q in qrels and q in run]
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = ev.evaluate({q: run[q] for q in target})
    n = len(results)
    if n == 0:
        return None
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    agg["n"] = n
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
    print(f"全トピック: {len(valid_qids)}   対象: {len(search_qids)}   candidate_k={RETRIEVE_K}"
          f"   p値: {P_VALUES}")

    runs = build_runs(search_qids, queries, webstyle)

    summary = {}
    for name in QREL_SETS:
        for m in METHODS:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], search_qids, f"{m}/{name}")

    suffix = "" if SUBSET_STRIDE == 1 and not LIMIT else f"_stride{SUBSET_STRIDE}_n{len(search_qids)}"
    out = os.path.join(RAG_DIR, f"bm25f_occurrence_decay_result{suffix}.json")
    with open(out, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "config": run_config(), "summary": summary},
                   f, indent=2)
    print(f"\n-> {out}")

    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print(f"{'p':>6}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for m, p in zip(METHODS, P_VALUES):
            a = summary[qs][m]
            if a: print(f"{p:>6}"+"".join(f"{a[k]:>10.4f}" for k in METRIC_KEYS))


if __name__ == "__main__":
    main()
