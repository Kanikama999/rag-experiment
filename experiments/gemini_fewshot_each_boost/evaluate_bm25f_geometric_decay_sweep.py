"""
「出現の1回1回に減衰する重みを与えて足し上げる」を、g(i)=べき乗族ではなく
**幾何減衰（等比）族**で検証する。

背景:
evaluate_bm25f_occurrence_decay_sweep.py はべき乗族 g(i)=i^(-p) を試し、両qrelsで
p=0.25付近にnDCG@10の山があることを確認した。ただし絶対値は k1定数を上げる実験
（k1=3.0でconsensus nDCG@10=0.5510）に遠く及ばなかった（p=0.25でも0.2457）。
原因は、べき乗族が p<=1 では **上限を持たず無限に伸び続ける**こと。特に p=0
（減衰なし）はBM25の飽和構造（分母にk1+pseudo_tfを持つ）が一切無い純粋な線形和で、
1語が極端に多く出る文書（このプロジェクトで繰り返し問題にしてきた「長い文書の水増し」）
にそのままやられる。

指摘: べき乗族ではなく**幾何減衰**を試すべき。

    g(i) = r^(i-1)      (0 < r < 1)
    Σ_{i=1}^{n} g(i) = (1 - r^n) / (1 - r)

べき乗族と違い、**どんなrでも必ず有限の上限 1/(1-r) に収束する**（BM25自身の飽和構造
と同じく「等比的に上限へ近づく」形）。rを1に近づけるほど上限が大きくなり「ほぼ減衰
しない」に近づく——k1を大きくする実験と同じ方向に繋がる別カーブとして比較できる。

r のスイープ（急な頭打ち→ほぼ減衰なしまで広く取る）:
    r ∈ {0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999}
    r=0.3  → 上限 1.43（案1のbinary coverageに近い急な頭打ち）
    r=0.999→ 上限 1000（現実的なtf範囲ではほぼ線形和に近い）

構造はevaluate_bm25f_occurrence_decay_sweep.pyと同一（bm25f_prepare 1回の生データから
全r値のスコアを計算、qtfはlinear固定、b=0.0/0.8/1.0、逐次処理でOOM対策）。

使い方:
    python3 evaluate_bm25f_geometric_decay_sweep.py [SUBSET_STRIDE] [LIMIT]
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
CKPT = os.path.join(RAG_DIR, "bm25f_geometric_decay_ckpt.json")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000
QUERY_REPEAT = 5
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
B_TITLE, B_HEADINGS, B_BODY = 0.0, 0.8, 1.0   # avgdlバグ修正後の正しい再探索値

R_VALUES = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
METHODS = [f"r{r:g}" for r in R_VALUES]

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


def geo_decayed_tf(n, r):
    """Σ_{i=1}^{n} r^(i-1) = (1-r^n)/(1-r)。閉じた式なので都度計算で十分軽い。"""
    if n <= 0:
        return 0.0
    return (1.0 - r ** n) / (1.0 - r)


def score_variants(raw, qtf, k):
    """bm25f_prepare()の生データから全r値のランキングを同時に作る（I/Oなし）。"""
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
            pseudo_tf_per_r = [0.0] * n_cond
            for field in ("title", "headings", "body"):
                info = field_terms[field].get(t)
                if not info:
                    continue
                dl = field_len[field]
                B = (1 - bs[field]) + bs[field] * (dl / avgdl[field]) if avgdl[field] else 1.0
                raw_tf = info["term_freq"]
                for i, r in enumerate(R_VALUES):
                    pseudo_tf_per_r[i] += weights[field] * geo_decayed_tf(raw_tf, r) / B
            for i in range(n_cond):
                if pseudo_tf_per_r[i] > 0:
                    acc[i] += t_idf * pseudo_tf_per_r[i] * n_qtf
        for i, name in enumerate(METHODS):
            scores[name][docid] = acc[i]

    return {name: sorted(scores[name].items(), key=lambda x: -x[1])[:k] for name in METHODS}


def run_config():
    return {"retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT, "topk": TOPK,
            "weights": [TITLE_W, HEADINGS_W, BODY_W], "b": [B_TITLE, B_HEADINGS, B_BODY],
            "r_values": R_VALUES, "stride": SUBSET_STRIDE, "limit": LIMIT}


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
          f"   r値: {R_VALUES}")

    runs = build_runs(search_qids, queries, webstyle)

    summary = {}
    for name in QREL_SETS:
        for m in METHODS:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], search_qids, f"{m}/{name}")

    suffix = "" if SUBSET_STRIDE == 1 and not LIMIT else f"_stride{SUBSET_STRIDE}_n{len(search_qids)}"
    out = os.path.join(RAG_DIR, f"bm25f_geometric_decay_result{suffix}.json")
    with open(out, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "config": run_config(), "summary": summary},
                   f, indent=2)
    print(f"\n-> {out}")

    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print(f"{'r':>7}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for m, r in zip(METHODS, R_VALUES):
            a = summary[qs][m]
            if a: print(f"{r:>7}"+"".join(f"{a[k]:>10.4f}" for k in METRIC_KEYS))


if __name__ == "__main__":
    main()
