"""
BM25F の k1 を「定数」ではなく「文書長の関数」にしたらどうなるかを検証する。

背景:
evaluate_k1_decay_sweep.py（位置ブースト系のbm25_fieldedベース）で k1 を定数として
0.1〜2.0でスイープしたところ、両qrels・全4指標で単調に「k1が大きいほど良い」となり、
「TFを減衰させて長い文書の水増しを抑える」という当初の狙いは裏目に出た（k1を下げると
本当に濃く関連している文書の正当なシグナルまで削ってしまうため）。

指摘: k1を全文書一律の定数にするのではなく、**文書長の関数**にすればどうか。
    - 平均より短い文書: k1=k1_base のまま（減衰させない。TFのシグナルをそのまま信じる
      ——ここはk1定数スイープで「高いほど良い」と分かった領域なので活かす）
    - 平均より長い文書: k1_base から徐々に下げる（繰り返しの水増しだけを狙って抑える）

    k1(D) = k1_base * (avgdl_body / max(|D|_body, avgdl_body)) ** alpha

alpha=0 で「k1定数」（evaluate_k1_decay_sweep.pyと同じ挙動）に一致する。alpha>0で
平均より長い文書だけk1が下がる。「長さ」はbody（最も変動が大きく、繰り返しの水増しが
起きるフィールド）のtf/文書長で測る。

qtf（クエリ側の重み付け）はevaluate_bm25f_k3_sweep.pyで既に線形(linear)が最良と
判明しているので、ここではlinear固定にし、k1関数化という新しい軸だけを切り分ける。

設定はevaluate_bm25f_k3_sweep.pyと同一（n=105、RETRIEVE_K=3000、QUERY_REPEAT=5、
weights 2/1/1、b=0.0/0.8/1.0——avgdlバグ修正後に再探索した正しい値、gemini疑似文書、
Subquery単位検索→RRF k=60）。逐次処理でトピックごとに即スコア化・破棄する
（OOM対策、evaluate_k1_decay_sweep.pyでの教訓を反映）。

k1_base × alpha のグリッド（alpha=0の行がk1定数スイープの再現＝検算）:
    k1_base ∈ {0.9, 1.5, 2.0, 3.0}
    alpha   ∈ {0.0, 0.25, 0.5, 1.0, 1.5, 2.0}

使い方:
    python3 evaluate_bm25f_k1_length_sweep.py [SUBSET_STRIDE] [LIMIT]
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
CKPT = os.path.join(RAG_DIR, "bm25f_k1_length_sweep_ckpt.json")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000
QUERY_REPEAT = 5
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
B_TITLE, B_HEADINGS, B_BODY = 0.0, 0.8, 1.0   # avgdlバグ修正後の正しい再探索値

K1_BASES = [0.9, 1.5, 2.0, 3.0]
ALPHAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
METHODS = [f"k1b{kb:g}_a{a:g}" for kb in K1_BASES for a in ALPHAS]

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


def score_variants(raw, qtf, k):
    """bm25f_prepare()の生データからk1_base×alphaの全グリッドを同時に作る（I/Oなし）。
    qtfはlinear（既に最良と判明済み）で固定。"""
    if raw is None:
        return {m: [] for m in METHODS}
    if not raw["query_terms"]:
        flat = [(d, 0.0) for d in raw["candidates"][:k]]
        return {m: list(flat) for m in METHODS}

    idf, avgdl, docs = raw["idf"], raw["avgdl"], raw["docs"]
    weights = {"title": TITLE_W, "headings": HEADINGS_W, "body": BODY_W}
    bs = {"title": B_TITLE, "headings": B_HEADINGS, "body": B_BODY}
    avgdl_body = avgdl.get("body") or 1.0

    n_cond = len(METHODS)
    scores = {m: {} for m in METHODS}
    for docid, doc in docs.items():
        field_terms, field_len = doc["field_terms"], doc["field_len"]
        body_len = field_len.get("body", 0) or 1
        # 文書ごとのk1（グリッド全条件ぶん先に計算）
        ratio = avgdl_body / max(body_len, avgdl_body)
        k1_eff = [kb * (ratio ** a) for kb in K1_BASES for a in ALPHAS]

        pseudo_tf_sum = 0.0
        acc_pseudo_tf = 0.0
        per_term_pseudo_idf = []   # (idf, pseudo_tf) を溜めて後でk1ごとに合算
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
                n = float(qtf.get(t, 1))
                per_term_pseudo_idf.append((t_idf, pseudo_tf, n))

        acc = [0.0] * n_cond
        for t_idf, pseudo_tf, n in per_term_pseudo_idf:
            for i, k1v in enumerate(k1_eff):
                acc[i] += t_idf * pseudo_tf / (k1v + pseudo_tf) * n
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


def run_config():
    return {"retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT, "topk": TOPK,
            "weights": [TITLE_W, HEADINGS_W, BODY_W], "b": [B_TITLE, B_HEADINGS, B_BODY],
            "k1_bases": K1_BASES, "alphas": ALPHAS,
            "stride": SUBSET_STRIDE, "limit": LIMIT}


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
          f"   条件数: {len(METHODS)}")

    runs = build_runs(search_qids, queries, webstyle)

    summary = {}
    for name in QREL_SETS:
        print(f"\n[{name}]")
        for m in METHODS:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], search_qids, f"{m}/{name}")

    suffix = "" if SUBSET_STRIDE == 1 and not LIMIT else f"_stride{SUBSET_STRIDE}_n{len(search_qids)}"
    out = os.path.join(RAG_DIR, f"bm25f_k1_length_sweep_result{suffix}.json")
    with open(out, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "config": run_config(), "summary": summary},
                   f, indent=2)
    print(f"\n-> {out}")

    for qs in QREL_SETS:
        print(f"\n=== {qs} nDCG@10 上位10 ===")
        rows = [(m, summary[qs][m]) for m in METHODS if summary[qs][m]]
        for m, a in sorted(rows, key=lambda x: -x[1]["ndcg_cut_10"])[:10]:
            print(f"  {m:<16} nDCG@10={a['ndcg_cut_10']:.4f}  R@1000={a['recall_1000']:.4f}")


if __name__ == "__main__":
    main()
