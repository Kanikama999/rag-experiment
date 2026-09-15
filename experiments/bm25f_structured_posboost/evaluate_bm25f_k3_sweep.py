"""
BM25F のクエリ側飽和パラメータ k3 をスイープする。

## なぜやるか

evaluate_bm25f_narrative_rep5.py は qtf の掛け方を3通り比較し、飽和版 (k3=8) が最良
（consensus nDCG@10 = 0.4456）と結論した。しかし **k3=8 は一度も振っていない**。
α（§7.9）と span_end（§7.10）で2回起きた「試した1点を最適値として扱う」問題が
そのまま当てはまる。α は最適値が 1.5〜3 で従来の 1 ではなく、span_boost も内点に
最適があったので、k3=8 が最適である保証はない。

## k3 は dedup と線形 qtf を連続的に繋ぐ

倍率は `mult(t) = (k3+1)·qtf(t) / (k3 + qtf(t))` で、両端が既存の2条件に一致する:

    k3 → 0   : mult = 1·qtf/qtf = 1        → bm25f_dedup（narrative×5 が無視される）
    k3 → ∞   : mult → qtf                  → bm25f_qtf（線形）

つまり k3 のスイープは「qtf を全く効かせない」から「線形に効かせる」までの
1次元の連続族であり、既存の3条件はその上の3点にすぎない。k3=0 と k3=8 は既存値の
再現確認になるので、妥当性チェックが2重に入る。

## コストがほぼゼロで済む理由

全条件を bm25f_prepare() の**同一の生データから計算する**。ネットワーク I/O
（_mtermvectors と候補取得）は1条件ぶんしかかからず、k3 を増やしても増えるのは
Python 側の掛け算だけ。したがって9条件を1回の実行（105トピックで約100分）でまとめて
測れる。条件間の差は純粋に k3 の違いだけになる。

設定は evaluate_bm25f_narrative_rep5.py と基本同一だが、b は search_bm25f_b_3way.py
で再探索した正しい値（avgdlバグ修正後）を使う
（n=105、RETRIEVE_K=3000、QUERY_REPEAT=5、weights 2/1/1、b=0.0/0.8/1.0、k1=0.9、
 gemini疑似文書、Subquery単位検索→RRF k=60）。

使い方:
    python3 evaluate_bm25f_k3_sweep.py            # 105トピック
    python3 evaluate_bm25f_k3_sweep.py 3          # stride=3 の35トピック
    python3 evaluate_bm25f_k3_sweep.py 1 5        # 先頭5トピック（動作確認）
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
DATA_DIR = os.path.join(RAG_DIR, "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
CKPT = os.path.join(RAG_DIR, "bm25f_k3_sweep_ckpt.json")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000
QUERY_REPEAT = 5
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
# 2026-09-10: avgdl バグ修正後に3軸グリッド(bm25f_b_3way_grid_result.json)で再探索した
# nDCG@10 最良点。旧値 0.6/0.2/0.2 は avgdl=1.0 のバグ下で選ばれたもので無効。
# b は定義上[0,1]なので b_body=1.0 / b_title=0.0 は定義域の端＝探索は完結している。
B_TITLE, B_HEADINGS, B_BODY = 0.0, 0.8, 1.0
BM25_K1 = 0.9

# k3=0 は dedup と、k3=8 は既存の飽和版と一致するはず（妥当性チェック）
K3_VALUES = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
METHODS = [f"k3_{k:g}" for k in K3_VALUES] + ["linear"]

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
    """論理クエリ本文を body の analyzer でトークン化し、語ごとの出現回数を数える
    （analyze_terms() は重複除去するので生トークン列から数える）。"""
    res = client.indices.analyze(index=INDEX, body={"field": "body", "text": logical_query})
    return Counter(t["token"] for t in res.get("tokens", []))


def score_variants(raw, qtf, k):
    """bm25f_prepare() の生データから全 k3 条件のランキングを同時に作る（I/Oなし）。

    倍率 mult(t) = (k3+1)*qtf/(k3+qtf)。k3=0 で 1（dedup）、k3→∞ で qtf（線形）。
    "linear" は qtf をそのまま掛ける条件（k3=∞ の極限）。
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
        n = float(qtf.get(t, 1))
        row = [(k3 + 1.0) * n / (k3 + n) if (k3 + n) > 0 else 1.0 for k3 in K3_VALUES]
        row.append(n)          # linear
        mult[t] = row

    n_cond = len(METHODS)
    scores = {m: {} for m in METHODS}
    for docid, doc in docs.items():
        field_terms, field_len = doc["field_terms"], doc["field_len"]
        acc = [0.0] * n_cond
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
                for i in range(n_cond):
                    acc[i] += contrib * m[i]
        for i, name in enumerate(METHODS):
            scores[name][docid] = acc[i]

    return {name: sorted(scores[name].items(), key=lambda x: -x[1])[:k] for name in METHODS}


def run_config():
    return {"retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT, "topk": TOPK,
            "weights": [TITLE_W, HEADINGS_W, BODY_W], "b": [B_TITLE, B_HEADINGS, B_BODY],
            "k1": BM25_K1, "k3_values": K3_VALUES,
            "stride": SUBSET_STRIDE, "limit": LIMIT}


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
        print(f"  {done}/{len(todo)}  qid={qid}  経過{el/60:.1f}分  残り約{eta/60:.0f}分",
              flush=True)
        with open(CKPT, "w") as f:
            json.dump({"config": run_config(), "runs": runs}, f)
    return runs


def evaluate(run, qrels, qids):
    target = [q for q in qids if q in qrels and run.get(q)]
    if not target:
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
    n = len(res)
    out = {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}
    out["n"] = n
    return out


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
        and any(d and d.get("body") for d in webstyle[qid].get("query2doc_docs_structured", []))
    )
    qids = valid_qids[::SUBSET_STRIDE]
    if LIMIT:
        qids = qids[:LIMIT]
    print(f"全トピック {len(valid_qids)} / 対象 {len(qids)}")
    print(f"k3: {K3_VALUES} + linear(k3=∞)  = {len(METHODS)} 条件")
    print(f"RETRIEVE_K={RETRIEVE_K} QUERY_REPEAT={QUERY_REPEAT} k1={BM25_K1} "
          f"weights={TITLE_W}/{HEADINGS_W}/{BODY_W} b={B_TITLE}/{B_HEADINGS}/{B_BODY}")
    print("=" * 72)

    runs = build_runs(qids, queries, webstyle)

    summary = {}
    for name in QREL_SETS:
        for m in METHODS:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], qids)

    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    print("\n" + "=" * 72)
    print(f"SUMMARY  対象 {len(qids)} クエリ")
    for name in QREL_SETS:
        print(f"\n[{name}]   mult(t) = (k3+1)·qtf/(k3+qtf)")
        header = "condition".ljust(12) + "".join(labels[k].ljust(15) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        for m in METHODS:
            agg = summary[name].get(m)
            row = m.ljust(12)
            if not agg:
                print(row + "-")
                continue
            row += "".join(f"{agg[k]:.4f}".ljust(15) for k in METRIC_KEYS)
            print(row)

        ok = [m for m in METHODS if summary[name].get(m)]
        if ok:
            print(f"\n  [{name}] 指標ごとの最良 k3")
            for k in METRIC_KEYS:
                best = max(ok, key=lambda m: summary[name][m][k])
                edge = ""
                if best in (METHODS[0], METHODS[-2], "linear"):
                    edge = "  ← 端（範囲を広げる検討）"
                print(f"    {labels[k]:14s} {best:10s} {summary[name][best][k]:.4f}{edge}")

    # 妥当性チェック（既知値は n=105 のものなので、全量実行のときだけ意味がある）
    print("\n" + "=" * 72)
    if SUBSET_STRIDE != 1 or LIMIT:
        print("妥当性チェックは全量実行(stride=1, limit=0)のときのみ実施。"
              f"今回は {len(qids)} トピックなのでスキップ。")
        known = {}
    else:
        print("※ 既存値は avgdl バグ下(b=0.6/0.2/0.2)のものなので一致しない。参考表示のみ。")
        known = {"k3_0":  {"consensus": {"ndcg_cut_10": 0.3832665879375721,
                                          "recall_1000": 0.28886390577593773}},
                 "k3_8":  {"consensus": {"ndcg_cut_10": 0.4456208718991201,
                                          "recall_1000": 0.2957690069677202}},
                 "linear": {"consensus": {"ndcg_cut_10": 0.424319071114132,
                                          "recall_1000": 0.28340620278099793}}}
    for m, exp in known.items():
        got = summary.get("consensus", {}).get(m)
        if not got:
            continue
        for k, v in exp["consensus"].items():
            print(f"  {m:8s} {k:12s} 既存 {v:.4f} / 今回 {got[k]:.4f}  "
                  f"差 {got[k] - v:+.4f}")

    out_path = os.path.join(RAG_DIR, "bm25f_k3_sweep_result.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(qids), "qids": qids,
                   "config": run_config(), "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"\n集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
