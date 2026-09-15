"""
位置ボーナスの診断実験。これまでのposudensity系（min-max版・meanratio版・weight_base版・
zoneアブレーション）がことごとくnDCG@10で±0.004以内に収まり、「前半のみ/後半のみ/全体」と
いう正反対の重み付けすら区別できなかったことを受け、原因を2つの仮説に切り分ける。

仮説1: 測定した対象と使う対象のズレ。
  U字カーブは「全トークンの出現」について相対位置ごとの平均IDFを測ったものだが、実装は
  各span_termの「初出位置」しか見ていなかった。診断（候補文書11,244サンプル）では初出位置は
  冒頭に強く偏り、88.4%が相対位置0.5未満、末尾の山(0.9〜1.0)はわずか1.49%。つまり測定した
  カーブの右半分は初出位置ベースでは事実上発火しない。
  → occurrences="all" で全出現をカーブで重み付けして平均し、測定時の意味と一致させる。

仮説2: ボーナス幅が小さすぎる。boost_ratio=0.05では順位がほとんど動かない。
  → boost_ratio=0.0（＝位置ボーナス無しの対照）と0.2/0.5を比較する。

条件:
- baseline                : narrativeそのままbm25_body（従来表との継続性のため）
- posboost_only           : 既存チャンピオン（参照用。同条件でnDCG@10=0.5007, consensus）
- noboost_control         : boost_ratio=0.0。位置ボーナスを一切かけない均等重み(equalweight)検索
                            （★これまで欠けていた対照条件。posudensity系の改善が本当に
                            位置ボーナス由来なのか、クエリ構成だけで説明できるのかを確定させる）
- occall_br0.5/1.0/2.0    : occurrences="all", boost_ratio=0.5/1.0/2.0, amplify=13
                            （ボーナス幅のスイープ。br=0.05が完全に不活性だったので大きめまで振る）
- occfirst_br1.0          : occurrences="first", boost_ratio=1.0, amplify=13
                            （occall_br1.0との差が「初出のみ vs 全出現」の正味の効果）

フィールド重みは全条件recallopt比率（title=2/headings=1/body=1）に固定。

使い方:
    python evaluate_posdensity_occ.py [QUERY_REPEAT]
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, rrf_fuse, bm25_equalweight_posboost_discourseboost,
                        bm25_equalweight_posdensity_occ, analyze_terms)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
AMPLIFY = 13.0

# (条件名, occurrences, boost_ratio)
# boost_ratioは大きめまで振る。ボーナスは boost_ratio × 基準スコア × (triggered_idf/idf_sum)
# で、amplify=13のmeanratio重みだと (triggered_idf/idf_sum) は概ね0.7〜1.7（1.0中心に±0.5）。
# つまり文書間の実効的な判別幅は約 0.5 × boost_ratio × 基準スコア にしかならない。
# 既存チャンピオンのspan_boost=15.0は実スコア170〜180に対して約8.6%相当なので、
# br=0.2でようやく同程度。br=0.05が完全に不活性だったことを踏まえ、1.0〜2.0まで見る。
POS_CONDITIONS = [
    ("noboost_control", "all", 0.0),
    ("occall_br0.5", "all", 0.5),
    ("occall_br1.0", "all", 1.0),
    ("occall_br2.0", "all", 2.0),
    ("occfirst_br1.0", "first", 1.0),
]

METHODS = ["baseline", "posboost_only"] + [name for name, _, _ in POS_CONDITIONS]

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


class CachedSearch:
    def __init__(self, fn, topk):
        self.fn = fn
        self.topk = topk
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, text):
        key = text.strip()
        if not key:
            return []
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self._cache[key] = self.fn(key, k=self.topk)
        return self._cache[key]

    def stats(self, label):
        return f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回"


class CachedSpanSearch:
    def __init__(self, fn, topk):
        self.fn = fn
        self.topk = topk
        self._cache = {}
        self.calls = self.hits = 0
        self.total_time = 0.0

    def __call__(self, title_text, headings_text, body_text, span_terms):
        key = (title_text.strip(), headings_text.strip(), body_text.strip(), tuple(span_terms))
        if not any(key[:3]):
            return []
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        t0 = time.time()
        self._cache[key] = self.fn(title_text, headings_text, body_text, list(span_terms), k=self.topk)
        self.total_time += time.time() - t0
        return self._cache[key]

    def stats(self, label):
        avg = self.total_time / self.calls if self.calls else 0.0
        return (f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回 / "
                f"合計 {self.total_time:.0f}s (平均 {avg:.2f}s/回)")


def fuse(lists, topk):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


def bm25_equalweight_posboost_only(title_text, headings_text, body_text, span_terms, k=100):
    return bm25_equalweight_posboost_discourseboost(title_text, headings_text, body_text, span_terms,
                                                  k=k, markers=[])


def make_occ_fn(occurrences, boost_ratio):
    def fn(title_text, headings_text, body_text, span_terms, k=100):
        return bm25_equalweight_posdensity_occ(title_text, headings_text, body_text, span_terms, k=k,
                                            title_boost=2, headings_boost=1, body_boost=1,
                                            boost_ratio=boost_ratio, amplify=AMPLIFY,
                                            occurrences=occurrences)
    return fn


def build_run(method, qids, queries, webstyle, search_body, span_searchers):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        if method == "baseline":
            run[qid] = fuse([search_body(q)], TOPK)
        elif method in span_searchers:
            search_fn = span_searchers[method]
            pairs = dq_pairs_structured(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = []
            for dq, doc in pairs:
                title_q = f"{repeated_q} {dq} {doc['title']}"
                headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
                body_q = f"{repeated_q} {dq} {doc['body']}"
                span_terms = analyze_terms(dq, field="body")
                lists.append(search_fn(title_q, headings_q, body_q, span_terms))
            run[qid] = fuse(lists, TOPK)
        else:
            raise ValueError(f"未知の method: {method}")
        if i % 20 == 0 or i == len(qids):
            print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return run


def evaluate(run, qrels, eval_qids, label):
    target = [q for q in eval_qids if q in qrels]
    if not target:
        return None, 0
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg, n


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries
        and webstyle[qid].get("decomposed_queries")
        and any(d and d.strip() for d in webstyle[qid].get("query2doc_docs", []))
    )
    search_qids = valid_qids[::3]
    print(f"全トピック: {len(valid_qids)}   探索用サブセット: {len(search_qids)}")
    print(f"条件: {', '.join(METHODS)}")
    print(f"TOPK={TOPK}  RETRIEVE_K={RETRIEVE_K}  QUERY_REPEAT={QUERY_REPEAT}  amplify={AMPLIFY}")
    print("=" * 72)

    search_body = CachedSearch(bm25_body, RETRIEVE_K)
    span_searchers = {"posboost_only": CachedSpanSearch(bm25_equalweight_posboost_only, RETRIEVE_K)}
    for name, occ, br in POS_CONDITIONS:
        span_searchers[name] = CachedSpanSearch(make_occ_fn(occ, br), RETRIEVE_K)

    runs = {m: build_run(m, search_qids, queries, webstyle, search_body, span_searchers)
            for m in METHODS}
    print(f"\n{search_body.stats('bm25_body')}")
    for name, searcher in span_searchers.items():
        print(searcher.stats(name))

    eval_qids = [q for q in search_qids if all(runs[m].get(q) for m in METHODS)]
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    summary = {}
    for method in METHODS:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            agg, n = evaluate(runs[method], qrels[name], eval_qids, f"{method} / {name}")
            summary.setdefault(name, {})[method] = agg

    print("\n" + "=" * 72)
    print(f"SUMMARY   対象: {len(eval_qids)} クエリ")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
                  "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
        header = "method".ljust(24) + "".join(labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("noboost_control")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(24)
            if agg is None:
                print(row + "-")
                continue
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method != "noboost_control" and base:
                    cell += f" ({agg[key] - base[key]:+.4f})"
                row += cell.ljust(16)
            print(row)
        print("（括弧内は noboost_control との差＝位置ボーナスの正味の寄与）")
    print("=" * 72)

    out_path = os.path.join(RAG_DIR, f"posdensity_occ_eval_summary_rep{QUERY_REPEAT}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK, "amplify": AMPLIFY,
                   "conditions": [(n, o, b) for n, o, b in POS_CONDITIONS],
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
