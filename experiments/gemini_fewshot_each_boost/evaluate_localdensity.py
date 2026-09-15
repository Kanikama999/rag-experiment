"""
文書ごとの局所情報密度（local_density_profile()）を使った位置ボーナスの評価。

グローバル位置カーブ（コーパス全体を平均した相対位置ごとの平均IDF）は、first/mean/max の
3通りの集約すべてで noboost_control を下回り、関連性に対してノイズであることが確定した
（evaluate_posdensity_occ.py / evaluate_posdensity_occmax.py）。原因は、文書ごとに情報の
濃い領域の位置が異なるため、コーパス平均を取ると構造が消えること。

そこで本スクリプトは、グローバルなカーブを一切使わず、各文書の中で高IDF語がどこに密集して
いるかをその文書自身から求める方式（local_density_profile()）を評価する。事前診断（qrelsは
信号の有無の確認にのみ使用）で分かったこと:

  - 文書内の密度のばらつきは p90/p10 = 1.35 倍あり、コーパス平均カーブの 1.08 倍より
    はるかに大きい（平均を取ると構造が消えるという читание は正しかった）
  - クエリ語位置の密度を全出現の平均で見ると、正解 0.9898 / 不正解 0.9846、
    Cohen's d = +0.189（ほぼ差なし）
  - ★ max集約（最も密度の高い出現位置）で見ると、正解 1.3099 / 不正解 1.3562、
    Cohen's d = -0.308。今日の実験で最大の分離であり、しかも符号が負
    ＝「クエリ語が異常に密度の高い場所に出現する文書はむしろ正解ではない」

  解釈: 局所密度が突出して高い箇所はレア語が不自然に密集した領域（キーワードの羅列、
  タグクラウド、用語集、ナビゲーション、インデックスページ等）である可能性が高く、
  そこにクエリ語が埋まっている文書は「そのトピックについて書かれた記事」ではない。
  本当に関連する文書は普通の散文の中で自然な密度で言及する。

したがって本評価では amplify に負値を渡し、密度の高い出現を減点する方向で使う
（weight = 1 + amplify*(density - 1)、amplify<0 なので density>1 で weight<1）。

条件:
- baseline           : narrativeそのままbm25_body
- posboost_only      : 既存チャンピオン（参照用。同条件で nDCG@10=0.5007, consensus）
- noboost_control    : boost_ratio=0.0。位置ボーナス無し（★比較の基準、0.4710）
- ld_max_a-1_br0.5   : max集約, amplify=-1.0, boost_ratio=0.5
- ld_max_a-1_br1.0   : max集約, amplify=-1.0, boost_ratio=1.0
- ld_max_a-2_br1.0   : max集約, amplify=-2.0, boost_ratio=1.0

フィールド重みは全条件recallopt比率（title=2/headings=1/body=1）に固定。

使い方:
    python evaluate_localdensity.py [QUERY_REPEAT]
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, rrf_fuse, bm25_equalweight_posboost_discourseboost,
                        bm25_equalweight_localdensity, analyze_terms)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
WINDOW = 50

# (条件名, aggregate, amplify, boost_ratio)
POS_CONDITIONS = [
    ("noboost_control", "max", -1.0, 0.0),
    ("ld_max_a-1_br0.5", "max", -1.0, 0.5),
    ("ld_max_a-1_br1.0", "max", -1.0, 1.0),
    ("ld_max_a-2_br1.0", "max", -2.0, 1.0),
]

METHODS = ["baseline", "posboost_only"] + [c[0] for c in POS_CONDITIONS]

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


def make_ld_fn(aggregate, amplify, boost_ratio):
    def fn(title_text, headings_text, body_text, span_terms, k=100):
        return bm25_equalweight_localdensity(title_text, headings_text, body_text, span_terms, k=k,
                                          title_boost=2, headings_boost=1, body_boost=1,
                                          boost_ratio=boost_ratio, window=WINDOW,
                                          amplify=amplify, aggregate=aggregate)
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
    print(f"TOPK={TOPK}  RETRIEVE_K={RETRIEVE_K}  QUERY_REPEAT={QUERY_REPEAT}  window={WINDOW}")
    print("=" * 72)

    search_body = CachedSearch(bm25_body, RETRIEVE_K)
    span_searchers = {"posboost_only": CachedSpanSearch(bm25_equalweight_posboost_only, RETRIEVE_K)}
    for name, agg, amp, br in POS_CONDITIONS:
        span_searchers[name] = CachedSpanSearch(make_ld_fn(agg, amp, br), RETRIEVE_K)

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

    out_path = os.path.join(RAG_DIR, f"localdensity_eval_summary_rep{QUERY_REPEAT}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK, "window": WINDOW,
                   "conditions": [list(c) for c in POS_CONDITIONS],
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
