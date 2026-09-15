"""
doc版の現状最良設定 webstyle_narrative_equalweight_posboost_only の span_boost を
全105トピックで再検証する。

きっかけ: championのスコア内訳を_explainで実測したところ、span_first節がスコア全体の
27〜45%を占めており（bodyのmatch節と同等かそれ以上）、「軽い位置ボーナス」ではなく
主要なランキング基準になっていることが分かった。加えて span_boost=15 という採用値は
posboost_param_search_result_round2.json（35トピック）において recall@1000 を最大化する
点であって、nDCG@10 の最適値ではない:

  span_boost   recall@1000   nDCG@10
      8          0.28314     0.49922  ← nDCG@10 のピーク
     10          0.28523     0.49812
     15          0.28636     0.49211  ← recall@1000 のピーク（採用値）
     20          0.28618     0.49130
     30          0.28159     0.47243
     50          0.26900     0.45728

つまり現行championは nDCG@10 を約 -0.0071 損している可能性がある（35トピックでの値）。
この探索は35トピックのサブセットで行われたものなので、全105トピックでも同じ順序になるかを
確認する。span_boost=15 の条件は既知の doc版最良値（nDCG@10=0.5251, consensus, 105
トピック、decomposed_query2doc_webstyle_eval_summary_equalweight_ablation_rep5.json）を
再現するはずで、セットアップの妥当性チェックを兼ねる。

パイプラインは evaluate_decomposed_webstyle_equalweight_ablation.py と完全に同一
（narrative×5+Subquery+疑似文書を均等重み(1:1:1)でフィールド別match、span_terms=analyze_terms(Subquery)、
RETRIEVE_K=1000、QUERY_REPEAT=5、markers=[]でdiscourseboostは無効）。
再ランキングを伴わないネイティブクエリ1回だけなので、1条件あたり約30分。

使い方:
    python evaluate_spanboost_sweep.py [QUERY_REPEAT]
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, rrf_fuse, bm25_equalweight_posboost_discourseboost,
                        analyze_terms)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SPAN_END = 100

SPAN_BOOSTS = [8.0, 10.0, 15.0]  # 15.0 = 現行champion（参照・再現確認用）

METHODS = ["baseline"] + [f"posboost_sb{int(b)}" for b in SPAN_BOOSTS]

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


def make_posboost_fn(span_boost):
    def fn(title_text, headings_text, body_text, span_terms, k=100):
        return bm25_equalweight_posboost_discourseboost(
            title_text, headings_text, body_text, span_terms, k=k,
            span_end=SPAN_END, span_boost=span_boost, markers=[])
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
        return None, 0, {}
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    per_topic = {q: results[q]["ndcg_cut_10"] for q in results}
    return agg, n, per_topic


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
    print(f"対象トピック: {len(valid_qids)}（全件）")
    print(f"条件: {', '.join(METHODS)}   span_end={SPAN_END}")
    print(f"TOPK={TOPK}  RETRIEVE_K={RETRIEVE_K}  QUERY_REPEAT={QUERY_REPEAT}")
    print("=" * 72)

    search_body = CachedSearch(bm25_body, RETRIEVE_K)
    span_searchers = {}
    for b in SPAN_BOOSTS:
        span_searchers[f"posboost_sb{int(b)}"] = CachedSpanSearch(make_posboost_fn(b), RETRIEVE_K)

    runs = {m: build_run(m, valid_qids, queries, webstyle, search_body, span_searchers)
            for m in METHODS}
    print(f"\n{search_body.stats('bm25_body')}")
    for name, searcher in span_searchers.items():
        print(searcher.stats(name))

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in METHODS)]
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    summary, per_topic_all = {}, {}
    for method in METHODS:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            agg, n, pt = evaluate(runs[method], qrels[name], eval_qids, f"{method} / {name}")
            summary.setdefault(name, {})[method] = agg
            if name == "consensus":
                per_topic_all[method] = pt

    print("\n" + "=" * 72)
    print(f"SUMMARY   対象: {len(eval_qids)} クエリ")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
                  "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
        header = "method".ljust(22) + "".join(labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("posboost_sb15")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(22)
            if agg is None:
                print(row + "-")
                continue
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method != "posboost_sb15" and base:
                    cell += f" ({agg[key] - base[key]:+.4f})"
                row += cell.ljust(16)
            print(row)
        print("（括弧内は現行champion span_boost=15 との差）")

    # span_boost=8 と 15 のトピック単位の勝敗（差が偶然かどうかの目安）
    if "posboost_sb8" in per_topic_all and "posboost_sb15" in per_topic_all:
        a, b = per_topic_all["posboost_sb8"], per_topic_all["posboost_sb15"]
        common = sorted(set(a) & set(b))
        win = sum(1 for q in common if a[q] > b[q] + 1e-9)
        lose = sum(1 for q in common if a[q] < b[q] - 1e-9)
        tie = len(common) - win - lose
        print(f"\nトピック単位 nDCG@10（consensus）: sb8 の勝ち {win} / 負け {lose} / 引き分け {tie}"
              f"  （n={len(common)}）")

    print("=" * 72)
    out_path = os.path.join(RAG_DIR, f"spanboost_sweep_eval_summary_rep{QUERY_REPEAT}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK, "span_end": SPAN_END,
                   "span_boosts": SPAN_BOOSTS, "qids": eval_qids, "summary": summary,
                   "per_topic_ndcg10_consensus": per_topic_all},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
