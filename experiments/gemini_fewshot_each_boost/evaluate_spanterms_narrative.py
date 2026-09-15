"""
span_terms（posboostの判定に使う語）にnarrativeを足すとどうなるかを doc版で検証する。

背景: championは span_terms に Subquery の語だけを使う（narrative×5 も疑似文書も
一次検索のmatch節には入っているが、posboostの判定には使わない）。「なぜnarrativeや
疑似文書ではなくSubqueryなのか」という問いに対する現状の根拠は2つあった:

  1. 機構: span_or は選言なので語を足すほど発火率が上がる。候補200件に対する実測では
     Subquery のみ 10.0語 → 74.7% / +narrative 34.6語 → 94.4% /
     +narrative+疑似文書 161.1語 → 99.7%。飽和すると全候補に同じ定数を足すだけになり
     選別力を失う。（doc版で実測済み）
  2. 実測: seg版の evaluate_webstyle_posboost_discourseboost_narrativeterms.py で
     narrativeを足すと4指標すべて悪化、nDCG@10の悪化幅が3.8倍に拡大した
     （-0.0093 → -0.0350）。

しかし 2 は seg版の結果であり、doc版では未検証だった。本スクリプトはその穴を埋める。

条件（span_end=100, span_boost=15, markers=[] は全条件共通＝championと同一）:
- baseline                : narrativeそのままbm25_body
- spanterms_all           : Subquery全語（現行champion、既知値 nDCG@10=0.5251 consensus）
- spanterms_plus_narrative: Subquery語 + narrative語

疑似文書まで足す条件は span_or が平均152節になり1クエリ17秒（105トピックで約136分）
かかるうえ、発火率99.7%で機構的にはほぼ答えが出ているため本スクリプトには含めない。

パイプラインは evaluate_decomposed_webstyle_equalweight_ablation.py と同一
（narrative×5+Subquery+疑似文書を均等重み(1:1:1)でフィールド別match、RETRIEVE_K=1000、QUERY_REPEAT=5、全105トピック）。

使い方:
    python evaluate_spanterms_narrative.py [QUERY_REPEAT]
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, rrf_fuse, bm25_equalweight_posboost_discourseboost,
                        analyze_terms, client, INDEX)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SPAN_END = 100
SPAN_BOOST = 15.0

# (条件名, narrative語を足すか)
VARIANTS = [("spanterms_all", False), ("spanterms_plus_narrative", True)]

METHODS = ["baseline"] + [name for name, _ in VARIANTS]

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
        self.fn, self.topk, self._cache = fn, topk, {}
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
    """span_termsにnarrative語を足すかどうかだけが違う。それ以外はchampionと同一。"""

    def __init__(self, topk, add_narrative):
        self.topk, self.add_narrative, self._cache = topk, add_narrative, {}
        self.calls = self.hits = 0
        self.total_time = 0.0
        self.term_counts = []

    def __call__(self, title_text, headings_text, body_text, span_terms, narrative_terms):
        terms = list(span_terms)
        if self.add_narrative:
            terms = list(dict.fromkeys(terms + list(narrative_terms)))
        key = (title_text.strip(), headings_text.strip(), body_text.strip(), tuple(terms))
        if not any(key[:3]):
            return []
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self.term_counts.append(len(terms))
        t0 = time.time()
        self._cache[key] = bm25_equalweight_posboost_discourseboost(
            title_text, headings_text, body_text, terms, k=self.topk,
            span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
        self.total_time += time.time() - t0
        return self._cache[key]

    def stats(self, label):
        avg = self.total_time / self.calls if self.calls else 0.0
        nt = sum(self.term_counts) / len(self.term_counts) if self.term_counts else 0.0
        return (f"{label}: 呼び出し {self.calls} 回 / 平均 {avg:.2f}s/回 / "
                f"span_terms 平均 {nt:.1f} 語")


def fuse(lists, topk):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


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
            narrative_terms = analyze_terms(q, field="body")
            lists = []
            for dq, doc in pairs:
                title_q = f"{repeated_q} {dq} {doc['title']}"
                headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
                body_q = f"{repeated_q} {dq} {doc['body']}"
                span_terms = analyze_terms(dq, field="body")
                lists.append(search_fn(title_q, headings_q, body_q, span_terms, narrative_terms))
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
    return agg, n, {q: results[q]["ndcg_cut_10"] for q in results}


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
    print(f"条件: {', '.join(METHODS)}   span_end={SPAN_END}  span_boost={SPAN_BOOST}")
    print(f"TOPK={TOPK}  RETRIEVE_K={RETRIEVE_K}  QUERY_REPEAT={QUERY_REPEAT}")
    print("=" * 72)

    search_body = CachedSearch(bm25_body, RETRIEVE_K)
    span_searchers = {}
    for name, add_nar in VARIANTS:
        span_searchers[name] = CachedSpanSearch(RETRIEVE_K, add_nar)

    runs = {m: build_run(m, valid_qids, queries, webstyle, search_body, span_searchers)
            for m in METHODS}
    print(f"\n{search_body.stats('bm25_body')}")
    for name, searcher in span_searchers.items():
        print(searcher.stats(name))

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in METHODS)]
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    summary, per_topic = {}, {}
    for method in METHODS:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            agg, n, pt = evaluate(runs[method], qrels[name], eval_qids, f"{method} / {name}")
            summary.setdefault(name, {})[method] = agg
            if name == "consensus":
                per_topic[method] = pt

    print("\n" + "=" * 72)
    print(f"SUMMARY   対象: {len(eval_qids)} クエリ")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
                  "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
        header = "method".ljust(22) + "".join(labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("spanterms_all")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(22)
            if agg is None:
                print(row + "-")
                continue
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method != "spanterms_all" and base:
                    cell += f" ({agg[key] - base[key]:+.4f})"
                row += cell.ljust(16)
            print(row)
        print("（括弧内は現行champion＝Subquery全語 との差）")

    for m in METHODS:
        if m in ("baseline", "spanterms_all") or m not in per_topic:
            continue
        a, b = per_topic[m], per_topic["spanterms_all"]
        common = sorted(set(a) & set(b))
        win = sum(1 for q in common if a[q] > b[q] + 1e-9)
        lose = sum(1 for q in common if a[q] < b[q] - 1e-9)
        print(f"\nトピック単位 nDCG@10（consensus）: {m} の勝ち {win} / 負け {lose} / "
              f"引き分け {len(common) - win - lose}  （n={len(common)}）")

    print("=" * 72)
    out_path = os.path.join(RAG_DIR, f"spanterms_narrative_eval_summary_rep{QUERY_REPEAT}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK, "span_end": SPAN_END,
                   "span_boost": SPAN_BOOST, "variants": [[n, a] for n, a in VARIANTS],
                   "qids": eval_qids, "summary": summary,
                   "per_topic_ndcg10_consensus": per_topic},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
