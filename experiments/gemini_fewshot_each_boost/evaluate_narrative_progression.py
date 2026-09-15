"""
narrativeそのままのbm25_bodyから、doc版の現状最良に至るまでの積み上げを7段階で比較する。

1. baseline: narrativeそのままbm25_body（弱ベースライン）
2. narrative_repeat5: narrative×5をbm25_body（繰り返しだけの効果）
3. subquery_rrf: narrativeをSubqueryに分解し、各Subqueryをそのままbm25_body、
   Subquery横断でRRF融合（疑似文書生成なし、フィールド分割なし）
4. fielded_recallopt: narrative×5+Subquery+LLM生成疑似文書(title/headings/body)を
   fielded検索（title=2,headings=1,body=1、search_field_boosts.pyのrecallopt。
   posboost/discourseboostを足す前の「真のベースライン」）
5. 4 + posboost（span_end=100, span_boost=15.0、search_posboost_params.pyの
   チューニング値）
6. 4 + discourseboost（要約系discourse marker直後、slop=20, marker_boost=3.0）
7. 4 + 両方（bm25_fielded_posboost_discourseboost、デフォルトが5・6と同じ値）

すべて同じeval_qids（疑似文書がある105トピック）・同じQUERY_REPEAT=5・同じRETRIEVE_K=1000
で統一して比較する（既存の個別jsonは実行タイミングが違うため、ここで揃えて再計算する）。

使い方:
    python evaluate_narrative_progression.py [QUERY_REPEAT]
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, bm25_fielded, bm25_fielded_posboost,
                        bm25_fielded_discourseboost, bm25_fielded_posboost_discourseboost,
                        rrf_fuse, analyze_terms)


def bm25_fielded_recallopt(title_text, headings_text, body_text, k=100):
    return bm25_fielded(title_text, headings_text, body_text, k=k,
                         title_boost=2, headings_boost=1, body_boost=1)


def bm25_fielded_posboost_tuned(title_text, headings_text, body_text, span_terms, k=100):
    return bm25_fielded_posboost(title_text, headings_text, body_text, span_terms, k=k,
                                  title_boost=2, headings_boost=1, body_boost=1,
                                  span_end=100, span_boost=15.0)


RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5

METHODS = ["baseline", "narrative_repeat5", "subquery_rrf", "fielded_recallopt",
           "fielded_recallopt_posboost", "fielded_recallopt_discourseboost",
           "fielded_recallopt_posboost_discourseboost"]

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


class CachedBody:
    def __init__(self, topk):
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
        self._cache[key] = bm25_body(key, k=self.topk)
        return self._cache[key]

    def stats(self, label):
        return f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回"


class CachedFielded:
    def __init__(self, fn, topk):
        self.fn = fn
        self.topk = topk
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, title_text, headings_text, body_text):
        key = (title_text.strip(), headings_text.strip(), body_text.strip())
        if not any(key):
            return []
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self._cache[key] = self.fn(*key, k=self.topk)
        return self._cache[key]

    def stats(self, label):
        return f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回"


class CachedSpan:
    def __init__(self, fn, topk):
        self.fn = fn
        self.topk = topk
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, title_text, headings_text, body_text, span_terms):
        key = (title_text.strip(), headings_text.strip(), body_text.strip(), tuple(span_terms))
        if not any(key[:3]):
            return []
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self._cache[key] = self.fn(title_text, headings_text, body_text, list(span_terms), k=self.topk)
        return self._cache[key]

    def stats(self, label):
        return f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回"


def fuse(lists, topk):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


def dq_pairs_structured(entry):
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"],
                                          entry["query2doc_docs_structured"])
            if doc and doc.get("body")]


def build_run(method, qids, queries, webstyle, search_body, fielded_searchers, span_searchers):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]

        if method == "baseline":
            run[qid] = fuse([search_body(q)], TOPK)
        elif method == "narrative_repeat5":
            repeated_q = " ".join([q] * QUERY_REPEAT)
            run[qid] = fuse([search_body(repeated_q)], TOPK)
        elif method == "subquery_rrf":
            entry = webstyle[qid]
            dqs = entry.get("decomposed_queries") or []
            lists = [search_body(dq) for dq in dqs if dq and dq.strip()]
            run[qid] = fuse(lists, TOPK)
        elif method in fielded_searchers:
            search_fn = fielded_searchers[method]
            pairs = dq_pairs_structured(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = []
            for dq, doc in pairs:
                title_q = f"{repeated_q} {dq} {doc['title']}"
                headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
                body_q = f"{repeated_q} {dq} {doc['body']}"
                lists.append(search_fn(title_q, headings_q, body_q))
            run[qid] = fuse(lists, TOPK)
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
            print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()
    return run


def evaluate(run, qrels, eval_qids, label):
    target = [q for q in eval_qids if q in qrels]
    if not target:
        print(f"[{label}] 採点対象なし")
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
    print(f"疑似文書がある {len(valid_qids)} クエリ")
    print(f"条件: {', '.join(METHODS)}   TOPK={TOPK}   RETRIEVE_K={RETRIEVE_K}   "
          f"QUERY_REPEAT={QUERY_REPEAT}")
    print("=" * 72)

    search_body = CachedBody(RETRIEVE_K)
    fielded_searchers = {
        "fielded_recallopt": CachedFielded(bm25_fielded_recallopt, RETRIEVE_K),
    }
    span_searchers = {
        "fielded_recallopt_posboost": CachedSpan(bm25_fielded_posboost_tuned, RETRIEVE_K),
        "fielded_recallopt_discourseboost": CachedSpan(bm25_fielded_discourseboost, RETRIEVE_K),
        "fielded_recallopt_posboost_discourseboost": CachedSpan(bm25_fielded_posboost_discourseboost, RETRIEVE_K),
    }

    runs = {m: build_run(m, valid_qids, queries, webstyle, search_body, fielded_searchers, span_searchers)
            for m in METHODS}
    print(f"\n{search_body.stats('bm25_body')}")
    for name, searcher in fielded_searchers.items():
        print(searcher.stats(name))
    for name, searcher in span_searchers.items():
        print(searcher.stats(name))

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in METHODS)]
    if len(eval_qids) < len(valid_qids):
        print(f"[warn] 検索結果が空になった {len(valid_qids) - len(eval_qids)} クエリを"
              f"全条件から除外", file=sys.stderr)
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
        header = "method".ljust(38) + "".join(labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(38)
            if agg is None:
                print(row + "-")
                continue
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method != "baseline" and base:
                    cell += f" ({agg[key] - base[key]:+.4f})"
                row += cell.ljust(16)
            print(row)
    print("=" * 72)

    out_path = os.path.join(RAG_DIR, f"decomposed_query2doc_narrative_progression_rep{QUERY_REPEAT}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
