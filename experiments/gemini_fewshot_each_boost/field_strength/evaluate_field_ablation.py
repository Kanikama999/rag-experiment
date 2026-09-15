"""
title/headings/body単体のアブレーション実験（REPORT.md 4.2節の再実行）。

Subquery単位で生成済みのweb風疑似文書（title/headings/body）を使い、
対応するフィールド1つだけをmatchした場合と、3フィールドを線形和で合算した場合
（bm25_fielded、既定boost title=3,headings=2,body=1）を比較する。

条件:
- baseline: narrativeそのままbm25_body
- title_only / headings_only / body_only: narrative×N + Subquery + 対応する
  疑似文書パートだけを、対応する1フィールドにのみmatch（重み1）
- fielded: title/headings/bodyをそれぞれ対応フィールドにmatchし、bm25_fieldedの
  既定boost（title^3+headings^2+body^1）でスコアを線形和
- fielded_equal: fieldedと同じだが、boostをtitle^1+headings^1+body^1（単体条件と
  揃えた等倍重み）にしたもの。title_only等は重み1なので、fieldedとの比較は
  「フィールド分割+線形和という構造の効果」と「titleを重めにするブースト自体の効果」
  が混ざってしまう。fielded_equalは後者を除いた、単体条件と公平に比較できる条件。
- title_headings / title_body / headings_body: 2フィールドだけを等倍重み（1:1）で
  matchしたペア条件（bm25_fieldedに対象外フィールドは空文字で渡し除外）。単体→ペア
  →3フィールド（fielded_equal）で相乗効果がどこから積み上がるかを見る。
- narrative_title_only / narrative_headings_only: baselineと同じく繰り返し・
  Subquery分解・疑似文書を一切使わない生のnarrativeを、bodyの代わりにtitle/headings
  フィールドだけへmatchした場合（baseline自体が「narrativeをbodyだけに」に相当する
  ので、この2つを合わせて3フィールド分がそろう）。

使い方:
    python evaluate_field_ablation.py [QUERY_REPEAT] [METHODS_CSV]
    # METHODS_CSV省略時は全条件（baseline,title_only,headings_only,body_only,
    #   title_headings,title_body,headings_body,fielded,fielded_equal,
    #   narrative_title_only,narrative_headings_only）
    # 例: 既に確定済みの条件を除いて再計算する場合
    #   python evaluate_field_ablation.py 5 baseline,fielded_equal
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
from retriever import bm25_body, bm25_title, bm25_headings, bm25_fielded, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PARENT_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(PARENT_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5

PAIR_FIELDS = {
    "title_headings": {"title", "headings"},
    "title_body": {"title", "body"},
    "headings_body": {"headings", "body"},
}

ALL_METHODS = (["baseline", "title_only", "headings_only", "body_only"]
               + list(PAIR_FIELDS) + ["fielded", "fielded_equal",
               "narrative_title_only", "narrative_headings_only"])
METHODS = sys.argv[2].split(",") if len(sys.argv) > 2 else ALL_METHODS

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


class CachedSearch:
    def __init__(self, fn, topk):
        self.fn = fn
        self.topk = topk
        self._cache = {}

    def __call__(self, text):
        key = text.strip()
        if not key:
            return []
        if key not in self._cache:
            self._cache[key] = self.fn(key, k=self.topk)
        return self._cache[key]


class CachedFieldedSearch:
    def __init__(self, fn, topk):
        self.fn = fn
        self.topk = topk
        self._cache = {}

    def __call__(self, title_text, headings_text, body_text):
        key = (title_text.strip(), headings_text.strip(), body_text.strip())
        if not any(key):
            return []
        if key not in self._cache:
            self._cache[key] = self.fn(*key, k=self.topk)
        return self._cache[key]


def fuse(lists, topk):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


def dq_pairs_structured(entry):
    """(Subquery, {"title","headings","body"}) のペアを、欠落文書を除いて返す。"""
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"],
                                          entry["query2doc_docs_structured"])
            if doc and doc.get("body")]


def build_run(method, qids, queries, webstyle, search_body, search_title,
               search_headings, search_fielded, search_fielded_equal):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        repeated_q = " ".join([q] * QUERY_REPEAT)

        if method == "baseline":
            run[qid] = fuse([search_body(q)], TOPK)
        elif method == "narrative_title_only":
            run[qid] = fuse([search_title(q)], TOPK)
        elif method == "narrative_headings_only":
            run[qid] = fuse([search_headings(q)], TOPK)
        else:
            pairs = dq_pairs_structured(webstyle[qid])
            if method == "title_only":
                lists = [search_title(f"{repeated_q} {dq} {doc['title']}")
                         for dq, doc in pairs]
            elif method == "headings_only":
                lists = [search_headings(f"{repeated_q} {dq} {' '.join(doc['headings'])}")
                         for dq, doc in pairs]
            elif method == "body_only":
                lists = [search_body(f"{repeated_q} {dq} {doc['body']}")
                         for dq, doc in pairs]
            elif method == "fielded":
                lists = [search_fielded(f"{repeated_q} {dq} {doc['title']}",
                                         f"{repeated_q} {dq} {' '.join(doc['headings'])}",
                                         f"{repeated_q} {dq} {doc['body']}")
                         for dq, doc in pairs]
            elif method == "fielded_equal":
                lists = [search_fielded_equal(f"{repeated_q} {dq} {doc['title']}",
                                               f"{repeated_q} {dq} {' '.join(doc['headings'])}",
                                               f"{repeated_q} {dq} {doc['body']}")
                         for dq, doc in pairs]
            elif method in PAIR_FIELDS:
                included = PAIR_FIELDS[method]
                lists = [search_fielded_equal(
                    f"{repeated_q} {dq} {doc['title']}" if "title" in included else "",
                    f"{repeated_q} {dq} {' '.join(doc['headings'])}" if "headings" in included else "",
                    f"{repeated_q} {dq} {doc['body']}" if "body" in included else "")
                    for dq, doc in pairs]
            else:
                raise ValueError(f"未知の method: {method}")
            run[qid] = fuse(lists, TOPK)

        if i % 20 == 0 or i == len(qids):
            print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()
    return run


def evaluate(run, qrels, eval_qids, label):
    target = [q for q in eval_qids if q in qrels]
    if not target:
        print(f"[{label}] 採点対象なし")
        return None
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg


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
        and any(d and d.get("body") for d in webstyle[qid].get("query2doc_docs_structured", [])))
    print(f"疑似文書がある {len(valid_qids)} クエリ")
    print(f"条件: {', '.join(METHODS)}   QUERY_REPEAT={QUERY_REPEAT}")
    print("=" * 72)

    search_body = CachedSearch(bm25_body, RETRIEVE_K)
    search_title = CachedSearch(bm25_title, RETRIEVE_K)
    search_headings = CachedSearch(bm25_headings, RETRIEVE_K)
    search_fielded = CachedFieldedSearch(bm25_fielded, RETRIEVE_K)

    def bm25_fielded_equal(title_text, headings_text, body_text, k):
        return bm25_fielded(title_text, headings_text, body_text, k=k,
                             title_boost=1, headings_boost=1, body_boost=1)

    search_fielded_equal = CachedFieldedSearch(bm25_fielded_equal, RETRIEVE_K)

    runs = {m: build_run(m, valid_qids, queries, webstyle, search_body, search_title,
                          search_headings, search_fielded, search_fielded_equal)
            for m in METHODS}

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in METHODS)]
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    summary = {}
    for method in METHODS:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            agg = evaluate(runs[method], qrels[name], eval_qids, f"{method} / {name}")
            summary.setdefault(name, {})[method] = agg

    print("\n" + "=" * 72)
    print(f"SUMMARY   対象: {len(eval_qids)} クエリ")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
                  "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
        header = "method".ljust(16) + "".join(labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(16)
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method != "baseline" and base:
                    cell += f" ({agg[key] - base[key]:+.4f})"
                row += cell.ljust(16)
            print(row)
    print("=" * 72)

    suffix = "" if METHODS == ALL_METHODS else "_" + "-".join(METHODS)
    out_path = os.path.join(RAG_DIR, f"field_ablation_eval_summary_rep{QUERY_REPEAT}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
