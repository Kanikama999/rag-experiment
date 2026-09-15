"""
Subquery を繰り返すと効くのかを検証する（doc版）。

背景:
  Query2doc論文の知見に従い、元クエリ(narrative)は QUERY_REPEAT=5 回繰り返してから
  疑似文書と連結している（繰り返さないと元クエリの語が疑似文書の語にTFで埋もれるため。
  1→5 で全指標が明確に改善したことを確認済み）。
  一方 Subquery はこれまで一貫して1回しか入れていなかった。繰り返しの効果が
  narrative 固有のものなのか、Subquery にも同じように効くのかは未検証だった。

設計（2x2）:
  「narrative の有無」×「Subquery の繰り返し回数」で切り分ける。
  Subquery×5 だけを champion と比べると、差が「narrative を外した効果」なのか
  「Subquery を5回にした効果」なのか判別できないため、n0_s1 を対照に入れてある。

    n5_s1 : narrative×5 + Subquery×1 + 各フィールドの生成テキスト   (= 現 champion)
    n5_s5 : narrative×5 + Subquery×5 + 各フィールドの生成テキスト
    n0_s1 : narrative なし + Subquery×1 + 各フィールドの生成テキスト
    n0_s5 : narrative なし + Subquery×5 + 各フィールドの生成テキスト

  検索構成（--retrieval）:
    champion    : bm25_equalweight_posboost_discourseboost(markers=[])。
                  nDCG@10 基準の現行最良（consensus 0.5251）。疑似文書の各フィールドは
                  クエリ文字列に連結して1つの match 節に入れる。
    idfweighted : bm25_fielded_weighted_posboost + IDF項重み付け。
                  recall@1000 基準の現行最良（consensus 0.3176 / coverage 0.6208）。
                  疑似文書の語は連結せず、語ごとの重み付き match 節として入れるので、
                  narrative/Subquery だけが「連結される側」になる。

  どちらの構成でも検索側のパラメータは固定する
  （title/headings/body 均等重み、span_end=100, span_boost=15, RETRIEVE_K=1000, RRF k=60）。
  1回の実行内で変えるのはクエリの組み立て方（narrative/Subqueryの繰り返し回数）だけ。

  なお位置ブーストの判定語 span_terms は analyze_terms(Subquery) で作っており、
  analyze_terms は重複を除去するので、Subquery を何回繰り返しても span_terms は変わらない。
  つまり本実験で動くのは match 節側の項頻度(TF)だけである。

使い方:
    python evaluate_subquery_repeat.py                  # 105トピック全部
    python evaluate_subquery_repeat.py --limit 5        # 動作確認
    python evaluate_subquery_repeat.py --retrieval idfweighted
    python evaluate_subquery_repeat.py --model gpt56terra
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, bm25_equalweight_posboost_discourseboost,
                       bm25_fielded_weighted_posboost, analyze_terms, rrf_fuse)
from term_weights import idf_term_weights

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

WEBSTYLE_FILES = {
    "gemini": os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json"),
    "gpt56terra": os.path.join(RAG_DIR,
                               "multi_query2doc_decomposed_webstyle_gpt56terra_L200.json"),
}

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
SPAN_END = 100
SPAN_BOOST = 15.0

# (条件名, narrativeの繰り返し回数, Subqueryの繰り返し回数)
GRID = [
    ("n5_s1", 5, 1),   # 現 champion
    ("n5_s5", 5, 5),
    ("n0_s1", 0, 1),
    ("n0_s5", 0, 5),
]
METHODS = ["baseline"] + [name for name, _, _ in GRID]

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


def fuse(lists, topk):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


class Cache:
    def __init__(self, fn):
        self.fn = fn
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, key, *args, **kwargs):
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self._cache[key] = self.fn(*args, **kwargs)
        return self._cache[key]

    def stats(self, label):
        return f"  {label}: 検索 {self.calls} 回 / キャッシュヒット {self.hits} 回"


def prepare(webstyle, qids, retrieval):
    """{qid: [(Subquery, structured_doc, span_terms, idf重み or None), ...]}"""
    out = {}
    for qid in qids:
        entry = webstyle[qid]
        pairs = []
        for dq, doc in zip(entry.get("decomposed_queries") or [],
                           entry.get("query2doc_docs_structured") or []):
            if not doc or not doc.get("body"):
                continue
            weights = idf_term_weights(doc) if retrieval == "idfweighted" else None
            pairs.append((dq, doc, analyze_terms(dq, field="body"), weights))
        out[qid] = pairs
    return out


def build_run(method, qids, queries, prepared, search_body, search_fielded, retrieval):
    run, t0 = {}, time.time()
    n_rep = s_rep = None
    for name, n, s in GRID:
        if name == method:
            n_rep, s_rep = n, s

    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        if method == "baseline":
            run[qid] = fuse([search_body(q.strip(), q, k=RETRIEVE_K)], TOPK)
        elif n_rep is not None:
            narr = " ".join([q] * n_rep) if n_rep else ""
            lists = []
            for dq, doc, span_terms, weights in prepared[qid]:
                sub = " ".join([dq] * s_rep)
                prefix = f"{narr} {sub}".strip()
                if retrieval == "champion":
                    title_q = f"{prefix} {doc['title']}"
                    headings_q = f"{prefix} {' '.join(doc['headings'])}"
                    body_q = f"{prefix} {doc['body']}"
                    lists.append(search_fielded(
                        (title_q, headings_q, body_q, tuple(span_terms)),
                        title_q, headings_q, body_q, span_terms, k=RETRIEVE_K,
                        span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[]))
                else:
                    key = (prefix, tuple(weights["title"]), tuple(weights["headings"]),
                           tuple(weights["body"]), tuple(span_terms))
                    lists.append(search_fielded(
                        key, prefix, weights["title"], weights["headings"],
                        weights["body"], span_terms, k=RETRIEVE_K,
                        title_boost=1, headings_boost=1, body_boost=1,
                        span_end=SPAN_END, span_boost=SPAN_BOOST))
            run[qid] = fuse(lists, TOPK)
        else:
            raise ValueError(f"未知の method: {method}")

        if i % 10 == 0 or i == len(qids):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(WEBSTYLE_FILES), default="gemini")
    ap.add_argument("--retrieval", choices=["champion", "idfweighted"], default="champion",
                    help="検索構成。champion=nDCG基準の最良、idfweighted=recall基準の最良")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"読み込み中... model={args.model}")
    with open(WEBSTYLE_FILES[args.model]) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
        and any(d and d.get("body") for d in webstyle[qid].get("query2doc_docs_structured", []))
    )
    if args.limit:
        valid_qids = valid_qids[:args.limit]

    print("span_terms を準備中...")
    prepared = prepare(webstyle, valid_qids, args.retrieval)
    valid_qids = [q for q in valid_qids if prepared[q]]
    n_docs = sum(len(v) for v in prepared.values())
    print(f"  {len(valid_qids)} クエリ / 疑似文書 {n_docs} 本")
    print(f"条件: {', '.join(METHODS)}   検索構成: {args.retrieval}")
    print(f"TOPK={TOPK} RETRIEVE_K={RETRIEVE_K} SPAN_END={SPAN_END} SPAN_BOOST={SPAN_BOOST}")
    print("=" * 72)

    search_body = Cache(bm25_body)
    fn = (bm25_equalweight_posboost_discourseboost if args.retrieval == "champion"
          else bm25_fielded_weighted_posboost)
    search_fielded = Cache(fn)

    runs = {m: build_run(m, valid_qids, queries, prepared, search_body, search_fielded,
                         args.retrieval)
            for m in METHODS}
    print(search_body.stats("bm25_body"))
    print(search_fielded.stats(fn.__name__))

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in METHODS)]
    if len(eval_qids) < len(valid_qids):
        print(f"[warn] 検索結果が空の {len(valid_qids) - len(eval_qids)} クエリを全条件から除外",
              file=sys.stderr)
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    summary = {}
    for method in METHODS:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            summary.setdefault(name, {})[method] = evaluate(
                runs[method], qrels[name], eval_qids, f"{method} / {name}")

    print("\n" + "=" * 72)
    print(f"SUMMARY  model={args.model} retrieval={args.retrieval}  対象 {len(eval_qids)} クエリ")
    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    for name in QREL_SETS:
        print(f"\n[{name}]")
        header = "method".ljust(12) + "".join(labels[k].ljust(17) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(12)
            if agg is None:
                print(row + "-")
                continue
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method != "baseline" and base:
                    cell += f" ({agg[key] - base[key]:+.4f})"
                row += cell.ljust(17)
            print(row)

        s = summary[name]
        if all(s.get(k) for k in ("n5_s1", "n5_s5", "n0_s1", "n0_s5")):
            print(f"\n  [{name} / 2x2] Subquery を1回→5回にしたときの改善幅")
            for key in METRIC_KEYS:
                d_with = s["n5_s5"][key] - s["n5_s1"][key]
                d_without = s["n0_s5"][key] - s["n0_s1"][key]
                print(f"    {labels[key]:14s} narrativeあり {d_with:+.4f} / "
                      f"narrativeなし {d_without:+.4f}")
            print(f"  [{name} / 2x2] narrative を外したときの変化")
            for key in METRIC_KEYS:
                d_s1 = s["n0_s1"][key] - s["n5_s1"][key]
                d_s5 = s["n0_s5"][key] - s["n5_s5"][key]
                print(f"    {labels[key]:14s} Subquery×1 {d_s1:+.4f} / "
                      f"Subquery×5 {d_s5:+.4f}")
    print("=" * 72)

    out_path = os.path.join(RAG_DIR, f"subquery_repeat_2x2_{args.model}_{args.retrieval}.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "retrieval": args.retrieval, "n_queries": len(eval_qids), "topk": TOPK,
                   "retrieve_k": RETRIEVE_K, "span_end": SPAN_END,
                   "span_boost": SPAN_BOOST, "grid": GRID,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
