"""
narrative を Subquery（分解した簡潔な質問文）に分割し、各 Subquery
1問につき Query2doc 疑似文書を1本生成したもの（decomposed_query2doc_expansion.py の出力）を使う評価。

evaluate_rep1.py の query2doc_k と同様、各 Query2doc 疑似文書を narrative（元クエリ）と連結してから
検索し、トピック内の全 Subquery 分を RRF 融合する方式（decomposed_query2doc）。

使い方:
    python evaluate_decomposed_rep1.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import bm25_body, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
QUERY2DOC_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000

METHODS = ["baseline", "decomposed_query2doc"]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


def load_qrels(path):
    qrels, skipped = {}, 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                skipped += 1
                if skipped <= 3:
                    print(f"  [warn] {os.path.basename(path)}:{lineno} "
                          f"列数 {len(parts)} をスキップ", file=sys.stderr)
                continue
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    if skipped:
        print(f"  [warn] {os.path.basename(path)}: 計 {skipped} 行スキップ",
              file=sys.stderr)
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


class CachedBM25:

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

    def stats(self):
        return f"BM25 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回"


def fuse(lists, topk):
    """空リストを除いて RRF 融合する。1 本だけなら順位はそのまま。"""
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


def build_run(method, qids, queries, q2d, search):
    """1 条件ぶんの run {qid: {docid: score}} を作る。"""
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]

        if method == "baseline":
            lists = [search(q)]
        elif method == "decomposed_query2doc":
            docs = [d for d in q2d[qid]["query2doc_docs"] if d and d.strip()]
            lists = [search(f"{q} {d}") for d in docs]
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
        return None, 0

    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}

    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg, n


def check_docid_overlap(run, qrels, eval_qids, label):
    run_ids, qrel_ids = set(), set()
    for qid in eval_qids:
        if qid in qrels:
            run_ids |= set(run.get(qid, {}))
            qrel_ids |= set(qrels[qid])
    if not qrel_ids:
        return
    overlap = len(run_ids & qrel_ids)
    print(f"  [check/{label}] qrels docid {len(qrel_ids)} 件中 {overlap} 件が run に出現")
    if overlap == 0:
        print(f"  [ERROR/{label}] docid が 1 件も一致しない。粒度を確認すること。",
              file=sys.stderr)
        print(f"    qrels 例: {sorted(qrel_ids)[:2]}", file=sys.stderr)
        print(f"    run   例: {sorted(run_ids)[:2]}", file=sys.stderr)


def main():
    print("読み込み中...")
    with open(QUERY2DOC_FILE) as f:
        q2d = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid, entry in q2d.items()
        if qid in queries
        and entry.get("decomposed_queries")
        and all(d and d.strip() for d in entry.get("query2doc_docs", []))
    )
    print(f"\n分解質問すべてでQuery2doc生成が揃った {len(valid_qids)} クエリ "
          f"（除外 {len(q2d) - len(valid_qids)}）")
    for name in QREL_SETS:
        print(f"  qrels[{name}]: {len(qrels[name])} qids / "
              f"うち対象内 {len(set(valid_qids) & set(qrels[name]))} qids")
    if not valid_qids:
        print("採点対象が空。decomposed Query2doc ファイルの生成状況を確認すること。",
              file=sys.stderr)
        return

    print(f"条件: {', '.join(METHODS)}   TOPK={TOPK}")
    print("=" * 72)

    search = CachedBM25(TOPK)
    runs = {m: build_run(m, valid_qids, queries, q2d, search) for m in METHODS}
    print(f"\n{search.stats()}")

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in METHODS)]
    if len(eval_qids) < len(valid_qids):
        print(f"[warn] 検索結果が空になった {len(valid_qids) - len(eval_qids)} クエリを"
              f"全条件から除外", file=sys.stderr)
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    print("\n--- docid 粒度チェック ---")
    for name in QREL_SETS:
        check_docid_overlap(runs["baseline"], qrels[name], eval_qids, name)

    summary, n_seen = {}, {}
    for method in METHODS:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            agg, n = evaluate(runs[method], qrels[name], eval_qids, f"{method} / {name}")
            summary.setdefault(name, {})[method] = agg
            n_seen.setdefault(name, set()).add(n)

    for name in QREL_SETS:
        if len(n_seen[name]) > 1:
            print(f"[ERROR] qrels[{name}] の n が条件間で不一致: {sorted(n_seen[name])}",
                  file=sys.stderr)

    print("\n" + "=" * 72)
    print(f"SUMMARY   対象: {len(eval_qids)} クエリ")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
                  "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
        header = "method".ljust(16) + "".join(
            labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(16)
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

    out_path = os.path.join(RAG_DIR, "decomposed_query2doc_eval_summary.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
