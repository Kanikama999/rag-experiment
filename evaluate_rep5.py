from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import bm25_body, rrf_fuse

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
HYDE_FILE = os.path.join(DATA_DIR, "multi_hyde_200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30]
QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
QUERY_REPEAT = 5   # 元クエリをN回繰り返してHyDE文書1本と連結してから検索

VARIANTS = ["hydecat"]

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


def build_run(method, qids, queries, hyde, search):
    """1 条件ぶんの run {qid: {docid: score}} を作る。"""
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]

        if method == "baseline":
            lists = [search(q)]
        else:
            variant, k = method.rsplit("_", 1)
            docs = [d for d in hyde[qid]["hyde_results"][f"hyde_{int(k)}"] if d.strip()]
            if variant == "hyde":
                lists = [search(d) for d in docs]
            elif variant == "hydecat":
                repeated_q = " ".join([q] * QUERY_REPEAT)
                lists = [search(f"{repeated_q} {d}") for d in docs]
            elif variant == "hydeq":
                lists = [search(q)] + [search(d) for d in docs]
            else:
                raise ValueError(f"未知の variant: {variant}")

        run[qid] = fuse(lists, TOPK)

        if i % 20 == 0 or i == len(qids):
            print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()
    return run


# ============================================================
# 評価
# ============================================================
def evaluate(run, qrels, eval_qids, label):
    """eval_qids に固定して採点する。対象集合が条件間でズレないことを保証する。"""
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
    """
    qrels と run の docid 粒度が食い違っていないか確認する。
    セグメント ID と文書 ID が混ざっていると全指標が静かに 0 付近に張り付く。
    """
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
    with open(HYDE_FILE) as f:
        hyde = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid, entry in hyde.items()
        if qid in queries
        and all(len(entry.get("hyde_results", {}).get(f"hyde_{k}", [])) == k
                for k in K_VALUES)
    )
    print(f"\nB方式: K={K_VALUES} が全て揃った {len(valid_qids)} クエリ "
          f"（除外 {len(hyde) - len(valid_qids)}）")
    for name in QREL_SETS:
        print(f"  qrels[{name}]: {len(qrels[name])} qids / "
              f"うち対象内 {len(set(valid_qids) & set(qrels[name]))} qids")
    if not valid_qids:
        print("採点対象が空。HyDE ファイルの生成枚数を確認すること。", file=sys.stderr)
        return

    methods = ["baseline"] + [f"{v}_{k}" for k in K_VALUES for v in VARIANTS]
    print(f"条件: {', '.join(methods)}   TOPK={TOPK}   QUERY_REPEAT={QUERY_REPEAT}")
    print("=" * 72)

    search = CachedBM25(TOPK)
    runs = {m: build_run(m, valid_qids, queries, hyde, search) for m in methods}
    print(f"\n{search.stats()}")

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in methods)]
    if len(eval_qids) < len(valid_qids):
        print(f"[warn] 検索結果が空になった {len(valid_qids) - len(eval_qids)} クエリを"
              f"全条件から除外", file=sys.stderr)
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    print("\n--- docid 粒度チェック ---")
    for name in QREL_SETS:
        check_docid_overlap(runs["baseline"], qrels[name], eval_qids, name)

    summary, n_seen = {}, {}
    for method in methods:
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
    print(f"SUMMARY   対象: {len(eval_qids)} クエリ   QUERY_REPEAT={QUERY_REPEAT}")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
                  "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
        header = "method".ljust(12) + "".join(
            labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in methods:
            agg = summary[name].get(method)
            row = method.ljust(12)
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

    out_path = os.path.join(DATA_DIR, f"hyde_eval_summary_rep{QUERY_REPEAT}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK,
                   "query_repeat": QUERY_REPEAT,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
