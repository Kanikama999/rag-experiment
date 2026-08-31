"""
decomposed_query2doc_expansion.py（自由文1段落）と
decomposed_query2doc_webstyle_expansion.py（title/headings/bodyのwebページ風JSON）で
生成した疑似文書を比較する。

いずれも Decomposed_Query（narrativeは使わない）+ 疑似文書 を連結して検索し、
トピック内の全 Decomposed_Query 分を RRF 融合する（evaluate_decomposed_variants.py の
dq_pseudodoc と同じ組み立て方）。

条件:
- baseline: narrativeそのままで bm25_body
- plain: Decomposed_Query + 自由文1段落の疑似文書 で bm25_body
- webstyle: Decomposed_Query + (title+headings+bodyを連結した)疑似文書 で bm25_body
- webstyle_keyterms: webstyleと同じ疑似文書だが、bm25_keyterms
  （title^3, headings^2, body^1のフィールドブースト）で検索
- webstyle_narrative_body: narrative（QUERY_REPEAT回繰り返し）+ Decomposed_Query +
  webstyle疑似文書 を連結して bm25_body で検索
- webstyle_narrative_keyterms: 上と同じ連結クエリを bm25_keyterms で検索
- webstyle_ctx_narrative_keyterms: webstyle_narrative_keyterms と同じ組み立てだが、
  疑似文書を decomposed_query2doc_webstyle_ctx_expansion.py の出力（生成プロンプトに
  narrative文脈を見せて書かせたもの）に差し替えたもの

新規のLLM生成は不要。疑似文書ファイルが既に生成済みであることが前提。

使い方:
    python evaluate_decomposed_webstyle.py [QUERY_REPEAT]   # 省略時は5
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import bm25_body, bm25_keyterms, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
PLAIN_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_L200.json")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
WEBSTYLE_CTX_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_ctx_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # narrativeをN回繰り返してから連結する

METHODS = ["baseline", "plain", "webstyle", "webstyle_keyterms",
           "webstyle_narrative_body", "webstyle_narrative_keyterms",
           "webstyle_ctx_narrative_keyterms"]

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


def fuse(lists, topk):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


def dq_pairs(entry):
    """(Decomposed_Query, 疑似文書) のペアを、欠落文書を除いて返す。"""
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"], entry["query2doc_docs"])
            if doc and doc.strip()]


def build_run(method, qids, queries, plain, webstyle, webstyle_ctx, search_body, search_keyterms):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]

        if method == "baseline":
            lists = [search_body(q)]
        elif method == "plain":
            pairs = dq_pairs(plain[qid])
            lists = [search_body(f"{dq} {doc}") for dq, doc in pairs]
        elif method == "webstyle":
            pairs = dq_pairs(webstyle[qid])
            lists = [search_body(f"{dq} {doc}") for dq, doc in pairs]
        elif method == "webstyle_keyterms":
            pairs = dq_pairs(webstyle[qid])
            lists = [search_keyterms(f"{dq} {doc}") for dq, doc in pairs]
        elif method == "webstyle_narrative_body":
            pairs = dq_pairs(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = [search_body(f"{repeated_q} {dq} {doc}") for dq, doc in pairs]
        elif method == "webstyle_narrative_keyterms":
            pairs = dq_pairs(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = [search_keyterms(f"{repeated_q} {dq} {doc}") for dq, doc in pairs]
        elif method == "webstyle_ctx_narrative_keyterms":
            pairs = dq_pairs(webstyle_ctx[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = [search_keyterms(f"{repeated_q} {dq} {doc}") for dq, doc in pairs]
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


def main():
    print("読み込み中...")
    if not os.path.exists(WEBSTYLE_FILE):
        raise SystemExit(
            f"{WEBSTYLE_FILE} がありません。先に "
            f"`python decomposed_query2doc_webstyle_expansion.py submit` → "
            f"`fetch` で生成してください。")
    with open(PLAIN_FILE) as f:
        plain = json.load(f)["results"]
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    webstyle_ctx = {}
    if os.path.exists(WEBSTYLE_CTX_FILE):
        with open(WEBSTYLE_CTX_FILE) as f:
            webstyle_ctx = json.load(f)["results"]
    elif "webstyle_ctx_narrative_keyterms" in METHODS:
        print(f"[warn] {WEBSTYLE_CTX_FILE} が無いので webstyle_ctx_narrative_keyterms を除外します",
              file=sys.stderr)
        METHODS.remove("webstyle_ctx_narrative_keyterms")
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    common = set(plain) & set(webstyle)
    if webstyle_ctx:
        common &= set(webstyle_ctx)
    valid_qids = sorted(
        qid for qid in common
        if qid in queries
        and plain[qid].get("decomposed_queries")
        and any(d and d.strip() for d in plain[qid].get("query2doc_docs", []))
        and any(d and d.strip() for d in webstyle[qid].get("query2doc_docs", []))
        and (not webstyle_ctx or any(d and d.strip() for d in webstyle_ctx[qid].get("query2doc_docs", [])))
    )
    print(f"\n疑似文書が揃っている {len(valid_qids)} クエリ "
          f"（plain={len(plain)}, webstyle={len(webstyle)}, webstyle_ctx={len(webstyle_ctx)}）")
    for name in QREL_SETS:
        print(f"  qrels[{name}]: {len(qrels[name])} qids / "
              f"うち対象内 {len(set(valid_qids) & set(qrels[name]))} qids")
    if not valid_qids:
        print("採点対象が空。両方の疑似文書ファイルの生成状況を確認すること。",
              file=sys.stderr)
        return

    print(f"条件: {', '.join(METHODS)}   TOPK={TOPK}   RETRIEVE_K={RETRIEVE_K}   "
          f"QUERY_REPEAT={QUERY_REPEAT}")
    print("=" * 72)

    search_body = CachedSearch(bm25_body, RETRIEVE_K)
    search_keyterms = CachedSearch(bm25_keyterms, RETRIEVE_K)
    runs = {m: build_run(m, valid_qids, queries, plain, webstyle, webstyle_ctx, search_body, search_keyterms)
            for m in METHODS}
    print(f"\n{search_body.stats('bm25_body')}")
    print(search_keyterms.stats('bm25_keyterms'))

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
        header = "method".ljust(20) + "".join(
            labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(20)
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

    out_path = os.path.join(RAG_DIR, f"decomposed_query2doc_webstyle_eval_summary_rep{QUERY_REPEAT}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
