"""
IDF項重み付けの「強さ」を指数 alpha で制御し、最適点を探索する。

問題意識:
  bm25_fielded_weighted_posboost に IDF 重み（weight = idf/平均idf）を boost として渡すと、
  BM25 がスコア計算時に既に idf を掛けているため、実効的な重みは **idf の2乗**になる。
  BM25 の idf は Robertson-Spärck Jones 重みの導出を持つ量だが、**それを2乗する導出は
  存在しない**。つまり従来の設定（実質 alpha=1）は理論的に正当化された値ではなく、
  試した2点（重み無し / IDF重み）のうちの片方にすぎなかった。

  一方で、クエリ側に項ごとの重みを置くこと自体は BM25 の定義に含まれる
  （完全形の BM25 は score = Σ_t [クエリ側の重み] × idf(t) × [文書側のtf正規化] で、
   クエリ側の重みには通常クエリ内項頻度の飽和項が入る。narrative×5 の繰り返しが
   効くのはこのスロットを操作しているため）。また拡張語に重みを付けること自体は
   Rocchio / RM3 など擬似適合性フィードバックの標準的な実務である。
  したがって争点は「IDF で重み付けすること」ではなく「その強さ」である。

  weight ∝ (idf/平均idf)**alpha を平均1に再正規化して使う（alpha は語間の強弱だけを変え、
  重みの総量は変えない）。実効的な重みは idf**(1+alpha) になるので:
      alpha=0   → idf**1  素のBM25と同じ（全語が重み1）
      alpha=1   → idf**2  従来の設定
## 2026-09-09 の実測結果（105トピック）

  alpha    consensus R@1000 / nDCG@10    coverage R@1000 / nDCG@10
  0        0.3095 / 0.5257               0.5984 / 0.4049
  0.25     0.3114 / 0.5245               0.6026 / 0.4037
  0.5      0.3131 / 0.5243               0.6078 / 0.4061
  0.75     0.3147 / 0.5256               0.6095 / 0.4123
  1        0.3160 / 0.5251               0.6119 / 0.4144
  1.5      0.3185 / 0.5271               0.6182 / 0.4266   ← 8セル全てで最良

  **alpha=1.5 が両qrels×全4指標のすべてで最良**だった。しかもグリッドの端なので
  最適点には到達していない（延長スイープは --alphas 2,3,5 --tag _extend で実施）。

  当初は「alpha=1 は強すぎるので中間に最適点がある」と予想していたが、逆だった。
  正しく正規化すると alpha は 1.5 まで単調に効き続ける。

## 重要: 旧 idf_term_weights には正規化のバグがあった

  alpha 引数を追加する際に判明した。旧実装は mean_idf を「ストップワードを含む全語」で
  計算しておきながら、重みは「idf>0 の語」にだけ配っていた。そのため残った語の重みの
  平均が1ではなく 1.25 程度に膨らみ、IDF重み付けの条件だけ総ブースト量が2割超過していた。

  この過剰ブーストのせいで、2x2実験（§7.4）では
      tw_weighted_posboost: recall@1000 0.3176 / nDCG@10 0.5182
  となり「IDFはrecallを上げnDCGを下げる」というトレードオフに見えていた。
  正規化を直した本スクリプトの alpha=1 は 0.3160 / 0.5251 で、nDCG の低下は消える。
  **トレードオフの正体は「IDF強調」ではなく「ブースト量の過剰」だった。**
  §7.4 の当該記述は訂正が必要。

検索構成は tw_weighted_posboost と完全に同一に固定する
（bm25_fielded_weighted_posboost、title/headings/body 均等重み、span_end=100、
 span_boost=15、RETRIEVE_K=1000、QUERY_REPEAT=5、gemini疑似文書、RRF k=60）。
変えるのは alpha だけ。

alpha=0 は既存の tw_uniform_posboost と一致するはずなので、再現確認を兼ねる
（ストップワードは idf=0 で落とすが、english analyzer では検索時にも消えるため
 スコアには影響しない。節数が減って速いだけ）。

使い方:
    python evaluate_idf_alpha_sweep.py
    python evaluate_idf_alpha_sweep.py --alphas 0,0.5,1.0 --limit 5
    python evaluate_idf_alpha_sweep.py --alphas 2,3,5 --tag _extend
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pytrec_eval

from retriever import bm25_body, bm25_fielded_weighted_posboost, analyze_terms, rrf_fuse
from term_weights import idf_term_weights

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
SPAN_END, SPAN_BOOST = 100, 15.0
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

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


def prepare(webstyle, qids, alphas):
    """{qid: [(Subquery, span_terms, {alpha: 重み}), ...]}
    重みは alpha ごとに1度だけ計算して使い回す（IDF自体は term_weights 側でキャッシュ済み）。"""
    out = {}
    for qid in qids:
        entry = webstyle[qid]
        items = []
        for dq, doc in zip(entry.get("decomposed_queries") or [],
                           entry.get("query2doc_docs_structured") or []):
            if not doc or not doc.get("body"):
                continue
            weights = {a: idf_term_weights(doc, alpha=a) for a in alphas}
            items.append((dq, analyze_terms(dq, field="body"), weights))
        out[qid] = items
    return out


def build_run(method, alpha, qids, queries, prepared):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        if method == "baseline":
            run[qid] = fuse([bm25_body(q, k=RETRIEVE_K)], TOPK)
        else:
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = []
            for dq, span_terms, weights in prepared[qid]:
                w = weights[alpha]
                lists.append(bm25_fielded_weighted_posboost(
                    f"{repeated_q} {dq}", w["title"], w["headings"], w["body"],
                    span_terms, k=RETRIEVE_K,
                    title_boost=1, headings_boost=1, body_boost=1,
                    span_end=SPAN_END, span_boost=SPAN_BOOST))
            run[qid] = fuse(lists, TOPK)

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
    ap.add_argument("--alphas", default="",
                    help=f"カンマ区切り。既定 {DEFAULT_ALPHAS}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="",
                    help="出力ファイル名の接尾辞。既存結果を上書きしたくないときに使う")
    args = ap.parse_args()

    alphas = ([float(a) for a in args.alphas.split(",") if a.strip()]
              if args.alphas else list(DEFAULT_ALPHAS))
    methods = ["baseline"] + [f"alpha_{a:g}" for a in alphas]

    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
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

    print(f"重みと span_terms を準備中... (alpha {alphas})")
    t0 = time.time()
    prepared = prepare(webstyle, valid_qids, alphas)
    valid_qids = [q for q in valid_qids if prepared[q]]
    print(f"  {len(valid_qids)} クエリ / 疑似文書 {sum(len(v) for v in prepared.values())} 本 "
          f"({time.time() - t0:.0f}s)")
    print(f"条件: {', '.join(methods)}")
    print(f"RETRIEVE_K={RETRIEVE_K} QUERY_REPEAT={QUERY_REPEAT} "
          f"SPAN_END={SPAN_END} SPAN_BOOST={SPAN_BOOST}")
    print("=" * 72)

    runs = {}
    for m in methods:
        a = None if m == "baseline" else float(m.split("_", 1)[1])
        runs[m] = build_run(m, a, valid_qids, queries, prepared)

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in methods)]
    if len(eval_qids) < len(valid_qids):
        print(f"[warn] 検索結果が空の {len(valid_qids) - len(eval_qids)} クエリを全条件から除外",
              file=sys.stderr)
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    summary = {}
    for method in methods:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            summary.setdefault(name, {})[method] = evaluate(
                runs[method], qrels[name], eval_qids, f"{method} / {name}")

    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    print("\n" + "=" * 72)
    print(f"SUMMARY  対象 {len(eval_qids)} クエリ")
    for name in QREL_SETS:
        print(f"\n[{name}]  実効重み = idf**(1+alpha)")
        header = "method".ljust(14) + "".join(labels[k].ljust(17) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in methods:
            agg = summary[name].get(method)
            row = method.ljust(14)
            if agg is None:
                print(row + "-")
                continue
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method != "baseline" and base:
                    cell += f" ({agg[key] - base[key]:+.4f})"
                row += cell.ljust(17)
            print(row)

        # alpha=0 を起点にした増減（IDF強調の効果そのもの）
        zero = summary[name].get("alpha_0")
        if zero:
            print(f"\n  [{name}] alpha=0 からの変化")
            for method in methods[1:]:
                agg = summary[name].get(method)
                if not agg or method == "alpha_0":
                    continue
                print(f"    {method:12s} " + "  ".join(
                    f"{labels[k]}: {agg[k] - zero[k]:+.4f}" for k in METRIC_KEYS))
        # 指標ごとの最適 alpha
        print(f"\n  [{name}] 指標ごとの最良 alpha")
        for key in METRIC_KEYS:
            cands = [(m, summary[name][m][key]) for m in methods[1:]
                     if summary[name].get(m)]
            if cands:
                best = max(cands, key=lambda x: x[1])
                print(f"    {labels[key]:14s} {best[0]} ({best[1]:.4f})")
    print("=" * 72)

    out_path = os.path.join(RAG_DIR, f"idf_alpha_sweep_result{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"alphas": alphas, "n_queries": len(eval_qids), "topk": TOPK,
                   "retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT,
                   "span_end": SPAN_END, "span_boost": SPAN_BOOST,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
