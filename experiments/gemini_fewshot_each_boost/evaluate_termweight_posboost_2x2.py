"""
「項(term)重み付け」と「位置ブースト(posboost)」を組み合わせたときに効果が足し合わさるかを
2x2 で検証する（doc版）。

背景:
  これまで両者は別々の系列で評価されており、一度も組み合わせて測られていなかった。
    - 位置ブースト側: evaluate_decomposed_webstyle_equalweight_ablation.py
      （旧名 notitle。title/headings/body 均等重み + span_first。consensus nDCG@10=0.5251）
    - 項重み付け側: evaluate_decomposed_webstyle_termweighted_tsweep.py
      （フィールド重み recallopt(2,1,1) + confidence重み。coverage nDCG@10=0.4297 が最高値）
  後者が coverage nDCG で最高値を出しているのに位置ブーストと併用されていないため、
  「足し合わさるのか、それとも同じ情報を二重に使っているだけなのか」が未解決だった。

  なお両者は疑似文書の生成モデルも違っていた（位置ブースト側=gemini-3.7-flash、
  項重み付け側=gpt-5.6-terra）ため、これまでの数値の直接比較にはモデル交絡があった。
  本スクリプトは1回の実行内でモデルを固定するので、その交絡は入らない。

2x2 の設計:
  4条件すべてを同一の検索関数 bm25_fielded_weighted_posboost で回し、
  「重みの値」と「span_boost」だけを変える。
    tw_uniform            : 全語 weight=1.0、位置ブーストなし
    tw_uniform_posboost   : 全語 weight=1.0、位置ブーストあり
    tw_weighted           : 項重み付けあり、位置ブーストなし
    tw_weighted_posboost  : 項重み付けあり、位置ブーストあり   <- 本命
  bm25_fielded(フィールド全体を1つのmatch節にまとめる)を対照に使わないのは、
  term単位に分解するとTFの効き方が変わり、重み付け以外の差が混入するため。

  参考行として、既存championそのもの(bm25_equalweight_posboost_discourseboost(markers=[]))
  も同時に回す。既知値と一致すれば環境・データが再現できていることの確認になる。

項重み付けの方式（--weighting）:
  idf        : 実コーパスのIDFのみから重みを作る（term_weights.idf_term_weights）。
               生成時のlogprobが不要なので gemini-3.7-flash の疑似文書でも使える。
  confidence : 生成時のtoken logprobをsoftmaxした重み（term_weights.extract_term_weights,
               sign=1）。logprobを返すモデルの疑似文書でのみ使える。
               2026-09-09時点で gemini-3.7-flash はlogprobsがnullのため gpt-5.6-terra 専用。

使い方:
    # 主実験: gemini-3.7-flash の疑似文書 + IDF項重み付け
    python evaluate_termweight_posboost_2x2.py --model gemini --weighting idf

    # 副実験: gpt-5.6-terra の疑似文書 + confidence項重み付け
    python evaluate_termweight_posboost_2x2.py --model gpt56terra --weighting confidence

    # 動作確認（先頭5トピックだけ）
    python evaluate_termweight_posboost_2x2.py --model gemini --weighting idf --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, bm25_fielded_weighted_posboost,
                       bm25_equalweight_posboost_discourseboost, analyze_terms, rrf_fuse)
from term_weights import extract_term_weights, idf_term_weights, uniform_term_weights

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
RETRIEVE_K = 1000        # championと揃える（項重み付け側の既存実験は3000だったので直接比較不可）
SPAN_END = 100           # championと同じ
SPAN_BOOST = 15.0        # championと同じ
TEMPERATURE = 0.2        # confidence方式のsoftmax温度。tsweepでcoverage nDCG最良だった値

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]

# (条件名, 項重み付けを使うか, 位置ブーストを使うか)
GRID = [
    ("tw_uniform", False, False),
    ("tw_uniform_posboost", False, True),
    ("tw_weighted", True, False),
    ("tw_weighted_posboost", True, True),
]
METHODS = ["baseline", "champion_ref"] + [name for name, _, _ in GRID]


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4 or line.startswith("#"):
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
    """検索関数の結果をキャッシュする薄いラッパ。キーは引数から作る。"""

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


def build_entries(webstyle, qid, weighting, temperature):
    """1トピック分の (Subquery, structured_doc, 重み付きterm, span_terms) を返す。
    重みの計算はトピックにつき1度で済ませ、4条件で使い回す。"""
    entry = webstyle[qid]
    docs = entry.get("query2doc_docs_structured") or []
    dqs = entry.get("decomposed_queries") or []
    logprobs = entry.get("query2doc_token_logprobs") or [None] * len(docs)
    out = []
    for dq, doc, tokens in zip(dqs, docs, logprobs):
        if not doc or not doc.get("body"):
            continue
        uniform = uniform_term_weights(doc)
        if weighting == "idf":
            weighted = idf_term_weights(doc)
        else:
            if not tokens:
                continue  # logprobが無い疑似文書はconfidence方式で使えない
            raw = "".join(t["token"] for t in tokens)
            weighted = extract_term_weights(raw, doc, tokens,
                                            temperature=temperature, sign=1)
        out.append((dq, doc, uniform, weighted, analyze_terms(dq, field="body")))
    return out


def build_run(method, qids, queries, webstyle, prepared, query_repeat,
              search_body, search_weighted, search_champion):
    run, t0 = {}, time.time()
    use_weighted, use_posboost = None, None
    for name, w, p in GRID:
        if name == method:
            use_weighted, use_posboost = w, p

    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        repeated_q = " ".join([q] * query_repeat)

        if method == "baseline":
            run[qid] = fuse([search_body(q.strip(), q, k=RETRIEVE_K)], TOPK)
        elif method == "champion_ref":
            lists = []
            for dq, doc, _, _, span_terms in prepared[qid]:
                title_q = f"{repeated_q} {dq} {doc['title']}"
                headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
                body_q = f"{repeated_q} {dq} {doc['body']}"
                lists.append(search_champion(
                    (title_q, headings_q, body_q, tuple(span_terms)),
                    title_q, headings_q, body_q, span_terms, k=RETRIEVE_K,
                    span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[]))
            run[qid] = fuse(lists, TOPK)
        elif use_weighted is not None:
            lists = []
            for dq, doc, uniform, weighted, span_terms in prepared[qid]:
                w = weighted if use_weighted else uniform
                plain_text = f"{repeated_q} {dq}"
                terms = span_terms if use_posboost else []
                key = (plain_text, tuple(w["title"]), tuple(w["headings"]),
                       tuple(w["body"]), tuple(terms))
                lists.append(search_weighted(
                    key, plain_text, w["title"], w["headings"], w["body"], terms,
                    k=RETRIEVE_K, title_boost=1, headings_boost=1, body_boost=1,
                    span_end=SPAN_END,
                    span_boost=SPAN_BOOST if use_posboost else 0.0))
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
        return None, 0
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(WEBSTYLE_FILES), default="gemini")
    ap.add_argument("--weighting", choices=["idf", "confidence"], default="idf")
    ap.add_argument("--query-repeat", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="先頭Nトピックだけ回す（動作確認用）")
    args = ap.parse_args()

    if args.weighting == "confidence" and args.model == "gemini":
        raise SystemExit(
            "gemini-3.7-flash の疑似文書には token logprob が無いため confidence 方式は"
            "使えません（--weighting idf を使うか、--model gpt56terra を指定してください）。")

    webstyle_file = WEBSTYLE_FILES[args.model]
    print(f"読み込み中... model={args.model} weighting={args.weighting}")
    with open(webstyle_file) as f:
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
    print(f"疑似文書がある {len(valid_qids)} クエリ")

    print(f"条件: {', '.join(METHODS)}")
    print(f"TOPK={TOPK} RETRIEVE_K={RETRIEVE_K} QUERY_REPEAT={args.query_repeat} "
          f"SPAN_END={SPAN_END} SPAN_BOOST={SPAN_BOOST} TEMPERATURE={TEMPERATURE}")
    print("=" * 72)

    print("項重みとspan_termsを準備中...")
    t0 = time.time()
    prepared = {}
    for i, qid in enumerate(valid_qids, 1):
        prepared[qid] = build_entries(webstyle, qid, args.weighting, TEMPERATURE)
        if i % 10 == 0 or i == len(valid_qids):
            print(f"\r  {i}/{len(valid_qids)} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    n_docs = sum(len(v) for v in prepared.values())
    print(f"  疑似文書 {n_docs} 本、平均 {n_docs / max(len(valid_qids), 1):.1f} 本/トピック")
    valid_qids = [q for q in valid_qids if prepared[q]]

    search_body = Cache(bm25_body)
    search_weighted = Cache(bm25_fielded_weighted_posboost)
    search_champion = Cache(bm25_equalweight_posboost_discourseboost)

    runs = {}
    for m in METHODS:
        runs[m] = build_run(m, valid_qids, queries, webstyle, prepared, args.query_repeat,
                            search_body, search_weighted, search_champion)
    print(search_body.stats("bm25_body"))
    print(search_weighted.stats("bm25_fielded_weighted_posboost"))
    print(search_champion.stats("bm25_equalweight_posboost_discourseboost"))

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in METHODS)]
    if len(eval_qids) < len(valid_qids):
        print(f"[warn] 検索結果が空の {len(valid_qids) - len(eval_qids)} クエリを全条件から除外",
              file=sys.stderr)
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    summary = {}
    for method in METHODS:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            agg, _ = evaluate(runs[method], qrels[name], eval_qids, f"{method} / {name}")
            summary.setdefault(name, {})[method] = agg

    print("\n" + "=" * 72)
    print(f"SUMMARY  model={args.model} weighting={args.weighting}  "
          f"対象 {len(eval_qids)} クエリ")
    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    for name in QREL_SETS:
        print(f"\n[{name}]")
        header = "method".ljust(24) + "".join(labels[k].ljust(17) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(24)
            if agg is None:
                print(row + "-")
                continue
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method != "baseline" and base:
                    cell += f" ({agg[key] - base[key]:+.4f})"
                row += cell.ljust(17)
            print(row)

        # 2x2 の交互作用: 位置ブーストの効果が、項重み付けの有無でどう変わるか
        s = summary[name]
        if all(s.get(k) for k in ("tw_uniform", "tw_uniform_posboost",
                                  "tw_weighted", "tw_weighted_posboost")):
            print(f"\n  [{name} / 交互作用] posboostによる改善幅")
            for key in METRIC_KEYS:
                d_plain = s["tw_uniform_posboost"][key] - s["tw_uniform"][key]
                d_weighted = s["tw_weighted_posboost"][key] - s["tw_weighted"][key]
                print(f"    {labels[key]:14s} 重み付けなし {d_plain:+.4f} / "
                      f"重み付けあり {d_weighted:+.4f}  (差 {d_weighted - d_plain:+.4f})")
    print("=" * 72)

    out_path = os.path.join(
        RAG_DIR, f"termweight_posboost_2x2_{args.model}_{args.weighting}_"
                 f"rep{args.query_repeat}.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "weighting": args.weighting,
                   "n_queries": len(eval_qids), "topk": TOPK, "retrieve_k": RETRIEVE_K,
                   "query_repeat": args.query_repeat, "span_end": SPAN_END,
                   "span_boost": SPAN_BOOST, "temperature": TEMPERATURE,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
