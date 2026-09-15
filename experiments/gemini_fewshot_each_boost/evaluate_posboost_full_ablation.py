"""
位置ブースト(posboost)の寄与を、全105トピックで切り分ける「フル実験」。

背景:
  発表資料（2026-09-07）で「位置ブーストだけでも consensus nDCG@10 = 0.4418 とかなり強く、
  構造化疑似文書を足した提案手法が 0.5007」という内訳を示したが、これは
  evaluate_long_unstructured_posboost.py が `valid_qids[::3]` で35トピックに間引いて
  実行した値だった（計算時間の都合）。他の主要な結果はすべて105トピックなので、
  この内訳だけトピック数が揃っていない。本スクリプトはその穴を埋める。

条件（すべて n=105、gemini疑似文書、QUERY_REPEAT=5、RETRIEVE_K=1000、
      span_end=100、span_boost=15、RRF k=60）:

  baseline        : narrative そのまま bm25_body（トピックにつき1検索、RRFなし）
                    参照用。他の主要な表と同じ baseline。
  long_nopos      : narrative×5 + Subquery を単一の body フィールドへ。位置ブーストなし。
                    疑似文書は一切使わない。「長いが構造化されていない」クエリ。
  long_pos        : 上に位置ブーストだけを足したもの。
                    → (long_nopos → long_pos) が「位置ブースト単体の寄与」
  long_pos_fielded: narrative×5 + Subquery を title/headings/body へ「それぞれ同じ文字列で」
                    フィールド別に match（均等重み）＋位置ブースト。疑似文書は使わない。
                    → (long_pos → long_pos_fielded) が「インデックス側のフィールド構造の寄与」
  champion        : 構造化疑似文書を title/headings/body へフィールド別に match（均等重み）
                    ＋位置ブースト。現行の最良設定。
                    → (long_pos_fielded → champion) が「LLM生成疑似文書そのものの寄与」

  long_pos_fielded を入れる理由: long_pos と champion の間では「1フィールド→3フィールド」と
  「疑似文書の追加」が同時に変わるため、LLM が本当に必要なのかを判定できない。
  この条件は LLM を一切使わずにフィールド構造だけを champion に揃えた最も厳しい対照であり、
  ここで champion に並ばれるなら「疑似文書は不要」という結論になる。

  なお msmarco-v21-doc の body は title と headings のテキストを（冒頭に）物理的に含む
  入れ子構造なので（2026-09-09 実測: titleは100%、headings先頭行も100%がbodyに出現）、
  3フィールドへの個別 match は「新しい情報を足す」のではなく
  「title/headings に出る語に最大3倍の重みを与える」操作に近い。

  long_* は疑似文書を使わないので LLM 生成が不要な条件である。つまり
  (baseline → long_pos) の改善は LLM を一切使わずに得られる分であり、
  (long_pos → champion) が LLM 生成疑似文書を導入して初めて得られる分になる。

使い方:
    python evaluate_posboost_full_ablation.py            # 105トピック
    python evaluate_posboost_full_ablation.py --limit 5  # 動作確認
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, bm25_bodyonly_posboost_discourseboost,
                       bm25_equalweight_posboost_discourseboost, analyze_terms, rrf_fuse)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
SPAN_END, SPAN_BOOST = 100, 15.0

METHODS = ["baseline", "long_nopos", "long_pos", "long_pos_fielded", "champion"]

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


def prepare(webstyle, qids):
    """{qid: [(Subquery, structured_doc, span_terms), ...]}"""
    out = {}
    for qid in qids:
        entry = webstyle[qid]
        pairs = []
        for dq, doc in zip(entry.get("decomposed_queries") or [],
                           entry.get("query2doc_docs_structured") or []):
            if not doc or not doc.get("body"):
                continue
            pairs.append((dq, doc, analyze_terms(dq, field="body")))
        out[qid] = pairs
    return out


def build_run(method, qids, queries, prepared):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        repeated_q = " ".join([q] * QUERY_REPEAT)

        if method == "baseline":
            run[qid] = fuse([bm25_body(q, k=RETRIEVE_K)], TOPK)
        else:
            lists = []
            for dq, doc, span_terms in prepared[qid]:
                if method == "long_nopos":
                    # narrative×5 + Subquery のみ。疑似文書も位置ブーストも使わない
                    lists.append(bm25_body(f"{repeated_q} {dq}", k=RETRIEVE_K))
                elif method == "long_pos":
                    # 同じクエリに位置ブーストだけを追加
                    lists.append(bm25_bodyonly_posboost_discourseboost(
                        f"{repeated_q} {dq}", span_terms, k=RETRIEVE_K,
                        span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[]))
                elif method == "long_pos_fielded":
                    # 疑似文書は使わず、同じ文字列を3フィールドへ個別に match
                    lt = f"{repeated_q} {dq}"
                    lists.append(bm25_equalweight_posboost_discourseboost(
                        lt, lt, lt, span_terms, k=RETRIEVE_K,
                        span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[]))
                elif method == "champion":
                    lists.append(bm25_equalweight_posboost_discourseboost(
                        f"{repeated_q} {dq} {doc['title']}",
                        f"{repeated_q} {dq} {' '.join(doc['headings'])}",
                        f"{repeated_q} {dq} {doc['body']}",
                        span_terms, k=RETRIEVE_K,
                        span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[]))
                else:
                    raise ValueError(f"未知の method: {method}")
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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="",
                    help="カンマ区切りで指定した条件だけ実行し、既存の結果JSONへマージする")
    args = ap.parse_args()

    methods = [m.strip() for m in args.only.split(",") if m.strip()] or list(METHODS)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"未知の条件: {unknown}（選べるのは {METHODS}）")

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

    print("span_terms を準備中...")
    prepared = prepare(webstyle, valid_qids)
    valid_qids = [q for q in valid_qids if prepared[q]]
    print(f"  {len(valid_qids)} クエリ / 疑似文書 {sum(len(v) for v in prepared.values())} 本")
    print(f"条件: {', '.join(methods)}")
    print(f"QUERY_REPEAT={QUERY_REPEAT} RETRIEVE_K={RETRIEVE_K} "
          f"SPAN_END={SPAN_END} SPAN_BOOST={SPAN_BOOST}")
    print("=" * 72)

    runs = {m: build_run(m, valid_qids, queries, prepared) for m in methods}

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in methods)]
    if len(eval_qids) < len(valid_qids):
        print(f"[warn] 検索結果が空の {len(valid_qids) - len(eval_qids)} クエリを全条件から除外",
              file=sys.stderr)
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    # --only 実行のときは既存結果を読み込んでマージする
    out_path = os.path.join(RAG_DIR, "posboost_full_ablation_result.json")
    summary = {}
    if args.only and os.path.exists(out_path):
        prev = json.load(open(out_path))
        summary = prev.get("summary", {})
        print(f"既存結果にマージ: {out_path}（既存条件 "
              f"{sorted(set(k for v in summary.values() for k in v))}）")

    for method in methods:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            summary.setdefault(name, {})[method] = evaluate(
                runs[method], qrels[name], eval_qids, f"{method} / {name}")

    # 表示は既存条件も含めて METHODS の順で
    show = [m for m in METHODS if any(summary.get(qr, {}).get(m) for qr in QREL_SETS)]

    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    print("\n" + "=" * 72)
    print(f"SUMMARY  対象 {len(eval_qids)} クエリ")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        header = "method".ljust(14) + "".join(labels[k].ljust(17) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in show:
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

        s = summary[name]
        need = ("long_nopos", "long_pos", "champion")
        if all(s.get(k) for k in need):
            has_f = bool(s.get("long_pos_fielded"))
            print(f"\n  [{name}] 寄与の内訳（long_nopos を起点）")
            for key in METRIC_KEYS:
                total = s["champion"][key] - s["long_nopos"][key]
                d_pos = s["long_pos"][key] - s["long_nopos"][key]
                pct = lambda d: (d / total * 100) if abs(total) > 1e-9 else float("nan")
                if has_f:
                    d_fld = s["long_pos_fielded"][key] - s["long_pos"][key]
                    d_doc = s["champion"][key] - s["long_pos_fielded"][key]
                    print(f"    {labels[key]:14s} 位置ブースト {d_pos:+.4f} ({pct(d_pos):3.0f}%) / "
                          f"フィールド構造 {d_fld:+.4f} ({pct(d_fld):3.0f}%) / "
                          f"疑似文書 {d_doc:+.4f} ({pct(d_doc):3.0f}%)")
                else:
                    d_doc = s["champion"][key] - s["long_pos"][key]
                    print(f"    {labels[key]:14s} 位置ブースト {d_pos:+.4f} ({pct(d_pos):3.0f}%) / "
                          f"フィールド構造+疑似文書 {d_doc:+.4f} ({pct(d_doc):3.0f}%)")
    print("=" * 72)

    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK, "retrieve_k": RETRIEVE_K,
                   "query_repeat": QUERY_REPEAT, "span_end": SPAN_END,
                   "span_boost": SPAN_BOOST, "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
