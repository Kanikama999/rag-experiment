"""
オラクル（人間が正解判定した本物のセグメント本文）を、現行パイプラインで測り直す。

## なぜやるか

`experiments/oracle/` は「疑似文書が理想的に書けたらどこまで伸びるか」の上限測定のはず
だったが、**champion がその上限を超えてしまっている**（coverage nDCG@10: オラクル最良
0.3070 vs champion 0.4049）。原因は、オラクル側が平文・フィールド無視・位置ブーストなしの
古いパイプラインで測られているため。**上限が低いのではなく上限の測り方が古い。**

そのため現在「あとどれだけ伸びしろがあるか」を誰も知らない状態にある。本スクリプトは
オラクル素材を現行パイプライン（フィールド別 match + 位置ブースト + RRF）に載せ替えて、
その穴を埋める。

結果の読み方:
  - オラクルが champion を**大きく上回る** → 生成の質を上げる余地がある。
    疑似文書路線を続ける根拠になる
  - **ほぼ同じ**    → 機構で頭打ち。疑似文書の中身をいくら良くしても伸びない。
    密検索・リランカーへ移る根拠になる

## 条件（22トピック。オラクル素材があるのはこの22件だけ）

  baseline         : narrative そのまま bm25_body（参照）
  oracle_flat      : 旧オラクル方式の再現。narrative×5 + セグメント本文を bm25_body で
                     検索し、上位ORACLE_FLAT_K本ぶんをRRF融合。フィールド分割も
                     位置ブーストもしない
  long_pos_fielded : LLMもオラクルも使わない対照。narrative×5 + Subquery を3フィールドへ
                     個別match + 位置ブースト（§7.8 の LLM不要条件と同じ）
  champion         : 現行最良。LLM生成の構造化疑似文書 + フィールド別match + 位置ブースト
  oracle_matched   : championの疑似文書をオラクルセグメントに置き換えた版。ただし
                     **クエリ量を champion に揃える**（後述）。素材の「質」だけを比べる条件
  oracle_body      : 同上だが切り詰めず、セグメント全文を body に入れる
  oracle_rich      : セグメント3本を title/headings/body すべてに入れる、最も寛大な条件。
                     素材量を増やしたときに伸びるかを見る

  オラクルセグメントは qrel スコアの降順で並んでおり、j番目のSubqueryには
  j番目（oracle_rich では 3j〜3j+2 番目）のセグメントを割り当てる。

## クエリ量を揃える必要がある（重要）

オラクルは平文なので素朴に3フィールドすべてへ入れたくなるが、それをやると
**champion よりクエリ量が2〜6倍になり、素材の質ではなくクエリ量の差を測ってしまう**。
実測した TermQuery 節数（1 Subquery あたり）:

    champion            : title 165 + headings 172 + body 296 =  633 節
    セグメント1本を3フィールド : 384 × 3                       = 1152 節
    セグメント3本を3フィールド : 1234 × 3                      = 3702 節

そこで oracle_matched は、**オラクル本文を body フィールドだけに入れ、しかも語数を
その Subquery の疑似文書 body と同じに切り詰める**。title/headings は long_pos_fielded と
同じく narrative×5 + Subquery のみ。これで champion とクエリ量がほぼ揃い、差は
「LLM生成の疑似文書」対「人間が正解判定した本物の本文」という素材の質だけになる。

oracle_body / oracle_rich は量の制約を外した条件で、量を増やせば伸びるのかを見る
（節数が既定の maxClauseCount=1024 を超えるため、クラスタ設定を
 transient で 8192 に引き上げて実行した。既存クエリは 633 節で上限未満なので
 過去の結果には影響しない）。

  検索側の設定は champion と完全に同一に固定
  （均等重み1:1:1、span_end=100、span_boost=15、RETRIEVE_K=1000、QUERY_REPEAT=5、
   span_terms=analyze_terms(Subquery)、RRF k=60）。

## 注意

オラクルは**検索対象の正解文書そのものの本文**を使うので、当然ながら極めて有利な条件。
これは達成可能な性能ではなく「素材が理想的なときの上限」を測るためのもの。

使い方:
    python evaluate_oracle_current_pipeline.py
    python evaluate_oracle_current_pipeline.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pytrec_eval

from retriever import (bm25_body, bm25_equalweight_posboost_discourseboost,
                       analyze_terms, rrf_fuse)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
ORACLE_FILE = os.path.join(RAG_DIR, "..", "oracle", "qrel_segment_pool.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
SPAN_END, SPAN_BOOST = 100, 15.0
ORACLE_FLAT_K = 10          # 旧オラクル方式で使うセグメント本数（既存の query2doc_10 に対応）
ORACLE_SEGS_PER_SUB = 3     # oracle_3seg で1Subqueryあたりに連結する本数

METHODS = ["baseline", "oracle_flat", "long_pos_fielded", "champion",
           "oracle_matched", "oracle_body", "oracle_rich"]

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


def prepare(webstyle, oracle, qids):
    """{qid: {"subs": [(Subquery, structured_doc, span_terms), ...], "segs": [本文, ...]}}"""
    out = {}
    for qid in qids:
        entry = webstyle[qid]
        subs = []
        for dq, doc in zip(entry.get("decomposed_queries") or [],
                           entry.get("query2doc_docs_structured") or []):
            if not doc or not doc.get("body"):
                continue
            subs.append((dq, doc, analyze_terms(dq, field="body")))
        out[qid] = {"subs": subs, "segs": oracle[qid]["pool"]}
    return out


def fielded(t_q, h_q, b_q, span_terms):
    return bm25_equalweight_posboost_discourseboost(
        t_q, h_q, b_q, span_terms, k=RETRIEVE_K,
        span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])


def build_run(method, qids, queries, prepared):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        rep = " ".join([q] * QUERY_REPEAT)
        subs, segs = prepared[qid]["subs"], prepared[qid]["segs"]
        lists = []

        if method == "baseline":
            lists = [bm25_body(q, k=RETRIEVE_K)]
        elif method == "oracle_flat":
            # 旧方式: narrative×5 + セグメント本文 を単一bodyへ。フィールド分割も位置ブーストもなし
            for seg in segs[:ORACLE_FLAT_K]:
                lists.append(bm25_body(f"{rep} {seg}", k=RETRIEVE_K))
        elif method == "long_pos_fielded":
            for dq, _doc, span_terms in subs:
                lt = f"{rep} {dq}"
                lists.append(fielded(lt, lt, lt, span_terms))
        elif method == "champion":
            for dq, doc, span_terms in subs:
                lists.append(fielded(
                    f"{rep} {dq} {doc['title']}",
                    f"{rep} {dq} {' '.join(doc['headings'])}",
                    f"{rep} {dq} {doc['body']}", span_terms))
        elif method in ("oracle_matched", "oracle_body"):
            # オラクル本文は body フィールドだけに入れる。title/headings は
            # long_pos_fielded と同じく narrative×5 + Subquery のみ。
            # oracle_matched はさらに、その Subquery の疑似文書 body と同じ語数に
            # 切り詰めてクエリ量を champion に揃える。
            for j, (dq, doc, span_terms) in enumerate(subs):
                if j >= len(segs):
                    break
                seg = segs[j]
                if method == "oracle_matched":
                    budget = len(doc["body"].split())
                    seg = " ".join(seg.split()[:budget])
                lt = f"{rep} {dq}"
                lists.append(fielded(lt, lt, f"{lt} {seg}", span_terms))
        elif method == "oracle_rich":
            # 量の制約を外した最も寛大な条件（セグメント3本を3フィールドすべてへ）
            n = ORACLE_SEGS_PER_SUB
            for j, (dq, _doc, span_terms) in enumerate(subs):
                chunk = segs[j * n:(j + 1) * n]
                if not chunk:
                    continue
                t = f"{rep} {dq} {' '.join(chunk)}"
                lists.append(fielded(t, t, t, span_terms))
        else:
            raise ValueError(f"未知の method: {method}")

        run[qid] = fuse(lists, TOPK)
        if i % 5 == 0 or i == len(qids):
            print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()
    return run


def evaluate(run, qrels, qids):
    target = [q for q in qids if q in qrels and run.get(q)]
    if not target:
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
    n = len(res)
    out = {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}
    out["n"] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    with open(ORACLE_FILE) as f:
        oracle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    qids = sorted(set(oracle) & set(webstyle) & set(queries))
    if args.limit:
        qids = qids[:args.limit]
    prepared = prepare(webstyle, oracle, qids)
    qids = [q for q in qids if prepared[q]["subs"] and prepared[q]["segs"]]

    print(f"オラクル素材があるトピック: {len(qids)}")
    print(f"条件: {', '.join(METHODS)}")
    print(f"RETRIEVE_K={RETRIEVE_K} QUERY_REPEAT={QUERY_REPEAT} "
          f"SPAN_END={SPAN_END} SPAN_BOOST={SPAN_BOOST} "
          f"ORACLE_FLAT_K={ORACLE_FLAT_K} ORACLE_SEGS_PER_SUB={ORACLE_SEGS_PER_SUB}")
    print("=" * 72)

    runs = {m: build_run(m, qids, queries, prepared) for m in METHODS}

    summary = {}
    for name in QREL_SETS:
        for m in METHODS:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], qids)

    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    print("\n" + "=" * 72)
    print(f"SUMMARY  対象 {len(qids)} トピック")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        header = "method".ljust(18) + "".join(labels[k].ljust(15) for k in METRIC_KEYS) + "n"
        print(header)
        print("-" * len(header))
        for m in METHODS:
            agg = summary[name].get(m)
            row = m.ljust(18)
            if not agg:
                print(row + "-")
                continue
            row += "".join(f"{agg[k]:.4f}".ljust(15) for k in METRIC_KEYS)
            row += str(agg["n"])
            print(row)

        ch = summary[name].get("champion")
        if ch:
            print(f"\n  [{name}] champion との差")
            for m in METHODS:
                agg = summary[name].get(m)
                if not agg or m == "champion":
                    continue
                print(f"    {m:18s} " + "  ".join(
                    f"{labels[k]}: {agg[k] - ch[k]:+.4f}" for k in METRIC_KEYS))
    print("=" * 72)

    out_path = os.path.join(RAG_DIR, "oracle_current_pipeline_result.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(qids), "qids": qids,
                   "config": {"retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT,
                              "span_end": SPAN_END, "span_boost": SPAN_BOOST,
                              "oracle_flat_k": ORACLE_FLAT_K,
                              "oracle_segs_per_sub": ORACLE_SEGS_PER_SUB},
                   "summary": summary}, f, ensure_ascii=False, indent=2)
    print(f"\n集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
