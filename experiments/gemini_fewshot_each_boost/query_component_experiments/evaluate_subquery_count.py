"""
「最高性能を出すには最低何本のサブクエリが必要か」を全数探索で測る。

背景:
championはトピックあたり平均4.30本（2〜7本）のSubqueryをそれぞれ検索し、RRF(k=60)で
融合している。本数を減らせるなら、疑似文書の生成コスト（サブクエリごとに1回LLMを呼ぶ）が
そのまま比例して下がる。しかし「何本必要か」は未検証。

設計上の鍵:
**検索結果は本数に依存しない。** 各Subqueryの検索は独立で、本数によって変わるのは
融合するリストの組み合わせだけ。したがって451本を一度だけ検索してキャッシュすれば、
あとの全ての本数・全ての組み合わせは追加検索ゼロでオフライン計算できる。

全トピックの部分集合の総数は 2591 通りしかないため、サンプリングではなく**全数列挙**で
正確な値を出す。本数kごとに次の4つを報告する:

  mean  : そのトピックのC(n,k)通り全部の平均 = 「ランダムにk本選んだときの期待値」
  best  : 全組み合わせ中の最良 = **オラクル選択の上限**（正解を見て選ぶので実現不可能）
  worst : 同 最悪 = 下限
  first : 生成順の先頭k本 = 実用上いちばん素朴な戦略

k=n（全部使う）がchampionに一致するので、既知値 consensus nDCG@10=0.5251 が検算になる。

クエリ構成はchampionと完全に同一:
  narrative×5 + Subquery×1 + 疑似文書の各フィールド対応部分、均等重み(1,1,1)、
  span_first位置ブースト(span_end=100, span_boost=15)、RETRIEVE_K=1000、RRF k=60。

使い方:
    python3 evaluate_subquery_count.py [LIMIT]
"""

from __future__ import annotations

import itertools
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import bm25_equalweight_posboost_discourseboost, analyze_terms, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
CACHE = os.path.join(RAG_DIR, "_cache_subquery_rankings_champion.json")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
SPAN_END, SPAN_BOOST = 100, 15.0
N_WORKERS = 4      # グリッド本番と同時に走らせるため控えめ
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


def load_qrels(path):
    q = {}
    for line in open(path):
        p = line.split()
        if len(p) == 4 and not line.startswith("#"):
            q.setdefault(p[0], {})[p[2]] = int(p[3])
    return q


def load_queries(path):
    q = {}
    for line in open(path):
        line = line.strip()
        if line:
            o = json.loads(line)
            q[o["id"]] = o["title"]
    return q


def search_one(qid, i, narrative, dq, pdoc):
    """championと同一構成でSubquery1本ぶんを検索する。"""
    rep = " ".join([narrative] * QUERY_REPEAT)
    t = f"{rep} {dq} {pdoc['title']}"
    h = f"{rep} {dq} {' '.join(pdoc['headings'])}"
    b = f"{rep} {dq} {pdoc['body']}"
    lst = bm25_equalweight_posboost_discourseboost(
        t, h, b, analyze_terms(dq), k=RETRIEVE_K,
        span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
    return qid, i, [(d, float(s)) for d, s in lst]


def build_cache(qids, queries, webstyle):
    if os.path.exists(CACHE):
        c = json.load(open(CACHE))
        if c.get("n_topics") == len(qids):
            print(f"検索キャッシュを再利用（{c['n_subqueries']}本）")
            return {q: {int(i): [tuple(x) for x in v]
                        for i, v in d.items()} for q, d in c["rankings"].items()}
    jobs = []
    for qid in qids:
        e = webstyle[qid]
        for i, dq in enumerate(e["decomposed_queries"]):
            pdoc = e["query2doc_docs_structured"][i]
            if pdoc and pdoc.get("body"):
                jobs.append((qid, i, queries[qid], dq, pdoc))
    print(f"Subquery {len(jobs)} 本を検索中（本数によらず1回だけ）...")
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(search_one, *j) for j in jobs]
        for n, f in enumerate(futs, 1):
            qid, i, lst = f.result()
            out.setdefault(qid, {})[i] = lst
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"n_topics": len(qids), "n_subqueries": len(jobs),
                "rankings": {q: {str(i): v for i, v in d.items()} for q, d in out.items()}},
              open(CACHE, "w"))
    print(f"-> 検索キャッシュ保存（{time.time()-t0:.0f}s）")
    return out


def main():
    print("読み込み中...")
    webstyle = json.load(open(WEBSTYLE_FILE))["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}
    qids = sorted(q for q in webstyle if q in queries and webstyle[q].get("decomposed_queries"))
    if LIMIT:
        qids = qids[:LIMIT]

    rankings = build_cache(qids, queries, webstyle)

    # 全部分集合を融合して評価（追加検索なし）
    print("\n部分集合を全数評価中...")
    # per_topic[qs][qid][k] = {"all": [各組み合わせのmetric dict], "first": metric dict}
    per_topic = {qs: {} for qs in QREL_SETS}
    n_subsets = 0
    for qid in qids:
        idxs = sorted(rankings.get(qid, {}))
        if not idxs:
            continue
        for qs in QREL_SETS:
            if qid not in qrels[qs]:
                continue
            byk = {}
            for k in range(1, len(idxs) + 1):
                allm, firstm = [], None
                for comb in itertools.combinations(idxs, k):
                    fused = rrf_fuse([rankings[qid][i] for i in comb], top_n=TOPK)
                    run = {qid: {d: float(s) for d, s in fused}}
                    # pytrec_eval の RelevanceEvaluator は使い回せない。同じインスタンスで
                    # evaluate() を2回目以降呼ぶと最大カットオフの指標（recall_1000）が
                    # 黙って結果から落ちる（diagnose_subquery_weight_signal.py の注記と同じ罠）。
                    # 必ず呼び出しごとに作り直すこと。
                    ev = pytrec_eval.RelevanceEvaluator({qid: qrels[qs][qid]}, METRICS)
                    m = ev.evaluate(run)[qid]
                    allm.append({kk: m[kk] for kk in METRIC_KEYS})
                    if comb == tuple(idxs[:k]):
                        firstm = allm[-1]
                    n_subsets += 1
                byk[k] = {"all": allm, "first": firstm}
            per_topic[qs][qid] = byk
    print(f"評価した部分集合: {n_subsets}")

    # 本数kごとに集計。トピックごとに有効な本数が違うので、
    # 「k本以上あるトピック」だけで平均する（母集団をkごとに明示する）
    summary = {}
    for qs in QREL_SETS:
        rows = {}
        maxk = max((max(b) for b in per_topic[qs].values()), default=0)
        for k in range(1, maxk + 1):
            tp = [b[k] for b in per_topic[qs].values() if k in b]
            if not tp:
                continue
            row = {"n_topics": len(tp)}
            for m in METRIC_KEYS:
                row[m] = {
                    "mean": statistics.mean(statistics.mean(x[m] for x in t["all"]) for t in tp),
                    "best": statistics.mean(max(x[m] for x in t["all"]) for t in tp),
                    "worst": statistics.mean(min(x[m] for x in t["all"]) for t in tp),
                    "first": statistics.mean(t["first"][m] for t in tp),
                }
            rows[k] = row
        # 「全部使う」= championの再現（各トピックの自分の本数すべて）
        allrow = {"n_topics": len(per_topic[qs])}
        for m in METRIC_KEYS:
            vals = []
            for b in per_topic[qs].values():
                kmax = max(b)
                vals.append(b[kmax]["all"][0][m])   # k=nのときC(n,n)=1通り
            allrow[m] = {"mean": statistics.mean(vals)}
        rows["all"] = allrow
        summary[qs] = rows

    # 母集団を固定した集計。上の rows は「k本以上あるトピック」で平均しているため
    # kごとに母集団が変わり、k=1行とk=4行を直接比較できない。ここでは
    # 「K_FIX本以上持つトピック」だけに限定し、その固定母集団上で k=1..K_FIX を並べる。
    fixed = {}
    for qs in QREL_SETS:
        for K_FIX in (3, 4, 5):
            tp_ids = [q for q, b in per_topic[qs].items() if max(b) >= K_FIX]
            if not tp_ids:
                continue
            rows = {"n_topics": len(tp_ids)}
            for k in range(1, K_FIX + 1):
                r = {}
                for m in METRIC_KEYS:
                    r[m] = {
                        "mean": statistics.mean(
                            statistics.mean(x[m] for x in per_topic[qs][q][k]["all"]) for q in tp_ids),
                        "best": statistics.mean(
                            max(x[m] for x in per_topic[qs][q][k]["all"]) for q in tp_ids),
                        "worst": statistics.mean(
                            min(x[m] for x in per_topic[qs][q][k]["all"]) for q in tp_ids),
                        "first": statistics.mean(per_topic[qs][q][k]["first"][m] for q in tp_ids),
                    }
                rows[k] = r
            # 同じ母集団での「全部使う」（各トピックの本数すべて）
            rows["all"] = {m: statistics.mean(
                per_topic[qs][q][max(per_topic[qs][q])]["all"][0][m] for q in tp_ids)
                for m in METRIC_KEYS}
            fixed.setdefault(qs, {})[f"min{K_FIX}"] = rows
    summary_fixed = fixed

    out = os.path.join(RAG_DIR, "subquery_count_result.json")
    json.dump({"n_topics": len(qids), "n_subsets_evaluated": n_subsets,
                "config": {"retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT,
                            "span_end": SPAN_END, "span_boost": SPAN_BOOST, "rrf_k": 60},
                "summary": summary, "summary_fixed_population": summary_fixed},
              open(out, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print(f"{'k':>4} {'n':>4}  {'nDCG@10 mean/best/first':>32}  {'R@1000 mean/best/first':>30}")
        for k, r in summary[qs].items():
            if k == "all":
                print(f"{'all':>4} {r['n_topics']:>4}  {r['ndcg_cut_10']['mean']:>10.4f}"
                      f"{'':>22}{r['recall_1000']['mean']:>10.4f}")
                continue
            n = r["ndcg_cut_10"]; rc = r["recall_1000"]
            print(f"{k:>4} {r['n_topics']:>4}  {n['mean']:>10.4f}/{n['best']:.4f}/{n['first']:.4f}"
                  f"   {rc['mean']:>10.4f}/{rc['best']:.4f}/{rc['first']:.4f}")
    for qs in QREL_SETS:
        for key, rows in summary_fixed.get(qs, {}).items():
            kmax = max(k for k in rows if isinstance(k, int))
            print(f"\n=== {qs} / 母集団固定: {key}本以上のトピック n={rows['n_topics']} ===")
            print(f"{'k':>4}  {'nDCG@10 mean/best/first':>30}  {'R@1000 mean/best/first':>30}")
            for k in range(1, kmax + 1):
                n, rc = rows[k]["ndcg_cut_10"], rows[k]["recall_1000"]
                print(f"{k:>4}  {n['mean']:>8.4f}/{n['best']:.4f}/{n['first']:.4f}"
                      f"   {rc['mean']:>8.4f}/{rc['best']:.4f}/{rc['first']:.4f}")
            print(f"{'all':>4}  {rows['all']['ndcg_cut_10']:>8.4f}{'':>16}{rows['all']['recall_1000']:>8.4f}")

    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
