"""
長いクエリと短いクエリの検索結果をRRF融合したらどうなるかを測る。

背景:
「titleがあるのに取れなかった正解」503件（grade>=2 未発見644件の78%）を実文書で調べると、
**トピックの一部のファセットしか扱わない短い文書**に偏っていた。

    ファセット被覆  未発見 66.2%（2.85/4.20 Subquery）vs 発見済み 80.2%（3.63/4.53）
    本文語数中央値  未発見 1160語 vs 発見済み 1751語   短文書(<800語)は2.2倍

現行のクエリは narrative×5 + Subquery + 疑似文書で700語規模あり、狭く短い文書は
マッチ語の絶対数で長文書に負ける構造になっている。ならば**長いクエリと短いクエリを
別々に検索して融合すれば、両方の取り分を得られる**のではないか、という仮説。

【重要な限界を先に】この融合は上記503件の回収手段にはならない。素材別融合実験で使った
`subquery` 単独（約15語＝最短）も含めた4素材の**どれもがあの503件を上位1000件に入れられ
なかった**ので、短くすれば取れるわけではないことが既に分かっている。狙いは未発見の回収
ではなく**順位づけの改善**である。候補プールには正解が入っているのに上位に来ておらず、
consensus recall@1000 は達成0.3095に対しプール内上限0.5061（到達率61%）と大きな差がある。

なお素材別融合では `narr+sub/flat`（nDCG 0.5193）が連結最良 `narr×5+Sub×5`（0.5085）を
上回っており、長短融合の効果は部分的に実証済み。**未検証なのは champion 級の長いクエリ
（約700語）と短いクエリの融合**で、本スクリプトはそこを埋める。

条件:
  long        : narrative×5 + Subquery + 疑似文書（championと同一構成）。単独ならchampion再現
  + subquery  : 長 × 短（約15語）        ← 本命
  + narrative : 長 × 中（約100語）
  + both      : 3本
  重み付き     : long を 2倍/3倍にした重み付きRRF（長側を主、短側を補正に使う形）

短い側のランキングは素材別融合の検索キャッシュ（_cache_material_rankings.json.gz）を
再利用するので、新規検索は long 側だけ。融合は全てオフライン。

使い方:
    python3 evaluate_longshort_fusion.py [LIMIT]
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import (bm25_fielded, bm25_equalweight_posboost_discourseboost,
                        analyze_terms, rrf_fuse)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
MATCACHE = os.path.join(RAG_DIR, "_cache_material_rankings.json.gz")
LONGCACHE = os.path.join(RAG_DIR, "_cache_long_rankings.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
SPAN_END, SPAN_BOOST = 100, 15.0
N_WORKERS = 6
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


def load_qrels(p):
    q = {}
    for line in open(p):
        s = line.split()
        if len(s) == 4 and not line.startswith("#"):
            q.setdefault(s[0], {})[s[2]] = int(s[3])
    return q


def load_queries(p):
    q = {}
    for line in open(p):
        line = line.strip()
        if line:
            o = json.loads(line)
            q[o["id"]] = o["title"]
    return q


def search_long(qid, i, narrative, dq, pdoc, span, pb):
    rep = " ".join([narrative] * QUERY_REPEAT)
    t = f"{rep} {dq} {pdoc['title']}"
    h = f"{rep} {dq} {' '.join(pdoc['headings'])}"
    b = f"{rep} {dq} {pdoc['body']}"
    if pb:
        lst = bm25_equalweight_posboost_discourseboost(
            t, h, b, span, k=RETRIEVE_K, span_end=SPAN_END,
            span_boost=SPAN_BOOST, markers=[])
    else:
        lst = bm25_fielded(t, h, b, k=RETRIEVE_K,
                            title_boost=1, headings_boost=1, body_boost=1)
    return qid, i, pb, [d for d, _ in lst]


def build_long(qids, queries, webstyle):
    if os.path.exists(LONGCACHE):
        with gzip.open(LONGCACHE, "rt") as f:
            c = json.load(f)
        if c.get("n_topics") == len(qids):
            print(f"long検索キャッシュを再利用（{c['n']}件）")
            return c["rankings"]
    jobs = []
    for qid in qids:
        e = webstyle[qid]
        for i, dq in enumerate(e["decomposed_queries"]):
            pdoc = e["query2doc_docs_structured"][i]
            if pdoc and pdoc.get("body"):
                span = analyze_terms(dq)
                for pb in (0, 1):
                    jobs.append((qid, i, queries[qid], dq, pdoc, span, pb))
    print(f"long クエリを {len(jobs)} 件検索中（短い側は既存キャッシュを再利用）...")
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(search_long, *j) for j in jobs]
        for n, f in enumerate(futs, 1):
            qid, i, pb, ids = f.result()
            out.setdefault(str(pb), {}).setdefault(qid, {})[str(i)] = ids
            if n % 200 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el/60:.1f}分  残り約{el/n*(len(jobs)-n)/60:.0f}分", flush=True)
    with gzip.open(LONGCACHE, "wt") as f:
        json.dump({"n_topics": len(qids), "n": len(jobs), "rankings": out}, f)
    print(f"-> longキャッシュ保存（{(time.time()-t0)/60:.1f}分）")
    return out


def pairs(ids):
    return [(d, 1.0) for d in ids]


def evaluate(run, qrels, qids):
    tg = [q for q in qids if q in qrels and run.get(q)]
    if not tg:
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in tg}, METRICS)
    res = ev.evaluate({q: run[q] for q in tg})
    if not res:
        return None
    n = len(res)
    a = {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}
    a["n"] = n
    return a


def main():
    print("読み込み中...")
    webstyle = json.load(open(WEBSTYLE_FILE))["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}
    qids = sorted(q for q in webstyle if q in queries and webstyle[q].get("decomposed_queries"))
    if LIMIT:
        qids = qids[:LIMIT]

    LONG = build_long(qids, queries, webstyle)
    with gzip.open(MATCACHE, "rt") as f:
        MAT = json.load(f)["rankings"]

    # 融合条件: (短い素材のリスト, longの重み)
    CONDS = [
        ("long のみ（champion再現）", [], 1),
        ("long + subquery", ["subquery"], 1),
        ("long + narrative", ["narrative"], 1),
        ("long + narrative + subquery", ["narrative", "subquery"], 1),
        ("long×2 + subquery", ["subquery"], 2),
        ("long×3 + subquery", ["subquery"], 3),
        ("long×2 + narrative + subquery", ["narrative", "subquery"], 2),
    ]

    results = {}
    for pb in ("0", "1"):
        for label, shorts, lw in CONDS:
            run = {}
            for qid in qids:
                lg = LONG[pb].get(qid, {})
                mt = MAT[pb].get(qid, {})
                lists, ws = [], []
                for i, ids in lg.items():
                    lists.append(pairs(ids)); ws.append(float(lw))
                for i, d in mt.items():
                    for m in shorts:
                        if m in d:
                            lists.append(pairs(d[m])); ws.append(1.0)
                run[qid] = {dd: float(s) for dd, s in
                            (rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else [])}
            rec = {"shorts": shorts, "long_weight": lw, "posboost": int(pb)}
            for qs in QREL_SETS:
                rec[qs] = evaluate(run, qrels[qs], qids)
            results[f"{label}|pb{pb}"] = rec

    out = os.path.join(RAG_DIR, "longshort_fusion_result.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

    for qs in QREL_SETS:
        for pb in ("0", "1"):
            print(f"\n=== {qs} / 位置ブースト{'あり' if pb=='1' else 'なし'} ===")
            print(f"{'条件':<32}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
            rows = [(k, v) for k, v in results.items() if k.endswith(f"pb{pb}") and v[qs]]
            for k, v in sorted(rows, key=lambda x: -x[1][qs]["ndcg_cut_10"]):
                a = v[qs]
                print(f"{k.split('|')[0]:<32}"+"".join(f"{a[m]:>10.4f}" for m in METRIC_KEYS))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
