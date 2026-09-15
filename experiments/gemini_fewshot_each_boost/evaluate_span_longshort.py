"""
位置ブーストの span_end（文書冒頭の何語をブースト対象にするか）を、
long側（narrative×5+Subquery+疑似文書）とshort側（Subquery単独）で別々にチューニングする。

背景:
現行championはlong側にspan_end=100・span_boost=15を使っている。この値はlong構成で
チューニングされたものであり、short側（match節が少なくBM25生スコアが小さい）に同じ値が
最適とは限らない。実際 evaluate_longshort_fusion.py では「short側を等倍で混ぜると
位置ブーストありでは性能が落ちる」ことが分かっており、span_endの不一致が一因の可能性がある。

さらに、long+short融合に第3の腕として拡張語リスト（キーワード拡張クエリ）を足せないか。
拡張語リストは疑似参照文書の代替として測ると大きく負けていた（§採用見送り）が、
「置き換え」ではなく「short側への追加」としてはまだ試していない。ランキング自体は
evaluate_material_fusion.py で既に検索・キャッシュ済み（_cache_material_rankings.json.gz の
termlist）なので、追加検索なしで試せる。

設計:
  ①long側 span_end を [25,50,100,150,200,300] で振る（span_boost=15固定、queryはchampionの
    long構成そのまま: narrative×5 + Subquery + 疑似文書full）→ 単独性能から最適点 EL を探す
  ②short側（Subquery単独、title/headings/bodyすべてdqを使う。素材別融合実験と同じ構成）で
    同じ span_end グリッドを振る → 最適点 ES を探す
  ③ long@EL と short@ES をRRF融合（long重み 1/3/5/8）
  ④ ③に拡張語リスト（キャッシュ済み、位置ブーストあり）を第3の腕として重み1で追加

【検算】
  long側 span_end=100 は champion に一致するはず（consensus 0.0689/0.3095/0.5251/0.7769）。
  short側 span_end=100 は素材別融合実験の sub|flat|pb1 に一致するはず（consensus nDCG=0.2697）。

使い方:
    python3 evaluate_span_longshort.py [LIMIT]
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import bm25_equalweight_posboost_discourseboost, analyze_terms, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
MATCACHE = os.path.join(RAG_DIR, "_cache_material_rankings.json.gz")
CACHE = os.path.join(RAG_DIR, "_cache_span_longshort.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
SPAN_ENDS = [25, 50, 100, 150, 200, 300]
SPAN_BOOST = 15.0
FUSE_WEIGHTS = [1, 3, 5, 8]
N_WORKERS = 8
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


def search_one(side, span_end, qid, i, narrative, dq, pdoc, span):
    if side == "long":
        rep = " ".join([narrative] * QUERY_REPEAT)
        t = f"{rep} {dq} {pdoc['title']}"
        h = f"{rep} {dq} {' '.join(pdoc['headings'])}"
        b = f"{rep} {dq} {pdoc['body']}"
    else:
        t = h = b = dq
    lst = bm25_equalweight_posboost_discourseboost(
        t, h, b, span, k=RETRIEVE_K, span_end=span_end, span_boost=SPAN_BOOST, markers=[])
    return side, span_end, qid, i, [d for d, _ in lst]


def build(qids, queries, webstyle):
    if os.path.exists(CACHE):
        with gzip.open(CACHE, "rt") as f:
            c = json.load(f)
        if c.get("n_topics") == len(qids):
            print(f"検索キャッシュを再利用（{c['n']}件）")
            return c["rankings"]
    jobs = []
    for qid in qids:
        e = webstyle[qid]
        for i, dq in enumerate(e["decomposed_queries"]):
            pdoc = e["query2doc_docs_structured"][i]
            if not (pdoc and pdoc.get("body")):
                continue
            span = analyze_terms(dq)
            for side in ("long", "short"):
                for se in SPAN_ENDS:
                    jobs.append((side, se, qid, i, queries[qid], dq, pdoc, span))
    print(f"long/short × span_end{SPAN_ENDS} = {len(jobs)} 検索...")
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(search_one, *j) for j in jobs]
        for n, f in enumerate(futs, 1):
            side, se, qid, i, ids = f.result()
            out.setdefault(f"{side}_{se}", {}).setdefault(qid, {})[str(i)] = ids
            if n % 500 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el/60:.1f}分  残り約{el/n*(len(jobs)-n)/60:.0f}分", flush=True)
    with gzip.open(CACHE, "wt") as f:
        json.dump({"n_topics": len(qids), "n": len(jobs), "rankings": out}, f)
    print(f"-> キャッシュ保存（{(time.time()-t0)/60:.1f}分）")
    return out


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


def to_run(per, qids):
    return {qid: {d: float(s) for d, s in
            (rrf_fuse([[(x, 1.0) for x in ids] for ids in per.get(qid, {}).values()], top_n=TOPK)
             if per.get(qid) else [])} for qid in qids}


def main():
    print("読み込み中...")
    webstyle = json.load(open(WEBSTYLE_FILE))["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}
    qids = sorted(q for q in webstyle if q in queries and webstyle[q].get("decomposed_queries"))
    if LIMIT:
        qids = qids[:LIMIT]

    RK = build(qids, queries, webstyle)
    with gzip.open(MATCACHE, "rt") as f:
        TERMLIST = json.load(f)["rankings"]["1"]     # 拡張語リスト、位置ブーストあり（既存キャッシュ）

    results = {"alone": {}, "fuse": {}}
    for side in ("long", "short"):
        for se in SPAN_ENDS:
            key = f"{side}_{se}"
            run = to_run(RK[key], qids)
            rec = {}
            for qs in QREL_SETS:
                rec[qs] = evaluate(run, qrels[qs], qids)
            results["alone"][key] = rec

    # ①②の最適点を consensus nDCG@10 で選ぶ
    def best_se(side):
        return max(SPAN_ENDS, key=lambda se: (results["alone"][f"{side}_{se}"]["consensus"] or {}).get("ndcg_cut_10", -1))
    EL, ES = best_se("long"), best_se("short")
    print(f"\nlong側 最適 span_end = {EL}   short側 最適 span_end = {ES}")

    # ③long@EL + short@ES の融合、重みを振る
    def pairs(ids):
        return [(d, 1.0) for d in ids]

    for w in FUSE_WEIGHTS:
        for with_termlist in (False, True):
            run = {}
            for qid in qids:
                lists, ws = [], []
                for i, ids in RK[f"long_{EL}"].get(qid, {}).items():
                    lists.append(pairs(ids)); ws.append(float(w))
                for i, ids in RK[f"short_{ES}"].get(qid, {}).items():
                    lists.append(pairs(ids)); ws.append(1.0)
                if with_termlist:
                    for i, d in TERMLIST.get(qid, {}).items():
                        if "termlist" in d:
                            lists.append(pairs(d["termlist"])); ws.append(1.0)
                run[qid] = {dd: float(s) for dd, s in
                            (rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else [])}
            key = f"long{EL}x{w}+short{ES}" + ("+termlist" if with_termlist else "")
            rec = {"EL": EL, "ES": ES, "w": w, "termlist": with_termlist}
            for qs in QREL_SETS:
                rec[qs] = evaluate(run, qrels[qs], qids)
            results["fuse"][key] = rec

    out = os.path.join(RAG_DIR, "span_longshort_result.json")
    json.dump({"n_topics": len(qids), "EL": EL, "ES": ES, "results": results},
              open(out, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n{'='*80}\n=== {qs} ===\n{'='*80}")
        print("\n[① long側 span_end 単独性能]")
        for se in SPAN_ENDS:
            a = results["alone"][f"long_{se}"].get(qs)
            if a: print(f"  span_end={se:<5} R@100={a['recall_100']:.4f} R@1000={a['recall_1000']:.4f} nDCG@10={a['ndcg_cut_10']:.4f} P@100={a['P_100']:.4f}")
        print("\n[② short側 span_end 単独性能]")
        for se in SPAN_ENDS:
            a = results["alone"][f"short_{se}"].get(qs)
            if a: print(f"  span_end={se:<5} R@100={a['recall_100']:.4f} R@1000={a['recall_1000']:.4f} nDCG@10={a['ndcg_cut_10']:.4f} P@100={a['P_100']:.4f}")
        print(f"\n[③④ long@{EL} + short@{ES} 融合（±拡張語リスト）]")
        rows = [(k, v) for k, v in results["fuse"].items() if v.get(qs)]
        for k, v in sorted(rows, key=lambda x: -x[1][qs]["ndcg_cut_10"]):
            a = v[qs]
            print(f"  {k:<40} R@100={a['recall_100']:.4f} R@1000={a['recall_1000']:.4f} nDCG@10={a['ndcg_cut_10']:.4f} P@100={a['P_100']:.4f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
