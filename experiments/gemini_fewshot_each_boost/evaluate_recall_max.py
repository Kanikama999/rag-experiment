"""
recall優先で、これまでの検証済み改善を全部組み合わせる。

背景:
これまでの全実験を横断すると、recall@1000 の現状最高は best_combination_result.json の
sub+narr|both|se200sb30|fuse3（consensus 0.3266 / coverage 0.6304）で、これは
「span_terms に narrative を追加 + クエリ構成を疑似文書+拡張語リスト両方(both) +
span_end=200/span_boost=30」というlong側の改善だけを積んだもの。

一方 evaluate_span_longshort.py で「short側(Subquery単独)の最適span_endはまだ300でも
未収束（急勾配で伸び続けている）」ことが分かった。この2つの改善は一度も組み合わせていない。

構成:
  long  : best_combination の "sub+narr|both|se200sb30"（既存キャッシュ、追加検索なし）
  short : Subquery単独、span_endを300から延長 [300,400,600,800] で追加検索
          （300は既存キャッシュを再利用、400/600/800だけ新規）
  third : 拡張語リスト（既存キャッシュ、位置ブーストあり、追加検索なし）

3つをRRF融合。long重み[1,3,5,8] × short span_end[300,400,600,800] × termlist有無 を
すべてオフラインで評価する。

使い方:
    python3 evaluate_recall_max.py [LIMIT]
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
BC_CACHE = os.path.join(RAG_DIR, "_cache_best_combination.json.gz")
SP_CACHE = os.path.join(RAG_DIR, "_cache_span_longshort.json.gz")
MAT_CACHE = os.path.join(RAG_DIR, "_cache_material_rankings.json.gz")
NEW_CACHE = os.path.join(RAG_DIR, "_cache_recall_max_short_ext.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K = 1000, 1000
SPAN_BOOST = 15.0
NEW_SPAN_ENDS = [400, 600, 800]
ALL_SPAN_ENDS = [300] + NEW_SPAN_ENDS
LONG_WEIGHTS = [1, 3, 5, 8]
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


def search_short(se, qid, i, dq, span):
    lst = bm25_equalweight_posboost_discourseboost(
        dq, dq, dq, span, k=RETRIEVE_K, span_end=se, span_boost=SPAN_BOOST, markers=[])
    return se, qid, i, [d for d, _ in lst]


def build_short_ext(qids, webstyle):
    if os.path.exists(NEW_CACHE):
        with gzip.open(NEW_CACHE, "rt") as f:
            c = json.load(f)
        if c.get("n_topics") == len(qids):
            print(f"short延長キャッシュを再利用（{c['n']}件）")
            return c["rankings"]
    jobs = []
    for qid in qids:
        for i, dq in enumerate(webstyle[qid]["decomposed_queries"]):
            pdoc = webstyle[qid]["query2doc_docs_structured"][i]
            if not (pdoc and pdoc.get("body")):
                continue
            span = analyze_terms(dq)
            for se in NEW_SPAN_ENDS:
                jobs.append((se, qid, i, dq, span))
    print(f"short側 span_end延長 {NEW_SPAN_ENDS} = {len(jobs)} 検索...")
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(search_short, *j) for j in jobs]
        for n, f in enumerate(futs, 1):
            se, qid, i, ids = f.result()
            out.setdefault(str(se), {}).setdefault(qid, {})[str(i)] = ids
            if n % 300 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el/60:.1f}分  残り約{el/n*(len(jobs)-n)/60:.0f}分", flush=True)
    with gzip.open(NEW_CACHE, "wt") as f:
        json.dump({"n_topics": len(qids), "n": len(jobs), "rankings": out}, f)
    print(f"-> 保存（{(time.time()-t0)/60:.1f}分）")
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


def pairs(ids):
    return [(d, 1.0) for d in ids]


def main():
    print("読み込み中...")
    webstyle = json.load(open(WEBSTYLE_FILE))["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}
    qids = sorted(q for q in webstyle if q in queries and webstyle[q].get("decomposed_queries"))
    if LIMIT:
        qids = qids[:LIMIT]

    with gzip.open(BC_CACHE, "rt") as f:
        LONG = json.load(f)["rankings"]["sub+narr|both|se200sb30"]
    with gzip.open(SP_CACHE, "rt") as f:
        SHORT_300 = json.load(f)["rankings"]["short_300"]
    with gzip.open(MAT_CACHE, "rt") as f:
        TERMLIST = json.load(f)["rankings"]["1"]

    SHORT_EXT = build_short_ext(qids, webstyle)
    SHORT = {"300": SHORT_300, **SHORT_EXT}

    results = {}
    for se in ALL_SPAN_ENDS:
        for lw in LONG_WEIGHTS:
            for use_tl in (False, True):
                run = {}
                for qid in qids:
                    lists, ws = [], []
                    for i, ids in LONG.get(qid, {}).items():
                        lists.append(pairs(ids)); ws.append(float(lw))
                    for i, ids in SHORT[str(se)].get(qid, {}).items():
                        lists.append(pairs(ids)); ws.append(1.0)
                    if use_tl:
                        for i, d in TERMLIST.get(qid, {}).items():
                            if "termlist" in d:
                                lists.append(pairs(d["termlist"])); ws.append(1.0)
                    run[qid] = {dd: float(s) for dd, s in
                                (rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else [])}
                key = f"long_bothse200sb30x{lw}+short_se{se}" + ("+termlist" if use_tl else "")
                rec = {"short_span_end": se, "long_weight": lw, "termlist": use_tl}
                for qs in QREL_SETS:
                    rec[qs] = evaluate(run, qrels[qs], qids)
                results[key] = rec

    out = os.path.join(RAG_DIR, "recall_max_result.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

    for qs in QREL_SETS:
        rows = [(k, v) for k, v in results.items() if v.get(qs)]
        print(f"\n=== {qs}  recall@1000 上位12 ===")
        print(f"{'条件':<52}{'R@100':>8}{'R@1000':>9}{'nDCG@10':>9}{'P@100':>8}")
        for k, v in sorted(rows, key=lambda x: -x[1][qs]["recall_1000"])[:12]:
            a = v[qs]
            print(f"{k:<52}{a['recall_100']:>8.4f}{a['recall_1000']:>9.4f}{a['ndcg_cut_10']:>9.4f}{a['P_100']:>8.4f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
