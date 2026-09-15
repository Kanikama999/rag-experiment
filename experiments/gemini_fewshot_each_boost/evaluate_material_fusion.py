"""
4つのクエリ素材を「連結」せず「別クエリとして検索してRRF融合」したらどうなるかを測る。

背景:
現行championは narrative×5 + Subquery + 疑似参照文書 を**1本のクエリ文字列に連結**して
検索している。一方 query_component_grid.py の③（固有貢献）で、
**4素材すべてが共通して見つける正解は12.5%しかなく、87.5%はどれか一部しか拾えていない**
ことが分かった。素材が相補的なら、連結して1本にするより**別々に検索して融合**した方が
和集合を取れるはずである。それを直接確かめる。

素材（サブクエリ単位で1本ずつ検索する）:
    narrative : 元クエリを3フィールドすべてにmatch
    subquery  : Subqueryを3フィールドすべてにmatch
    pseudo    : 疑似参照文書 title→title / headings→headings / body→body
    termlist  : 拡張語リスト title_terms→title / heading_terms→headings / body_terms→body

【注意】単独クエリでは繰り返しは無意味なので narrative は×1で検索する。
match節が5本になっても全文書のスコアが一律5倍になるだけで**順位は完全に同じ**になるため
（連結時のみ、他素材とのTFバランスを変える意味で×5が効く）。

融合トポロジー2種:
    flat  : 全 (素材 × Subquery) 本のランキングを一度にRRF融合
    hier  : まず素材横断で融合してSubqueryごとの1本にし、次にSubquery横断で融合
            （現行championのSubquery単位RRFに素材の段を1つ足した形）

検索は素材×Subqueryぶん一度だけ行い、あとの融合は全てオフライン。したがって
**素材の部分集合15通り × トポロジー2種**を追加検索ゼロで評価できる。
位置ブーストなし/ありの2armを回す。

比較対象（既知）:
    champion                        consensus nDCG@10 0.5251 / R@1000 0.3095
    連結・全部入り（位置ブーストあり）      0.5018 / 0.3142
    連結・narrative×5+拡張（2素材・最良）  0.5251 / 0.3108

使い方:
    python3 evaluate_material_fusion.py [LIMIT]
"""

from __future__ import annotations

import gzip
import itertools
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
TERMLIST_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_termlist_T8_H20_B40.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
CACHE = os.path.join(RAG_DIR, "_cache_material_rankings.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K = 1000, 1000
SPAN_END, SPAN_BOOST = 100, 15.0
N_WORKERS = 8
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

MATERIALS = ["narrative", "subquery", "pseudo", "termlist"]
SHORT = {"narrative": "narr", "subquery": "sub", "pseudo": "疑似", "termlist": "拡張"}
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


def material_fields(mat, narrative, dq, pdoc, terms):
    """素材ごとの (title, headings, body) クエリ本文。"""
    if mat == "narrative":
        return narrative, narrative, narrative
    if mat == "subquery":
        return dq, dq, dq
    if mat == "pseudo":
        return pdoc["title"], " ".join(pdoc["headings"]), pdoc["body"]
    return (" ".join(terms["title_terms"]), " ".join(terms["heading_terms"]),
            " ".join(terms["body_terms"]))


def search_one(mat, qid, i, narrative, dq, pdoc, terms, span, pb):
    t, h, b = material_fields(mat, narrative, dq, pdoc, terms)
    if pb:
        lst = bm25_equalweight_posboost_discourseboost(
            t, h, b, span, k=RETRIEVE_K, span_end=SPAN_END,
            span_boost=SPAN_BOOST, markers=[])
    else:
        lst = bm25_fielded(t, h, b, k=RETRIEVE_K,
                            title_boost=1, headings_boost=1, body_boost=1)
    return mat, qid, i, pb, [d for d, _ in lst]      # RRFは順位しか使わないのでidだけ保存


def build_cache(qids, queries, webstyle, termlist):
    if os.path.exists(CACHE):
        with gzip.open(CACHE, "rt") as f:
            c = json.load(f)
        if c.get("n_topics") == len(qids):
            print(f"検索キャッシュを再利用（{c['n_searches']}件）")
            return c["rankings"]
    jobs = []
    for qid in qids:
        we, te = webstyle[qid], termlist[qid]
        for i, dq in enumerate(we["decomposed_queries"]):
            pdoc = we["query2doc_docs_structured"][i]
            if not (pdoc and pdoc.get("body")):
                continue
            span = analyze_terms(dq)
            for pb in (0, 1):
                for mat in MATERIALS:
                    jobs.append((mat, qid, i, queries[qid], dq, pdoc,
                                  te["query2doc_docs_terms"][i], span, pb))
    print(f"検索 {len(jobs)} 件（素材×Subquery×arm。以降の融合は全てオフライン）...")
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(search_one, *j) for j in jobs]
        for n, f in enumerate(futs, 1):
            mat, qid, i, pb, ids = f.result()
            out.setdefault(str(pb), {}).setdefault(qid, {}).setdefault(str(i), {})[mat] = ids
            if n % 200 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el/60:.1f}分  残り約{el/n*(len(jobs)-n)/60:.0f}分", flush=True)
    with gzip.open(CACHE, "wt") as f:
        json.dump({"n_topics": len(qids), "n_searches": len(jobs), "rankings": out}, f)
    print(f"-> 検索キャッシュ保存（{(time.time()-t0)/60:.1f}分）")
    return out


def as_pairs(ids):
    return [(d, 1.0) for d in ids]


def fuse_topic(per_sub, mats, topology):
    """per_sub = {subquery_idx: {material: [docid,...]}}"""
    if topology == "flat":
        lists = [as_pairs(per_sub[i][m]) for i in per_sub for m in mats if m in per_sub[i]]
        return rrf_fuse(lists, top_n=TOPK) if lists else []
    inner = []
    for i in per_sub:
        ls = [as_pairs(per_sub[i][m]) for m in mats if m in per_sub[i]]
        if ls:
            inner.append(rrf_fuse(ls, top_n=TOPK))
    return rrf_fuse(inner, top_n=TOPK) if inner else []


def evaluate(run, qrels, qids):
    target = [q for q in qids if q in qrels and run.get(q)]
    if not target:
        return None
    # RelevanceEvaluator は使い回すと最大カットオフの指標が落ちるので毎回作り直す
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
    if not res:
        return None
    n = len(res)
    agg = {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}
    agg["n"] = n
    return agg


def main():
    print("読み込み中...")
    webstyle = json.load(open(WEBSTYLE_FILE))["results"]
    termlist = json.load(open(TERMLIST_FILE))["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}
    qids = sorted(q for q in webstyle
                   if q in queries and q in termlist and webstyle[q].get("decomposed_queries"))
    if LIMIT:
        qids = qids[:LIMIT]

    rankings = build_cache(qids, queries, webstyle, termlist)

    print("\n融合を全数評価中（追加検索なし）...")
    results = {}
    for pb in (0, 1):
        rk = rankings[str(pb)]
        for r in range(1, len(MATERIALS) + 1):
            for mats in itertools.combinations(MATERIALS, r):
                for topo in ("flat", "hier"):
                    if r == 1 and topo == "hier":
                        continue          # 1素材ならflatと同一
                    run = {}
                    for qid in qids:
                        if qid not in rk:
                            continue
                        fused = fuse_topic(rk[qid], mats, topo)
                        run[qid] = {d: float(s) for d, s in fused}
                    name = f"{'+'.join(SHORT[m] for m in mats)}|{topo}|pb{pb}"
                    rec = {"materials": list(mats), "topology": topo, "posboost": pb}
                    for qs in QREL_SETS:
                        rec[qs] = evaluate(run, qrels[qs], qids)
                    results[name] = rec

    out = os.path.join(RAG_DIR, "material_fusion_result.json")
    json.dump({"n_topics": len(qids), "config": {"retrieve_k": RETRIEVE_K, "rrf_k": 60,
                "span_end": SPAN_END, "span_boost": SPAN_BOOST},
                "results": results}, open(out, "w"), indent=2)

    for pb in (0, 1):
        print(f"\n=== consensus / 位置ブースト{'あり' if pb else 'なし'}（nDCG@10 降順）===")
        rows = [(k, v) for k, v in results.items() if v["posboost"] == pb and v["consensus"]]
        for k, v in sorted(rows, key=lambda x: -x[1]["consensus"]["ndcg_cut_10"]):
            c = v["consensus"]
            print(f"  {k:<34} nDCG@10={c['ndcg_cut_10']:.4f}  R@1000={c['recall_1000']:.4f}"
                  f"  R@100={c['recall_100']:.4f}  P@100={c['P_100']:.4f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
