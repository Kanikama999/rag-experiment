"""
長い側（narrative×5 + Subquery + 疑似文書）を、短い側(Subquery単独)との融合を保った
まま、どこまで削れるかを調べる。

背景:
evaluate_best_combination.py で新記録が出た（consensus nDCG@10 0.5485、旧記録0.5368、
champion 0.5251）。構成は span_terms=sub+narr・span_end=200/span_boost=30・
長い側=champion構成(narrative×5+Subquery+疑似文書)・短い側=Subquery単独をfuse重み5で
RRF融合、というものだった。

この勝ち筋（span_terms・span パラメータ・融合重み）を固定し、**長い側の構成だけ**を
narrative の有無・疑似文書系の有無で段階的に削り、性能がどこで崩れるかを見る。
「疑似文書もnarrativeも削って長い側がSubquery1回だけになる」極限まで含める
——そこでは長い側と短い側がほぼ同じものになり、融合の意味が消える下限になる。

構成:
    narrative ∈ {0, 5}          （×5 or なし）
    compo ∈ {none, pseudo, termlist, both}
        none     = narrativeとSubqueryだけ（疑似文書系なし）
        pseudo   = championと同じ疑似参照文書
        termlist = 拡張語リスト
        both     = 両方

span_terms = sub+narr（固定・検証済み最良）、span_end=200/span_boost=30（固定）。
短い側は Subquery 単独ランキング（既存の _cache_material_rankings.json.gz を再利用、
新規検索なし）。融合重み {0,3,5,8} は全てオフラインで評価する。

検算: narrative=5, compo=pseudo, fuse=5 が新記録 consensus nDCG@10 0.5485 に一致するはず。

使い方:
    python3 evaluate_long_cut.py [LIMIT]
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
TERMLIST_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_termlist_T8_H20_B40.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
MATCACHE = os.path.join(RAG_DIR, "_cache_material_rankings.json.gz")
CACHE = os.path.join(RAG_DIR, "_cache_long_cut.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K = 1000, 1000
SPAN_END, SPAN_BOOST = 200, 30.0
N_WORKERS = 8
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

NARR_LEVELS = [0, 5]
COMPOS = ["none", "pseudo", "termlist", "both"]
FUSEW = [0, 3, 5, 8]

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


def fields_for(narr_rep, compo, narrative, dq, pdoc, terms):
    rep = " ".join([narrative] * narr_rep) if narr_rep else ""
    hd = " ".join(pdoc["headings"])
    tt, th, tb = (" ".join(terms["title_terms"]), " ".join(terms["heading_terms"]),
                   " ".join(terms["body_terms"]))
    if compo == "none":
        pt = ph = pb_ = ""
    elif compo == "pseudo":
        pt, ph, pb_ = pdoc["title"], hd, pdoc["body"]
    elif compo == "termlist":
        pt, ph, pb_ = tt, th, tb
    else:  # both
        pt, ph, pb_ = f"{pdoc['title']} {tt}", f"{hd} {th}", f"{pdoc['body']} {tb}"
    return (f"{rep} {dq} {pt}".strip(), f"{rep} {dq} {ph}".strip(), f"{rep} {dq} {pb_}".strip())


def search_one(cond, qid, i, narrative, dq, pdoc, terms, sterms):
    narr_rep, compo = cond
    t, h, b = fields_for(narr_rep, compo, narrative, dq, pdoc, terms)
    lst = bm25_equalweight_posboost_discourseboost(
        t, h, b, sterms, k=RETRIEVE_K, span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
    return cond, qid, i, [d for d, _ in lst]


def cname(cond):
    narr_rep, compo = cond
    return f"narr{narr_rep}|{compo}"


def build(qids, queries, webstyle, termlist):
    if os.path.exists(CACHE):
        with gzip.open(CACHE, "rt") as f:
            c = json.load(f)
        if c.get("n_topics") == len(qids):
            print(f"検索キャッシュを再利用（{c['n']}件）")
            return c["rankings"]
    conds = [(n, c) for n in NARR_LEVELS for c in COMPOS]
    jobs = []
    for qid in qids:
        we, te = webstyle[qid], termlist[qid]
        narr_terms = analyze_terms(queries[qid], field="body")
        for i, dq in enumerate(we["decomposed_queries"]):
            pdoc = we["query2doc_docs_structured"][i]
            if not (pdoc and pdoc.get("body")):
                continue
            sub_terms = analyze_terms(dq, field="body")
            sterms = list(dict.fromkeys(sub_terms + narr_terms))   # span_terms = sub+narr 固定
            for cond in conds:
                jobs.append((cond, qid, i, queries[qid], dq, pdoc,
                              te["query2doc_docs_terms"][i], sterms))
    print(f"{len(conds)} 条件 × Subquery = {len(jobs)} 検索...")
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(search_one, *j) for j in jobs]
        for n, f in enumerate(futs, 1):
            cond, qid, i, ids = f.result()
            out.setdefault(cname(cond), {}).setdefault(qid, {})[str(i)] = ids
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

    RK = build(qids, queries, webstyle, termlist)
    with gzip.open(MATCACHE, "rt") as f:
        MAT = json.load(f)["rankings"]["0"]

    print("\n融合をオフライン評価中...")
    results = {}
    for cn, per in RK.items():
        for w in FUSEW:
            run = {}
            for qid in qids:
                lists, ws = [], []
                for i, ids in per.get(qid, {}).items():
                    lists.append([(d, 1.0) for d in ids]); ws.append(float(w or 1))
                if w:
                    for i, d in MAT.get(qid, {}).items():
                        if "subquery" in d:
                            lists.append([(x, 1.0) for x in d["subquery"]]); ws.append(1.0)
                run[qid] = {dd: float(s) for dd, s in
                            (rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else [])}
            rec = {"cond": cn, "fuse_w": w}
            for qs in QREL_SETS:
                rec[qs] = evaluate(run, qrels[qs], qids)
            results[f"{cn}|fuse{w}"] = rec

    out = os.path.join(RAG_DIR, "long_cut_result.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

    for qs in ("consensus", "coverage"):
        rows = [(k, v) for k, v in results.items() if v[qs]]
        print(f"\n=== {qs}  全条件（narr → compo → fuse順） ===")
        print(f"{'条件':<26}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for k, v in sorted(rows, key=lambda x: x[0]):
            a = v[qs]
            print(f"{k:<26}"+"".join(f"{a[m]:>10.4f}" for m in METRIC_KEYS))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
