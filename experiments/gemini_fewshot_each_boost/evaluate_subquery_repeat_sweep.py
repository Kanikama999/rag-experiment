"""
Subqueryの繰り返し回数を細かく振る（0,1,2,3,5）。narrativeは×3固定
（recall_shrink実験で見つかった最適値）、疑似文書+拡張語リストの両方を使う
composition="both"（recall記録と同じ土台）。

背景:
これまでnarrativeの繰り返し回数は0〜5まで細かく振ってきた（evaluate_recall_shrink.py等）が、
Subquery側は2^4グリッド実験の「0か5」の二択でしか振っていなかった。しかもそのグリッドは
narrative×5固定・composition=champion(疑似文書のみ)という別の土台だった。今回はrecall記録
（narr3+both+se200/30+sub+narr全体像）そのものを土台に、Subquery側の最適回数を探す。

削る軸:
    Subquery の繰り返し回数 ∈ {0, 1, 2, 3, 5}   （narrative×3, composition=both固定）
    0 = long側の本文にSubqueryを一切入れない（ただしspan_terms判定語としては常に使う。
        championと同じ「判定語は繰り返し回数と独立」という扱い）
    1 = 現行のrecall記録と同じ（Subqueryは1回だけ）

short側（Subquery単独、位置ブーストなし）・third側（拡張語リスト単独）は既存キャッシュを
再利用。融合重みはrecall記録のx6に固定。

【検算】sub_rep=1 が既存のrecall記録 narr3x6+sub+termlist
（consensus R@1000=0.3318, nDCG@10=0.5147）に一致するはず。

使い方:
    python3 evaluate_subquery_repeat_sweep.py [LIMIT]
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
CACHE = os.path.join(RAG_DIR, "_cache_subquery_repeat_sweep.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K = 1000, 1000
NARRATIVE_REP = 3          # recall_shrinkで見つかった最適値に固定
SUBQUERY_REPS = [0, 1, 2, 3, 5]
SPAN_END, SPAN_BOOST = 200, 30.0
FUSE_WEIGHT = 6.0          # recall記録の重みに固定
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


def search_one(qid, i, narrative, dq, pdoc, terms, sub_rep, span):
    n_rep_text = " ".join([narrative] * NARRATIVE_REP)
    s_rep_text = " ".join([dq] * sub_rep) if sub_rep else ""
    base = " ".join(x for x in (n_rep_text, s_rep_text) if x)
    tt, th, tb = " ".join(terms["title_terms"]), " ".join(terms["heading_terms"]), " ".join(terms["body_terms"])
    t = f"{base} {pdoc['title']} {tt}".strip()
    h = f"{base} {' '.join(pdoc['headings'])} {th}".strip()
    b = f"{base} {pdoc['body']} {tb}".strip()
    lst = bm25_equalweight_posboost_discourseboost(
        t, h, b, span, k=RETRIEVE_K, span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
    return sub_rep, qid, i, [d for d, _ in lst]


def build(qids, queries, webstyle, termlist):
    if os.path.exists(CACHE):
        with gzip.open(CACHE, "rt") as f:
            c = json.load(f)
        if c.get("n_topics") == len(qids):
            print(f"検索キャッシュを再利用（{c['n']}件）")
            return c["rankings"]
    jobs = []
    for qid in qids:
        we, te = webstyle[qid], termlist[qid]
        for i, dq in enumerate(we["decomposed_queries"]):
            pdoc = we["query2doc_docs_structured"][i]
            if not (pdoc and pdoc.get("body")):
                continue
            narrative = queries[qid]
            span = list(dict.fromkeys(analyze_terms(dq) + analyze_terms(narrative)))
            for sr in SUBQUERY_REPS:
                jobs.append((qid, i, narrative, dq, pdoc, te["query2doc_docs_terms"][i], sr, span))
    print(f"Subquery繰り返し{SUBQUERY_REPS} × {len(qids)}トピック分 = {len(jobs)} 検索...")
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(search_one, *j) for j in jobs]
        for n, f in enumerate(futs, 1):
            sr, qid, i, ids = f.result()
            out.setdefault(str(sr), {}).setdefault(qid, {})[str(i)] = ids
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

    results = {}
    for sr in SUBQUERY_REPS:
        per = RK[str(sr)]
        run = {}
        for qid in qids:
            lists, ws = [], []
            for i, ids in per.get(qid, {}).items():
                lists.append([(d, 1.0) for d in ids]); ws.append(FUSE_WEIGHT)
            for i, d in MAT.get(qid, {}).items():
                if "subquery" in d:
                    lists.append([(x, 1.0) for x in d["subquery"]]); ws.append(1.0)
                if "termlist" in d:
                    lists.append([(x, 1.0) for x in d["termlist"]]); ws.append(1.0)
            run[qid] = {d: float(s) for d, s in
                        (rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else [])}
        rec = {"subquery_rep": sr}
        for qs in QREL_SETS:
            rec[qs] = evaluate(run, qrels[qs], qids)
        results[f"subrep{sr}"] = rec

    out = os.path.join(RAG_DIR, "subquery_repeat_sweep_result.json")
    json.dump({"n_topics": len(qids), "narrative_rep": NARRATIVE_REP,
                "fuse_weight": FUSE_WEIGHT, "results": results}, open(out, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n=== {qs}（narrative×{NARRATIVE_REP}固定, composition=both, fuse重み{FUSE_WEIGHT}）===")
        print(f"{'Subquery回数':<14}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for sr in SUBQUERY_REPS:
            v = results[f"subrep{sr}"].get(qs)
            if v: print(f"{sr:<14}"+"".join(f"{v[m]:>10.4f}" for m in METRIC_KEYS))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
