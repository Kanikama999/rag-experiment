"""
recall記録パイプライン（sub+narr|both|se200sb30|fuse3、consensus R@1000=0.3266が最良）を
土台に、クエリ長と検索時間をどこまで削れるかを調べる。

背景:
既存の long_shrink 系実験はすべて champion（span_terms=Subquery単独、se100/sb15、
composition=疑似文書のみ）を土台にしていた。今回のrecall記録パイプラインは
span_terms=Subquery+narrative、se200/sb30、composition=両方(疑似文書+拡張語リスト)と
土台が異なるため、削減余地を測り直す必要がある。

事前に確認済み: composition=champion単独 と termlist単独 は recall がほぼ同値
（consensus R@1000≈0.3107で完全に並ぶ）だが、両方使う both だけが +0.016〜0.019 高い。
したがって拡張語リストは落とせない（生成コストは既に払い済みでもある）。
削る対象は narrative の繰り返し回数と、疑似文書bodyの語数の2軸のみとする
（拡張語リストは常にフル、8/20/40語の構造で既に十分小さい）。

削る軸:
  【軸1】narrative 繰り返し回数（0,1,2,3,5）。疑似文書bodyは常にフル。
  【軸2】疑似文書bodyの語数（0,25,50,100,フル）。narrative×5固定。

short側（Subquery単独、位置ブーストなし）は既存キャッシュを再利用。融合重みは
recall記録の fuse3 に合わせる（fuse0=融合なしも併記して融合の効果も見る）。

検索時間も記録する（1条件あたりの累計秒数、105トピック分）。

【検算】narr5_bodyfull|fuse0 が sub+narr|both|se200sb30|fuse0 に、
narr5_bodyfull|fuse3 が sub+narr|both|se200sb30|fuse3（recall記録そのもの、
consensus R@1000=0.3266）に一致するはず。

使い方:
    python3 evaluate_recall_shrink.py [LIMIT]
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
CACHE = os.path.join(RAG_DIR, "_cache_recall_shrink.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K = 1000, 1000
SPAN_END, SPAN_BOOST = 200, 30.0
FUSE_WEIGHTS = [0, 3]
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


def conditions():
    out = [("narr0", 0, None), ("narr1", 1, None), ("narr2", 2, None),
           ("narr3", 3, None), ("narr5_bodyfull", 5, None)]
    for w in (0, 25, 50, 100):
        out.append((f"narr5_body{w}", 5, w))
    return out


def build_body(pdoc, body_words):
    if body_words is None:
        return pdoc["body"]
    return " ".join(pdoc["body"].split()[:body_words])


def search_one(label, qid, i, narrative, dq, pdoc, terms, n_rep, body_words):
    rep = " ".join([narrative] * n_rep) if n_rep else ""
    tt = " ".join(terms["title_terms"])
    th = " ".join(terms["heading_terms"])
    tb = " ".join(terms["body_terms"])
    t = f"{rep} {dq} {pdoc['title']} {tt}".strip()
    h = f"{rep} {dq} {' '.join(pdoc['headings'])} {th}".strip()
    b = f"{rep} {dq} {build_body(pdoc, body_words)} {tb}".strip()
    span = list(dict.fromkeys(analyze_terms(dq) + (analyze_terms(narrative) if n_rep else [])))
    t0 = time.time()
    lst = bm25_equalweight_posboost_discourseboost(
        t, h, b, span, k=RETRIEVE_K, span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
    dt = time.time() - t0
    return label, qid, i, [d for d, _ in lst], dt


def build(qids, queries, webstyle, termlist):
    if os.path.exists(CACHE):
        with gzip.open(CACHE, "rt") as f:
            c = json.load(f)
        if c.get("n_topics") == len(qids):
            print(f"検索キャッシュを再利用（{c['n']}件）")
            return c["rankings"], c["timing"]
    conds = conditions()
    jobs = []
    for qid in qids:
        we, te = webstyle[qid], termlist[qid]
        for i, dq in enumerate(we["decomposed_queries"]):
            pdoc = we["query2doc_docs_structured"][i]
            if not (pdoc and pdoc.get("body")):
                continue
            for label, n_rep, body_words in conds:
                jobs.append((label, qid, i, queries[qid], dq, pdoc,
                              te["query2doc_docs_terms"][i], n_rep, body_words))
    print(f"{len(conds)} 条件 × Subquery = {len(jobs)} 検索...")
    out, timing, t0 = {}, {c[0]: 0.0 for c in conds}, time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(search_one, *j) for j in jobs]
        for n, f in enumerate(futs, 1):
            label, qid, i, ids, dt = f.result()
            out.setdefault(label, {}).setdefault(qid, {})[str(i)] = ids
            timing[label] += dt
            if n % 500 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(jobs)}  {el/60:.1f}分  残り約{el/n*(len(jobs)-n)/60:.0f}分", flush=True)
    with gzip.open(CACHE, "wt") as f:
        json.dump({"n_topics": len(qids), "n": len(jobs), "rankings": out, "timing": timing}, f)
    print(f"-> キャッシュ保存（{(time.time()-t0)/60:.1f}分）")
    return out, timing


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


def word_budget(n_rep, body_words, narr_avg=45.2, dq_avg=10, title_avg=8.6,
                 head_avg=19.6, body_avg=175.3, termlist_avg=217.0):
    bw = body_avg if body_words is None else min(body_words, body_avg)
    return round(narr_avg * n_rep + dq_avg + title_avg + head_avg + bw + termlist_avg)


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

    RK, TIMING = build(qids, queries, webstyle, termlist)
    with gzip.open(MATCACHE, "rt") as f:
        SUBQ = json.load(f)["rankings"]["0"]     # short側: Subquery単独、位置ブーストなし

    results = {}
    for label, per in RK.items():
        n_rep, body_words = {c[0]: (c[1], c[2]) for c in conditions()}[label]
        wb = word_budget(n_rep, body_words)
        for fw in FUSE_WEIGHTS:
            run = {}
            for qid in qids:
                lists, ws = [], []
                for i, ids in per.get(qid, {}).items():
                    lists.append([(d, 1.0) for d in ids]); ws.append(float(fw) if fw else 1.0)
                if fw:
                    for i, d in SUBQ.get(qid, {}).items():
                        if "subquery" in d:
                            lists.append([(x, 1.0) for x in d["subquery"]]); ws.append(1.0)
                run[qid] = {dd: float(s) for dd, s in
                            (rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else [])}
            rec = {"narrative_rep": n_rep, "body_words": body_words, "word_budget": wb,
                    "fuse_weight": fw, "search_sec": round(TIMING[label], 1)}
            for qs in QREL_SETS:
                rec[qs] = evaluate(run, qrels[qs], qids)
            results[f"{label}|fuse{fw}"] = rec

    out = os.path.join(RAG_DIR, "recall_shrink_result.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

    axis1 = ["narr0", "narr1", "narr2", "narr3", "narr5_bodyfull"]
    axis2 = ["narr5_body0", "narr5_body25", "narr5_body50", "narr5_body100", "narr5_bodyfull"]
    for qs in QREL_SETS:
        print(f"\n{'='*100}\n=== {qs} ===\n{'='*100}")
        for axis_name, axis in (("軸1: narrative繰り返し回数（疑似文書bodyフル）", axis1),
                                  ("軸2: 疑似文書bodyの語数（narrative×5固定）", axis2)):
            print(f"\n[{axis_name}]")
            print(f"{'条件':<18}{'語数':>6}{'検索秒(105topic)':>16}  {'fuse0 R@1000':>13}{'nDCG':>8}  {'fuse3 R@1000':>13}{'nDCG':>8}")
            for label in axis:
                r0 = results[f"{label}|fuse0"]
                r3 = results[f"{label}|fuse3"]
                v0, v3 = r0.get(qs), r3.get(qs)
                if not v0 or not v3: continue
                print(f"{label:<18}{r0['word_budget']:>6}{r0['search_sec']:>16.1f}  "
                      f"{v0['recall_1000']:>13.4f}{v0['ndcg_cut_10']:>8.4f}  "
                      f"{v3['recall_1000']:>13.4f}{v3['ndcg_cut_10']:>8.4f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
