"""
evaluate_long_shrink.py の位置ブーストあり版。

背景:
位置ブーストなしでは「long（narrative×n + Subquery + 疑似文書）を削って subquery と融合」
したところ、疑似文書bodyは25語程度が最良でフル(176語)は過剰・有害、narrativeの繰り返しは
5回でもまだ単調増加中という結果が出た（105トピック確定、evaluate_long_shrink.py）。

ただし evaluate_longshort_fusion.py で「位置ブーストありでは等倍(long×1)のRRF融合は性能を
落とし、long側を3倍以上に重くして初めて championにわずかに勝てる」ことが分かっている。
したがって今回は融合重みも振る（0=融合なし, 1, 3, 5, 8）。

削る軸は long_shrink と同じ2つ:
  【軸1】narrative の繰り返し回数（0,1,2,3,5）。疑似文書は常にフル。
  【軸2】疑似文書bodyの語数（0,25,50,100,フル）。narrative×5固定。

位置ブーストは champion と同じ span_end=100, span_boost=15、判定語は Subquery
（span_terms=analyze_terms(dq)）。short側（Subquery単独ランキング、位置ブーストあり）は
既存キャッシュ（_cache_material_rankings.json.gz の pb=1 側）を再利用する。

【検算】narrative_rep=5・body_words=フル・fuse無し が champion に一致するはず
（consensus 0.0689/0.3095/0.5251/0.7769、coverage 0.2373/0.5995/0.4049/0.3768）。

使い方:
    python3 evaluate_long_shrink_posboost.py [LIMIT]
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
CACHE = os.path.join(RAG_DIR, "_cache_long_shrink_posboost.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K = 1000, 1000
SPAN_END, SPAN_BOOST = 100, 15.0
N_WORKERS = 8
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

NARRATIVE_LEVELS = [0, 1, 2, 3, 5]
BODY_LEVELS = [0, 25, 50, 100, None]
FUSE_WEIGHTS = [0, 1, 3, 5, 8]     # 0=融合なし（long単独）

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
    for w in BODY_LEVELS:
        if w is None:
            continue
        out.append((f"narr5_body{w}", 5, w))
    return out


def build_body(pdoc, body_words):
    if body_words is None:
        return pdoc["body"]
    return " ".join(pdoc["body"].split()[:body_words])


def search_long(label, qid, i, narrative, dq, pdoc, n_rep, body_words, span):
    rep = " ".join([narrative] * n_rep) if n_rep else ""
    t = f"{rep} {dq} {pdoc['title']}".strip()
    h = f"{rep} {dq} {' '.join(pdoc['headings'])}".strip()
    b = f"{rep} {dq} {build_body(pdoc, body_words)}".strip()
    lst = bm25_equalweight_posboost_discourseboost(
        t, h, b, span, k=RETRIEVE_K, span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
    return label, qid, i, [d for d, _ in lst]


def build(qids, queries, webstyle):
    if os.path.exists(CACHE):
        with gzip.open(CACHE, "rt") as f:
            c = json.load(f)
        if c.get("n_topics") == len(qids):
            print(f"検索キャッシュを再利用（{c['n']}件）")
            return c["rankings"]
    conds = conditions()
    jobs = []
    for qid in qids:
        e = webstyle[qid]
        for i, dq in enumerate(e["decomposed_queries"]):
            pdoc = e["query2doc_docs_structured"][i]
            if not (pdoc and pdoc.get("body")):
                continue
            span = analyze_terms(dq)
            for label, n_rep, body_words in conds:
                jobs.append((label, qid, i, queries[qid], dq, pdoc, n_rep, body_words, span))
    print(f"{len(conds)} 条件 × Subquery = {len(jobs)} 検索（位置ブーストあり）...")
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(search_long, *j) for j in jobs]
        for n, f in enumerate(futs, 1):
            label, qid, i, ids = f.result()
            out.setdefault(label, {}).setdefault(qid, {})[str(i)] = ids
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


def word_budget(n_rep, body_words, narr_avg=45.2, dq_avg=10, title_avg=8.6, head_avg=19.6, body_avg=175.3):
    bw = body_avg if body_words is None else min(body_words, body_avg)
    return round(narr_avg * n_rep + dq_avg + title_avg + head_avg + bw)


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
        SUBQ = json.load(f)["rankings"]["1"]     # short側: Subquery単独、位置ブーストあり

    conds = conditions()
    cond_map = {label: (n, w) for label, n, w in conds}

    results = {}
    for label, per in RK.items():
        n_rep, body_words = cond_map[label]
        wb = word_budget(n_rep, body_words)
        for fw in FUSE_WEIGHTS:
            run = {}
            for qid in qids:
                lists, ws = [], []
                for i, ids in per.get(qid, {}).items():
                    lists.append([(d, 1.0) for d in ids])
                    ws.append(float(fw) if fw else 1.0)
                if fw:
                    for i, d in SUBQ.get(qid, {}).items():
                        if "subquery" in d:
                            lists.append([(x, 1.0) for x in d["subquery"]])
                            ws.append(1.0)
                run[qid] = {dd: float(s) for dd, s in
                            (rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else [])}
            rec = {"narrative_rep": n_rep, "body_words": body_words,
                    "word_budget": wb, "fuse_weight": fw}
            for qs in QREL_SETS:
                rec[qs] = evaluate(run, qrels[qs], qids)
            results[f"{label}|fuse{fw}"] = rec

    out = os.path.join(RAG_DIR, "long_shrink_posboost_result.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

    axis1 = ["narr0", "narr1", "narr2", "narr3", "narr5_bodyfull"]
    axis2 = ["narr5_body0", "narr5_body25", "narr5_body50", "narr5_body100", "narr5_bodyfull"]
    for qs in QREL_SETS:
        print(f"\n{'='*100}\n=== {qs}（位置ブーストあり） ===\n{'='*100}")
        for axis_name, axis in (("軸1: narrative繰り返し回数（疑似文書フル）", axis1),
                                  ("軸2: 疑似文書bodyの語数（narrative×5固定）", axis2)):
            print(f"\n[{axis_name}]  各fuse重みでのnDCG@10")
            print(f"{'条件':<18}{'語数':>7}"+"".join(f"  fuse{w:<4}" for w in FUSE_WEIGHTS))
            for label in axis:
                wb = results[f"{label}|fuse0"]["word_budget"]
                row = []
                for w in FUSE_WEIGHTS:
                    v = results[f"{label}|fuse{w}"].get(qs)
                    row.append(f"{v['ndcg_cut_10']:.4f}" if v else "  -   ")
                print(f"{label:<18}{wb:>7}  "+"  ".join(row))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
