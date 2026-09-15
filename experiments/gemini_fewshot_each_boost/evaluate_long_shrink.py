"""
long+short融合の「long」側は、どこまで削っても性能を保てるか。

背景:
evaluate_longshort_fusion.py で「long（narrative×5 + Subquery + 疑似文書、body中央値176語・
全体で平均440語規模）+ subquery（Subquery単独、約10語）」のRRF融合が、位置ブーストなしで
long単独を全指標で上回ることを確認した（nDCG@10 +0.0329）。short側が「取りこぼしの穴」を
埋める形になっているなら、long側は今ほど長く保つ必要がないかもしれない。LLM生成コストの
観点でも、疑似文書のbodyを短く生成できれば節約になる。

削る軸を2つ独立に振る:

  【軸1】narrative の繰り返し回数（0,1,2,3,5）。疑似文書は常にフル。
         現行championは5。0はnarrativeを一切使わない構成。
  【軸2】疑似文書bodyの語数（0,25,50,100,フル≈176語中央値）。narrative×5は固定。
         0は疑似文書を実質使わない構成（title/headingsは残す。これらは平均8.6語/19.6語と
         元々安いので削る対象にしない）。フルは現行championと同一。

各水準で「long単独」と「long + subquery（RRF融合、等重み）」の両方を評価する。
short側（subquery単独ランキング）は既存キャッシュ（_cache_material_rankings.json.gz の
pb0側）を再利用するので追加検索は無い。位置ブーストは使わない
（evaluate_longshort_fusion.py の実験で「融合は位置ブーストなしでのみ明確に効く」
ことが分かっているため。位置ブーストありでの縮小は別途の検討が要る）。

【検算】narrative_rep=5・body_words=フル・fuse無し が champion の「位置ブーストなし版」
（evaluate_query_component_grid.py の N1_S1_P1_E0_pb0 = nDCG@10 0.4610）に一致するはず。

使い方:
    python3 evaluate_long_shrink.py [LIMIT]
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import bm25_fielded, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
MATCACHE = os.path.join(RAG_DIR, "_cache_material_rankings.json.gz")
CACHE = os.path.join(RAG_DIR, "_cache_long_shrink.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K = 1000, 1000
N_WORKERS = 8
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

NARRATIVE_LEVELS = [0, 1, 2, 3, 5]     # 軸1（疑似文書はフル）
BODY_LEVELS = [0, 25, 50, 100, None]   # 軸2（narrative×5固定）。None=フル

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
    """(label, narrative_rep, body_words) のリスト。フル×5 は両軸で共有するので重複させない。"""
    out = [("narr0", 0, None), ("narr1", 1, None), ("narr2", 2, None),
           ("narr3", 3, None), ("narr5_bodyfull", 5, None)]
    for w in BODY_LEVELS:
        if w is None:
            continue
        out.append((f"narr5_body{w}", 5, w))
    return out


def build_body(pdoc, body_words):
    body = pdoc["body"]
    if body_words is None:
        return body
    return " ".join(body.split()[:body_words])


def search_long(label, qid, i, narrative, dq, pdoc, n_rep, body_words):
    rep = " ".join([narrative] * n_rep) if n_rep else ""
    t = f"{rep} {dq} {pdoc['title']}".strip()
    h = f"{rep} {dq} {' '.join(pdoc['headings'])}".strip()
    b = f"{rep} {dq} {build_body(pdoc, body_words)}".strip()
    lst = bm25_fielded(t, h, b, k=RETRIEVE_K, title_boost=1, headings_boost=1, body_boost=1)
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
            for label, n_rep, body_words in conds:
                jobs.append((label, qid, i, queries[qid], dq, pdoc, n_rep, body_words))
    print(f"{len(conds)} 条件 × Subquery = {len(jobs)} 検索...")
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
    """このconditionのクエリ本文の推定総語数（1 Subqueryぶん、3フィールド合計）。"""
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
        SUBQ = json.load(f)["rankings"]["0"]     # short側: Subquery単独、位置ブーストなし

    conds = conditions()
    cond_map = {label: (n, w) for label, n, w in conds}

    results = {}
    for label, per in RK.items():
        n_rep, body_words = cond_map[label]
        wb = word_budget(n_rep, body_words)
        for mode in ("alone", "fused"):
            run = {}
            for qid in qids:
                lists = [[(d, 1.0) for d in ids] for i, ids in per.get(qid, {}).items()]
                if mode == "fused":
                    for i, d in SUBQ.get(qid, {}).items():
                        if "subquery" in d:
                            lists.append([(x, 1.0) for x in d["subquery"]])
                run[qid] = {dd: float(s) for dd, s in
                            (rrf_fuse(lists, top_n=TOPK) if lists else [])}
            rec = {"narrative_rep": n_rep, "body_words": body_words,
                    "word_budget": wb, "mode": mode}
            for qs in QREL_SETS:
                rec[qs] = evaluate(run, qrels[qs], qids)
            results[f"{label}|{mode}"] = rec

    out = os.path.join(RAG_DIR, "long_shrink_result.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n{'='*90}\n=== {qs} ===\n{'='*90}")
        print("\n[軸1] narrative 繰り返し回数（疑似文書フル）")
        print(f"{'条件':<18}{'語数目安':>9}  {'--- long単独 ---':^38}  {'--- +subquery融合 ---':^38}")
        print(f"{'':<18}{'':>9}  "+"".join(f"{k:>9}" for k in ("R@100","R@1000","nDCG@10","P@100"))
              +"  "+"".join(f"{k:>9}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for label in ["narr0","narr1","narr2","narr3","narr5_bodyfull"]:
            a, f = results[f"{label}|alone"], results[f"{label}|fused"]
            wb = a["word_budget"]
            av, fv = a.get(qs), f.get(qs)
            if not av or not fv: continue
            print(f"{label:<18}{wb:>9}  "+"".join(f"{av[m]:>9.4f}" for m in METRIC_KEYS)
                  +"  "+"".join(f"{fv[m]:>9.4f}" for m in METRIC_KEYS))

        print("\n[軸2] 疑似文書bodyの語数（narrative×5固定）")
        print(f"{'条件':<18}{'語数目安':>9}  {'--- long単独 ---':^38}  {'--- +subquery融合 ---':^38}")
        for label in ["narr5_body0","narr5_body25","narr5_body50","narr5_body100","narr5_bodyfull"]:
            a, f = results[f"{label}|alone"], results[f"{label}|fused"]
            wb = a["word_budget"]
            av, fv = a.get(qs), f.get(qs)
            if not av or not fv: continue
            print(f"{label:<18}{wb:>9}  "+"".join(f"{av[m]:>9.4f}" for m in METRIC_KEYS)
                  +"  "+"".join(f"{fv[m]:>9.4f}" for m in METRIC_KEYS))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
