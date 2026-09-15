"""
これまで独立に検証された改善を組み合わせて、最高性能（consensus nDCG@10 0.5368）を超えられるか。

現在の記録:
    nDCG@10    0.5368  spanterms_plus_narrative（§2.1）※recall@1000 は 0.2988 と落ちる
    recall@1000 0.3178  IDF項重み付け + 位置ブースト + Subquery×5（§7.6）
    champion   0.5251 / 0.3095

未検証の組み合わせ（それぞれ単独では効くことが確認済み）:
  1. span_terms に narrative を足す        単独で nDCG +0.0117（§2.1）
  2. 疑似参照文書を拡張語リストに差し替える     grid で narr×5+拡張 が champion と同値かつ recall 上
                                          （疑似文書の Shapley は8セルすべて負）
  3. 短いクエリとの重み付きRRF融合           位置ブーストなしで nDCG +0.0329。ありでは long×3 が
                                          探索の端で単調増加のまま（0.5251→0.5270）
  4. span_end / span_boost の拡張          (200,30) が (100,15) より上（§5、35トピック）

この3因子×2水準（1・2・4）で 12 条件を検索し、3 は既存の短クエリキャッシュを使って
オフラインで重み {なし,3,5,8} を掛ける。検索は 12 セットで済み、融合は無料。

【検算】span_terms=sub / 構成=champion / span=(100,15) / 融合なし が champion に一致するはず
（consensus 0.0689 / 0.3095 / 0.5251 / 0.7769）。

使い方:
    python3 evaluate_best_combination.py [LIMIT]
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

from retriever import (bm25_equalweight_posboost_discourseboost, analyze_terms, rrf_fuse)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
TERMLIST_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_termlist_T8_H20_B40.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
MATCACHE = os.path.join(RAG_DIR, "_cache_material_rankings.json.gz")
CACHE = os.path.join(RAG_DIR, "_cache_best_combination.json.gz")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
N_WORKERS = 8
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

SPANTERMS = ["sub", "sub+narr"]                 # 因子1
COMPOS = ["champion", "termlist", "both"]       # 因子2: 疑似文書 / 拡張語リスト / 両方
SPANP = [(100, 15.0), (200, 30.0)]              # 因子4
FUSEW = [0, 3, 5, 8]                            # 因子3: 0=融合なし、それ以外は long の重み

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


def fields_for(compo, rep, dq, pdoc, terms):
    hd = " ".join(pdoc["headings"])
    if compo == "champion":
        return (f"{rep} {dq} {pdoc['title']}", f"{rep} {dq} {hd}", f"{rep} {dq} {pdoc['body']}")
    tt, th, tb = (" ".join(terms["title_terms"]), " ".join(terms["heading_terms"]),
                   " ".join(terms["body_terms"]))
    if compo == "termlist":
        return (f"{rep} {dq} {tt}", f"{rep} {dq} {th}", f"{rep} {dq} {tb}")
    return (f"{rep} {dq} {pdoc['title']} {tt}", f"{rep} {dq} {hd} {th}",
            f"{rep} {dq} {pdoc['body']} {tb}")


def search_one(cond, qid, i, narrative, dq, pdoc, terms, sterms):
    st, compo, (se, sb) = cond
    rep = " ".join([narrative] * QUERY_REPEAT)
    t, h, b = fields_for(compo, rep, dq, pdoc, terms)
    lst = bm25_equalweight_posboost_discourseboost(
        t, h, b, sterms, k=RETRIEVE_K, span_end=se, span_boost=sb, markers=[])
    return cond, qid, i, [d for d, _ in lst]


def cname(cond):
    st, compo, sp = cond
    return f"{st}|{compo}|se{sp[0]}sb{int(sp[1])}"


def build(qids, queries, webstyle, termlist):
    if os.path.exists(CACHE):
        with gzip.open(CACHE, "rt") as f:
            c = json.load(f)
        if c.get("n_topics") == len(qids):
            print(f"検索キャッシュを再利用（{c['n']}件）")
            return c["rankings"]
    conds = [(st, cp, sp) for st in SPANTERMS for cp in COMPOS for sp in SPANP]
    jobs = []
    for qid in qids:
        we, te = webstyle[qid], termlist[qid]
        narr_terms = analyze_terms(queries[qid], field="body")
        for i, dq in enumerate(we["decomposed_queries"]):
            pdoc = we["query2doc_docs_structured"][i]
            if not (pdoc and pdoc.get("body")):
                continue
            sub_terms = analyze_terms(dq, field="body")
            plus = list(dict.fromkeys(sub_terms + narr_terms))
            for cond in conds:
                st = cond[0]
                jobs.append((cond, qid, i, queries[qid], dq, pdoc,
                              te["query2doc_docs_terms"][i],
                              sub_terms if st == "sub" else plus))
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
        MAT = json.load(f)["rankings"]["0"]      # 短クエリは位置ブーストなし側を使う

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
            rec = {"cond": cn, "fuse_long_weight": w}
            for qs in QREL_SETS:
                rec[qs] = evaluate(run, qrels[qs], qids)
            results[f"{cn}|fuse{w}"] = rec

    out = os.path.join(RAG_DIR, "best_combination_result.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

    for qs in ("consensus", "coverage"):
        rows = [(k, v) for k, v in results.items() if v[qs]]
        print(f"\n=== {qs}  nDCG@10 上位15 ===")
        print(f"{'条件':<44}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for k, v in sorted(rows, key=lambda x: -x[1][qs]["ndcg_cut_10"])[:15]:
            a = v[qs]
            print(f"{k:<44}"+"".join(f"{a[m]:>10.4f}" for m in METRIC_KEYS))
        print(f"--- {qs}  recall@1000 上位5 ---")
        for k, v in sorted(rows, key=lambda x: -x[1][qs]["recall_1000"])[:5]:
            a = v[qs]
            print(f"{k:<44}"+"".join(f"{a[m]:>10.4f}" for m in METRIC_KEYS))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
