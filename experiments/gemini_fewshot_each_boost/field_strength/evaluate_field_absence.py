"""
title欠損文書の取りこぼしを、フィールド結合方式を変えて回収できるか検証する。

背景（診断済み）:
grade>=2 の正解のうち **title欠損文書の発見率は52.4%** で、titleがある文書の88.7%に対し
**36.3pt低い**。title欠損文書は grade>=2 正解の6.2%（296件）で、完全に解消できても
追加で取れるのは108件（grade>=2 未発見644件の16.8%）。効果は本物だが母集団は小さい。

メカニズムの仮説:
現行の bool/should 線形和は **各フィールドが独立に飽和関数 tf/(k1+tf) を適用する**ため、
3フィールドに散って出現する文書は最大3個ぶんの加点を得る。titleが空の文書は加点源が
構造的に1つ失われる。BM25F と cross_fields はどちらも **フィールド横断で統計を合算してから
1回だけ飽和/IDF を適用する**ので、この不利が消えるはず。

さらにこれは §3.4 の未説明の非対称——「BM25F は nDCG では線形和に負けるのに recall では
両qrelsで勝つ」——の説明候補でもある。**BM25F の recall 優位が title欠損層で説明できるか**を
層別 recall で直接確かめる。

条件（クエリ構成は champion と同じ narrative×5 + Subquery + 疑似文書。位置ブーストなしで
フィールド結合方式だけを比較する。位置ブーストは span_first で body だけを見るため、
フィールド結合の議論に無関係なノイズを持ち込む）:

  linear      : 現行。フィールド別テキストを bool/should で線形和（均等重み）
  crossfields : 連結した1本のテキストを multi_match type=cross_fields で3フィールドへ
  bm25f_qtf   : フィールド別テキスト。tf合算→単一飽和。qtf修正版（§3.4）

出力は集計指標に加えて、**title有無で層別した grade>=2 正解の発見率**。

使い方:
    python3 evaluate_field_absence.py [LIMIT]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
from retriever import (client, INDEX, bm25_fielded, bm25_crossfields, bm25f_prepare,
                        analyze_terms, rrf_fuse, _bm25f_avgdl, _bm25f_total_docs,
                        _bm25f_combined_df)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PARENT_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(PARENT_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
TITLEMAP = os.path.join(RAG_DIR, "_cache_relevant_title_presence.json")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
BM25_K1 = 0.9
B_TITLE, B_HEADINGS, B_BODY = 0.6, 0.2, 0.2
N_WORKERS = 8
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

METHODS = ["linear", "crossfields", "bm25f_qtf"]
METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


def load_qrels(path):
    q = {}
    for line in open(path):
        s = line.split()
        if len(s) == 4 and not line.startswith("#"):
            q.setdefault(s[0], {})[s[2]] = int(s[3])
    return q


def load_queries(path):
    q = {}
    for line in open(path):
        line = line.strip()
        if line:
            o = json.loads(line)
            q[o["id"]] = o["title"]
    return q


def bm25f_qtf_search(title_q, headings_q, body_q, logical_q, k):
    """§3.4 の qtf 修正版 BM25F。クエリ側の出現回数を線形に反映する。"""
    raw = bm25f_prepare(title_q, headings_q, body_q,
                         title_weight=1.0, headings_weight=1.0, body_weight=1.0,
                         candidate_k=k, rerank_n=k)
    if raw is None or not raw["query_terms"]:
        return [] if raw is None else [(d, 0.0) for d in raw["candidates"][:k]]
    an = client.indices.analyze(index=INDEX, body={"field": "body", "text": logical_q})
    qtf = Counter(t["token"] for t in an.get("tokens", []))
    idf, avgdl, docs = raw["idf"], raw["avgdl"], raw["docs"]
    ws = {"title": 1.0, "headings": 1.0, "body": 1.0}
    bs = {"title": B_TITLE, "headings": B_HEADINGS, "body": B_BODY}
    scores = {}
    for docid, doc in docs.items():
        ft, fl = doc["field_terms"], doc["field_len"]
        sc = 0.0
        for t in raw["query_terms"]:
            ti = idf.get(t, 0.0)
            if ti <= 0:
                continue
            ptf = 0.0
            for f in ("title", "headings", "body"):
                info = ft[f].get(t)
                if not info:
                    continue
                B = (1 - bs[f]) + bs[f] * (fl[f] / avgdl[f]) if avgdl[f] else 1.0
                ptf += ws[f] * info["term_freq"] / B
            if ptf > 0:
                sc += ti * ptf / (BM25_K1 + ptf) * qtf.get(t, 1)
        scores[docid] = sc
    return sorted(scores.items(), key=lambda x: -x[1])[:k]


def run_topic(method, qid, narrative, entry):
    rep = " ".join([narrative] * QUERY_REPEAT)
    lists = []
    for i, dq in enumerate(entry["decomposed_queries"]):
        pdoc = entry["query2doc_docs_structured"][i]
        if not (pdoc and pdoc.get("body")):
            continue
        hd = " ".join(pdoc["headings"])
        t = f"{rep} {dq} {pdoc['title']}"
        h = f"{rep} {dq} {hd}"
        b = f"{rep} {dq} {pdoc['body']}"
        if method == "linear":
            lists.append(bm25_fielded(t, h, b, k=RETRIEVE_K,
                                       title_boost=1, headings_boost=1, body_boost=1))
        elif method == "crossfields":
            combined = f"{rep} {dq} {pdoc['title']} {hd} {pdoc['body']}"
            lists.append(bm25_crossfields(combined, k=RETRIEVE_K,
                                           title_boost=1, headings_boost=1, body_boost=1))
        else:
            logical = f"{rep} {dq} {pdoc['title']} {hd} {pdoc['body']}"
            lists.append(bm25f_qtf_search(t, h, b, logical, RETRIEVE_K))
    lists = [l for l in lists if l]
    return qid, {d: float(s) for d, s in (rrf_fuse(lists, top_n=TOPK) if lists else [])}


def title_presence(docids):
    """正解文書のtitle有無を引く（キャッシュする）。"""
    cache = json.load(open(TITLEMAP)) if os.path.exists(TITLEMAP) else {}
    need = [d for d in docids if d not in cache]
    for i in range(0, len(need), 400):
        ch = need[i:i + 400]
        r = client.mget(index=INDEX, body={"ids": ch}, params={"_source_includes": "title"})
        for d, h in zip(ch, r["docs"]):
            cache[d] = bool((h.get("_source", {}).get("title") or "").strip()) if h.get("found") else None
    json.dump(cache, open(TITLEMAP, "w"))
    return cache


def evaluate(run, qrels, qids):
    target = [q for q in qids if q in qrels and run.get(q)]
    if not target:
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
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

    runs = {}
    for method in METHODS:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(run_topic, method, q, queries[q], webstyle[q]) for q in qids]
            runs[method] = dict(f.result() for f in futs)
        print(f"  {method}: {time.time()-t0:.0f}s", flush=True)

    # grade>=2 正解のtitle有無
    hi = {}
    for qs in QREL_SETS:
        for qid in qids:
            for d, g in qrels[qs].get(qid, {}).items():
                if g >= 2:
                    hi.setdefault(qs, set()).add((qid, d))
    tp = title_presence(sorted({d for s in hi.values() for _, d in s}))

    summary, strat = {}, {}
    for qs in QREL_SETS:
        for m in METHODS:
            summary.setdefault(qs, {})[m] = evaluate(runs[m], qrels[qs], qids)
            yes = [0, 0]; no = [0, 0]
            for qid, d in hi.get(qs, ()):
                t = tp.get(d)
                if t is None:
                    continue
                bucket = yes if t else no
                bucket[1] += 1
                if d in runs[m].get(qid, {}):
                    bucket[0] += 1
            strat.setdefault(qs, {})[m] = {"title_yes": yes, "title_no": no}

    out = os.path.join(RAG_DIR, "field_absence_result.json")
    json.dump({"n_topics": len(qids), "summary": summary, "stratified": strat},
              open(out, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print(f"{'手法':<14}"+"".join(f"{k:>11}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for m in METHODS:
            a = summary[qs][m]
            if a: print(f"{m:<14}"+"".join(f"{a[k]:>11.4f}" for k in METRIC_KEYS))
        print(f"\n  grade>=2 正解の発見率（title有無で層別）")
        print(f"  {'手法':<14}{'titleあり':>12}{'title欠損':>12}{'差':>10}")
        for m in METHODS:
            s = strat[qs][m]; y, n = s["title_yes"], s["title_no"]
            ry = y[0]/y[1]*100 if y[1] else 0; rn = n[0]/n[1]*100 if n[1] else 0
            print(f"  {m:<14}{ry:>11.1f}%{rn:>11.1f}%{ry-rn:>9.1f}pt   ({y[0]}/{y[1]}, {n[0]}/{n[1]})")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
