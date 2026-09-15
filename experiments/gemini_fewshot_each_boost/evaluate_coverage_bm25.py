"""
「短い文書が高いスコアになる」を候補生成の段階から実現する（リランカーではない）。

背景:
通常のBM25 matchはtf(出現回数)×文書長正規化を積み上げる。長い文書はクエリの多数の語に
散らばって触れられる余地があるため、大して関連していなくても出現回数の絶対値でスコアが
積み上がる。短い文書は的確でも出現回数で勝負にならない。

これまで試したspan_first位置ブーストは基礎スコアへの固定加点でしかなく、リランカー
（候補プール生成後の並べ替え）は候補プール自体が既に長い文書で埋まっている問題を
解決できなかった（grade>=2未発見文書は本文語数中央値1160語 vs 発見済み1751語、
短文書(<800語)率26% vs 12%——候補プールの組成自体が長さに偏っている）。

提案: bodyフィールドのスコアを「出現回数」ではなく「その語が1回でも出現するか(0/1)」
だけで計算し、出現すればIDF値をそのまま加点する。文書長にもtf(出現回数)にも一切
依存しないため、100語の文書でも18,000語の文書でも、同じ語を含んでいれば同じ点数になる。

これはOpenSearchのconstant_score + bool/shouldだけで組めるnativeクエリなので、
_mtermvectorsやオンザフライのterm vector計算が不要——**通常のmatchクエリと同じ
候補生成の1パスで完結し**、リランカー的な「候補生成→再スコア」の2段構成にならない。
すなわち候補プール自体がこの公平な基準で決まる。

条件（title/headingsは短いフィールドなので通常のmatchのまま、bodyだけ置き換える）:
    champion  : 現行（body全体の通常BM25 match）。既存championを再現する検算
    coverage  : body を IDF被覆スコア（Σ idf(t) for t in query_terms if t in body）に置換

使い方:
    python3 evaluate_coverage_bm25.py [SUBSET_STRIDE]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import (client, INDEX, bm25_fielded, analyze_terms, rrf_fuse,
                        _bm25f_total_docs, _bm25f_combined_df)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
N_WORKERS = 8
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1

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


def bm25_coverage_body(title_text, headings_text, body_text, k=100,
                        title_boost=1, headings_boost=1):
    """title/headingsは通常のmatch、bodyだけIDF被覆スコアに置換する。
    出現回数(tf)も文書長も一切使わない——長い文書が同じ語を繰り返して積み上げる有利を
    構造的に消す。IDFは既存のbm25f系と同じ定義（title/headings/body横断のcombined_df）
    を使い、他実験との比較可能性を保つ。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        N = _bm25f_total_docs()
        for t in analyze_terms(body_text, field="body"):
            df = _bm25f_combined_df(t)
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0
            if idf > 0:
                should.append({"constant_score": {"filter": {"term": {"body": t}}, "boost": idf}})
    if not should:
        return []
    res = client.search(index=INDEX, body={
        "size": k, "_source": False, "query": {"bool": {"should": should}}})
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


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
        if method == "champion":
            lists.append(bm25_fielded(t, h, b, k=RETRIEVE_K,
                                       title_boost=1, headings_boost=1, body_boost=1))
        else:
            lists.append(bm25_coverage_body(t, h, b, k=RETRIEVE_K))
    lists = [l for l in lists if l]
    return qid, {d: float(s) for d, s in (rrf_fuse(lists, top_n=TOPK) if lists else [])}


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
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}
    qids = sorted(q for q in webstyle if q in queries and webstyle[q].get("decomposed_queries"))
    qids = qids[::SUBSET_STRIDE]
    print(f"{len(qids)}トピック")

    METHODS = ["champion", "coverage"]
    runs = {}
    for method in METHODS:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = [ex.submit(run_topic, method, q, queries[q], webstyle[q]) for q in qids]
            runs[method] = dict(f.result() for f in futs)
        print(f"  {method}: {time.time()-t0:.0f}s", flush=True)

    results = {}
    for qs in QREL_SETS:
        for m in METHODS:
            results.setdefault(qs, {})[m] = evaluate(runs[m], qrels[qs], qids)

    suffix = "" if SUBSET_STRIDE == 1 else f"_stride{SUBSET_STRIDE}_n{len(qids)}"
    out_path = os.path.join(RAG_DIR, f"coverage_bm25_result{suffix}.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out_path, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print(f"{'手法':<12}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for m in METHODS:
            a = results[qs][m]
            if a: print(f"{m:<12}"+"".join(f"{a[k]:>10.4f}" for k in METRIC_KEYS))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
