"""
championそのもの（title/headings/body均等 + 位置ブースト）に、クエリ語のIDF順位減衰
g(i)を追加したら実際に championを超えられるかを検証する。

背景: evaluate_champion_termrank_decay_multi.py は位置ブーストなしのbm25_fielded
ベースで検証し、g=1一律のベースライン比では大きな改善（R@1000 +20〜35%）を示したが、
**championそのもの（位置ブーストあり）と比べると全指標で大敗**していた
（位置ブーストが総改善の約75%を稼ぐ主役であるため、それが無い状態での比較には
意味が無かった）。本スクリプトはその欠落を埋め、championの構造
（bm25_equalweight_posboost_discourseboost、markers=[]、span_end=100、span_boost=15）
をそのまま使い、title/headings/bodyのmatch節だけをg(i)付きのterm節に置き換える。

g(i)=1一律の条件は championと完全に一致するはずで、これが検算になる。

前回の19種スイープで上位だった候補に絞って検証:
    baseline（g=1、championそのもの）
    power_p0.25, power_p0.5
    geo_r0.98, geo_r0.99
    cutoff_K60
    sigmoid_K60_s20

使い方:
    python3 evaluate_champion_posboost_termrank_decay.py [SUBSET_STRIDE] [LIMIT]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from collections import Counter

from retriever import (client, INDEX, analyze_terms, rrf_fuse,
                        _bm25f_total_docs, _bm25f_combined_df)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
SPAN_END, SPAN_BOOST = 100, 15.0   # championと同じ
N_WORKERS = 4
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

GFUNCS = [
    ("baseline_g1", lambda i: 1.0),
    ("power_p0.25", lambda i: i ** (-0.25)),
    ("power_p0.5", lambda i: i ** (-0.5)),
    ("geo_r0.98", lambda i: 0.98 ** (i - 1)),
    ("geo_r0.99", lambda i: 0.99 ** (i - 1)),
    ("cutoff_K60", lambda i: 1.0 if i <= 60 else 0.0),
    ("sigmoid_K60_s20", lambda i: 1.0 / (1.0 + math.exp((i - 60) / 20))),
]
METHODS = [name for name, _ in GFUNCS]
GFUNC_MAP = dict(GFUNCS)

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                queries[obj["id"]] = obj["title"]
    return queries


def dq_pairs_structured(entry):
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"],
                                         entry["query2doc_docs_structured"])
            if doc and doc.get("body")]


def term_ranks(title_text, headings_text, body_text):
    seen, terms = set(), []
    for field, text in (("title", title_text), ("headings", headings_text), ("body", body_text)):
        if not text or not text.strip():
            continue
        for t in analyze_terms(text, field=field):
            if t not in seen:
                seen.add(t)
                terms.append(t)
    N = _bm25f_total_docs()
    idf_of = {}
    for t in terms:
        df = _bm25f_combined_df(t)
        idf_of[t] = (N - df) / (df + 0.5) if df > 0 else 0.0
    ranked = sorted(terms, key=lambda t: -idf_of[t])
    return {t: i + 1 for i, t in enumerate(ranked)}


def qtf_counts(text, field="body"):
    """textを重複除去せずトークン化し、語ごとの出現回数を数える（analyze_terms()は
    重複除去してしまうため、これとは別に生トークン列から数える必要がある）。
    2026-09-13発見のバグ修正: 従来のterm_ranksベースのg(i)実装はanalyze_terms()で
    重複除去した語に対してterm節を1つずつしか作っておらず、narrative×5などの
    繰り返し（qtf）が完全に無視されていた。championのmatchクエリはクエリ文字列内で
    同じ語が5回出れば5節ぶんスコアが乗るため、この欠落によりbaseline_g1（g=1一律）
    がchampion本来の値（consensus nDCG@10=0.5251）を大きく下回っていた
    （実測0.4279、-0.0972）。"""
    res = client.indices.analyze(index=INDEX, body={"field": field, "text": text})
    return Counter(t["token"] for t in res.get("tokens", []))


def champion_posboost_query(title_text, headings_text, body_text, span_terms, ranks, gfunc):
    """championのbm25_equalweight_posboost_discourseboost（markers=[]）と同じ構造で、
    title/headings/bodyのmatch節だけをg(i)*qtf(t)付きterm節に置き換える。span_first節は
    championと完全に同一（span_terms・span_end・span_boostも同じ）。
    g(i)*qtf(t) とすることで、championのmatchクエリが自然に持っていた「クエリ文字列内で
    語が複数回出れば比例して寄与が増える」性質を維持しつつ、g(i)によるIDF順位減衰を
    掛け合わせる（qtfはfield_clause呼び出し側でtermごとに渡す）。"""
    def field_clause(text, field):
        if not text or not text.strip():
            return None
        qtf = qtf_counts(text, field=field)
        seen, clauses = set(), []
        for t in analyze_terms(text, field=field):
            if t in seen:
                continue
            seen.add(t)
            i = ranks.get(t, len(ranks) + 1)
            g = gfunc(i) * qtf.get(t, 1)
            if g <= 0:
                continue
            clauses.append({"term": {field: {"value": t, "boost": g}}})
        if not clauses:
            return None
        return {"bool": {"should": clauses, "boost": 1}}   # championと同じboost=1

    should = []
    for text, field in ((title_text, "title"), (headings_text, "headings"), (body_text, "body")):
        c = field_clause(text, field)
        if c:
            should.append(c)
    if span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": SPAN_END, "boost": SPAN_BOOST}})
    if not should:
        return None
    return {"bool": {"should": should}}


def search_one(title_text, headings_text, body_text, span_terms, ranks, gfunc, k):
    q = champion_posboost_query(title_text, headings_text, body_text, span_terms, ranks, gfunc)
    if q is None:
        return []
    res = client.search(index=INDEX, body={"size": k, "_source": False, "query": q})
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def run_topic(qid, narrative, entry):
    rep = " ".join([narrative] * QUERY_REPEAT)
    lists = {m: [] for m in METHODS}
    for dq, doc in dq_pairs_structured(entry):
        hd = " ".join(doc["headings"])
        t = f"{rep} {dq} {doc['title']}"
        h = f"{rep} {dq} {hd}"
        b = f"{rep} {dq} {doc['body']}"
        span_terms = analyze_terms(dq)   # championと同じ：span_termsはSubqueryから
        ranks = term_ranks(t, h, b)
        for m, gfunc in GFUNCS:
            lst = search_one(t, h, b, span_terms, ranks, gfunc, RETRIEVE_K)
            if lst:
                lists[m].append(lst)
    return qid, {m: {d: float(s) for d, s in (rrf_fuse(lists[m], top_n=TOPK) if lists[m] else [])}
                 for m in METHODS}


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
    if LIMIT:
        qids = qids[:LIMIT]
    print(f"{len(qids)}トピック  g(i)候補={len(METHODS)}種: {METHODS}")

    t0 = time.time()
    runs = {m: {} for m in METHODS}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_topic, q, queries[q], webstyle[q]) for q in qids]
        for n, f in enumerate(futs, 1):
            qid, out = f.result()
            for m in METHODS:
                runs[m][qid] = out[m]
            if n % 10 == 0 or n == len(qids):
                el = time.time() - t0
                print(f"  {n}/{len(qids)}  {el/60:.1f}分  残り約{el/n*(len(qids)-n)/60:.0f}分", flush=True)

    results = {}
    for qs in QREL_SETS:
        for m in METHODS:
            results.setdefault(qs, {})[m] = evaluate(runs[m], qrels[qs], qids)

    suffix = "" if SUBSET_STRIDE == 1 and not LIMIT else f"_stride{SUBSET_STRIDE}_n{len(qids)}"
    out = os.path.join(RAG_DIR, f"champion_posboost_termrank_decay_result{suffix}.json")
    json.dump({"n_topics": len(qids), "methods": METHODS, "results": results}, open(out, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n=== {qs}  nDCG@10 降順 ===")
        rows = [(m, results[qs][m]) for m in METHODS if results[qs][m]]
        print(f"{'g(i)':<18}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for m, a in sorted(rows, key=lambda x: -x[1]["ndcg_cut_10"]):
            print(f"{m:<18}"+"".join(f"{a[k]:>10.4f}" for k in METRIC_KEYS))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
