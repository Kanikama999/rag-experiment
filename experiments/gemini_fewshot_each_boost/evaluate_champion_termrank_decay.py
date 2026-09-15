"""
championの実際の式（①：フィールド別・独立飽和・線形和のBM25）に、
「クエリ語をIDF降順に並べ、その順位iに応じてg(i)で重みを減衰させる」を組み込む。

背景: これまでのk1関連実験（k1定数・k1の文書長関数化・出現回数の減衰）は全て
BM25F（②：フィールド横断で合算してから1回だけ飽和）の枠組みで行っていた。
championが実際に使っているのは①（フィールドごとに独立計算・独立飽和・線形和）で、
②の実験結果がそのまま①に当てはまる保証はない。

指摘: 文書内の出現回数（tf）の減衰ではなく、**クエリ側の語**を対象にする。
championのクエリ（narrative×5 + Subquery + 疑似文書）は数百語あり、大半はありふれた
語。IDFで降順に並べ t=q_i (i=1..n) とし、i番目の語の寄与全体にg(i)を掛ける:

    score(D,Q) = Σ_{i=1}^{n} g(i) · idf(q_i) · [通常のBM25飽和項]

これはOpenSearchのtermクエリのboostパラメータで直接実現できる（自前のPython再計算・
_mtermvectors不要——boost=g(rank)を付けたtermクエリをbool/shouldで束ねるだけで、
OpenSearchネイティブのBM25飽和・idf計算はそのまま使われる）。したがって
champion（①）そのものにg(i)を追加した形になり、かつ従来のBM25F実験群より
大幅に高速（ネイティブクエリ、mtermvectors不要）。

IDF順位は3フィールド横断のcombined_dfで決める（bm25f系と同じ定義を流用、
ランキングの基準を統一するためだけに使う。実際のスコア計算はOpenSearchネイティブ）。

g(i) はべき乗族から開始:
    g(i) = i^(-p)      p ∈ {0.0, 0.1, 0.25, 0.5, 1.0, 2.0}
    p=0 → g(i)=1で全語が対等（championそのものに一致する検算になる）

使い方:
    python3 evaluate_champion_termrank_decay.py [SUBSET_STRIDE] [LIMIT]
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import (client, INDEX, analyze_terms, rrf_fuse,
                        _bm25f_total_docs, _bm25f_combined_df)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0   # championと同じフィールド重み
N_WORKERS = 4
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

P_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
METHODS = [f"p{p:g}" for p in P_VALUES]

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
    """3フィールド横断の語彙をIDF降順に並べ、{term: rank(1始まり)} を返す。"""
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
        idf_of[t] = (N - df) / (df + 0.5) if df > 0 else 0.0   # 順位付けだけに使う簡易IDF近似
    ranked = sorted(terms, key=lambda t: -idf_of[t])
    return {t: i + 1 for i, t in enumerate(ranked)}


def build_query(title_text, headings_text, body_text, ranks, p):
    def field_clause(text, field, weight):
        if not text or not text.strip():
            return None
        seen, clauses = set(), []
        for t in analyze_terms(text, field=field):
            if t in seen:
                continue
            seen.add(t)
            i = ranks.get(t, len(ranks) + 1)
            g = 1.0 if p == 0 else (i ** (-p))
            clauses.append({"term": {field: {"value": t, "boost": g}}})
        if not clauses:
            return None
        return {"bool": {"should": clauses, "boost": weight}}

    should = []
    for text, field, w in ((title_text, "title", TITLE_W),
                            (headings_text, "headings", HEADINGS_W),
                            (body_text, "body", BODY_W)):
        c = field_clause(text, field, w)
        if c:
            should.append(c)
    if not should:
        return None
    return {"bool": {"should": should}}


def search_one(title_text, headings_text, body_text, ranks, p, k):
    q = build_query(title_text, headings_text, body_text, ranks, p)
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
        ranks = term_ranks(t, h, b)
        for m, p in zip(METHODS, P_VALUES):
            lst = search_one(t, h, b, ranks, p, RETRIEVE_K)
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
    print(f"{len(qids)}トピック  p値={P_VALUES}")

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
    out = os.path.join(RAG_DIR, f"champion_termrank_decay_result{suffix}.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print(f"{'p':>6}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for m, p in zip(METHODS, P_VALUES):
            a = results[qs][m]
            if a: print(f"{p:>6}"+"".join(f"{a[k]:>10.4f}" for k in METRIC_KEYS))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
