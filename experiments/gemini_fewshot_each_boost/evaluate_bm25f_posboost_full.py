"""
真のBM25F(k1=3.0)に位置ブーストを組み込んだときの効果を、記録値と直接比較できる
本番条件（n=105、RETRIEVE_K=3000）で測る。

## 何を変えるのか

真のBM25Fは2段構成になっている:
  1段目(OpenSearch): bool/shouldのmatchクエリで候補をRETRIEVE_K件取ってくる
  2段目(Python)   : その候補を_mtermvectorsのtf・フィールド長からBM25F式で再スコアする

2段目は語の「出現回数」しか見ないので、位置情報を使えない。そこで**1段目のクエリに
span_first節を足す**ことで、「Subqueryの語がbody先頭100語に出る文書」を候補プールに
引き込む。championのposboostと同じく「検索時に効いて候補集合そのものを変える」ので、
事後リランク（上位100件の並べ替え）と違いrecall@1000にも効きうる。
※EXPERIMENT_SUMMARY §7.12の事後リランク実験は全条件でrecall@1000が0.2455に固定されていた。

## 条件（2x2 + baseline）

                        | 位置ブーストなし        | 位置ブーストあり
  構造化疑似文書あり    | structured_base        | structured_posboost
  narrative+Subqueryのみ| narrative_only_base    | narrative_only_posboost

  baseline: bm25_body(narrative) — 疑似文書もフィールドも位置ブーストも使わない素のBM25

structured_base は既存の k1b3_a0 と同一条件なので、既知値を再現するかが妥当性チェックになる
（EXPERIMENT_SUMMARY §7.4 の champion_ref と同じ流儀）。

## コスト最適化

4条件は候補プールが違うので本来は4回_mtermvectorsが要るが、termvectorsは
「どのクエリで引かれたか」と無関係なので、**4条件の候補IDの和集合に対して1回だけ**
取得して使い回す。各文書は取得直後に「必要な語のtfとフィールド長」だけに縮約して
保持する（OOM対策。evaluate_bm25f_k1_length_sweep.pyの教訓）。

使い方:
    python3 evaluate_bm25f_posboost_full.py
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter

import pytrec_eval

from retriever import (client, INDEX, analyze_terms, rrf_fuse, bm25_body,
                        _bm25f_avgdl, _bm25f_total_docs, _bm25f_combined_df)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
CKPT = os.path.join(RAG_DIR, "bm25f_posboost_full_ckpt.json")
OUT = os.path.join(RAG_DIR, "bm25f_posboost_full_result.json")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000          # 記録値 k1b3_a0 と同一
QUERY_REPEAT = 5

TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
B_TITLE, B_HEADINGS, B_BODY = 0.0, 0.8, 1.0
BM25_K1 = 3.0
SPAN_END, SPAN_BOOST = 100, 15.0      # championと同じ
TV_CHUNK = 1500                        # _mtermvectorsの1リクエストあたり件数

MODES = ["structured", "narrative_only"]
VARIANTS = [("base", False), ("posboost", True)]
METHODS = [f"{m}_{v}" for m in MODES for v, _ in VARIANTS] + ["baseline"]

# 既知値（bm25f_k1_length_sweep_result.json の k1b3_a0、n=105 / RETRIEVE_K=3000）
KNOWN_K1B3 = {
    "consensus": {"recall_100": 0.0739, "recall_1000": 0.3468,
                   "ndcg_cut_10": 0.5510, "P_100": 0.8446},
    "coverage": {"recall_100": 0.2771, "recall_1000": 0.6835,
                  "ndcg_cut_10": 0.4750, "P_100": 0.4423},
}

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


def field_texts(mode, repeated_q, dq, doc):
    """条件ごとの title/headings/body クエリテキストと、qtf用の論理クエリ。"""
    if mode == "structured":
        headings_txt = " ".join(doc["headings"])
        t = f"{repeated_q} {dq} {doc['title']}"
        h = f"{repeated_q} {dq} {headings_txt}"
        b = f"{repeated_q} {dq} {doc['body']}"
        logical = f"{repeated_q} {dq} {doc['title']} {headings_txt} {doc['body']}"
    else:
        plain = f"{repeated_q} {dq}"
        t = h = b = plain
        logical = plain
    return t, h, b, logical


def candidate_search(t, h, b, span_terms, use_posboost, k):
    """1段目: 候補プールを取る。use_posboost=Trueならspan_first節を足す
    （retriever.bm25_fielded_posboost と同一の構築）。"""
    should = []
    if t and t.strip():
        should.append({"match": {"title": {"query": t, "boost": TITLE_W}}})
    if h and h.strip():
        should.append({"match": {"headings": {"query": h, "boost": HEADINGS_W}}})
    if b and b.strip():
        should.append({"match": {"body": {"query": b, "boost": BODY_W}}})
    if use_posboost and span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": x}} for x in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": SPAN_END, "boost": SPAN_BOOST}})
    if not should:
        return []
    res = client.search(index=INDEX, body={
        "size": k, "_source": False, "query": {"bool": {"should": should}}})
    return [hit["_id"] for hit in res["hits"]["hits"]]


def fetch_reduced_termvectors(docids, needed_terms):
    """和集合の候補IDについて_mtermvectorsを1回（チャンク分割）だけ叩き、
    各文書を {field: {語: tf}}（needed_termsのみ） と field_len に縮約して返す。"""
    needed = set(needed_terms)
    out = {}
    ids = list(docids)
    for i in range(0, len(ids), TV_CHUNK):
        chunk = ids[i:i + TV_CHUNK]
        res = client.mtermvectors(index=INDEX, body={
            "ids": chunk,
            "parameters": {
                "fields": ["title", "headings", "body"],
                "term_statistics": False, "field_statistics": False,
                "positions": False, "offsets": False, "payloads": False,
            }})
        for doc in res.get("docs", []):
            docid = doc.get("_id")
            tvs = doc.get("term_vectors", {})
            ft, fl = {}, {}
            for field in ("title", "headings", "body"):
                terms_info = tvs.get(field, {}).get("terms", {})
                # フィールド長は全語の合計（縮約前に計算すること）
                fl[field] = sum(info["term_freq"] for info in terms_info.values())
                ft[field] = {t: info["term_freq"]
                             for t, info in terms_info.items() if t in needed}
            out[docid] = (ft, fl)
    return out


def bm25f_score(cand_ids, tvs, query_terms, idf, avgdl, qtf, k):
    """2段目: 縮約済みtermvectorsからBM25F式でスコアリング。
    retriever.bm25f_score と同一の式（検証済み: スコア差0.0）に qtf 線形を掛けたもの。"""
    weights = {"title": TITLE_W, "headings": HEADINGS_W, "body": BODY_W}
    bs = {"title": B_TITLE, "headings": B_HEADINGS, "body": B_BODY}
    scores = {}
    for docid in cand_ids:
        got = tvs.get(docid)
        if not got:
            continue
        ft, fl = got
        score = 0.0
        for t in query_terms:
            t_idf = idf.get(t, 0.0)
            if t_idf <= 0:
                continue
            pseudo_tf = 0.0
            for field in ("title", "headings", "body"):
                tf = ft[field].get(t)
                if not tf:
                    continue
                dl = fl[field]
                B = (1 - bs[field]) + bs[field] * (dl / avgdl[field]) if avgdl[field] else 1.0
                pseudo_tf += weights[field] * tf / B
            if pseudo_tf > 0:
                score += t_idf * pseudo_tf / (BM25_K1 + pseudo_tf) * float(qtf.get(t, 1))
        scores[docid] = score
    return sorted(scores.items(), key=lambda x: -x[1])[:k]


def qtf_counts(logical_query):
    res = client.indices.analyze(index=INDEX, body={"field": "body", "text": logical_query})
    return Counter(t["token"] for t in res.get("tokens", []))


def run_config():
    return {"retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT, "topk": TOPK,
            "weights": [TITLE_W, HEADINGS_W, BODY_W], "b": [B_TITLE, B_HEADINGS, B_BODY],
            "k1": BM25_K1, "span_end": SPAN_END, "span_boost": SPAN_BOOST, "n": "all"}


def process_topic(qid, queries, webstyle, avgdl, N):
    """1トピック分。全Subquery・全条件の候補検索を先に済ませ、トピック全体の候補IDと語の
    和集合に対して_mtermvectorsを1回だけ取得して使い回す（同一トピックのSubqueryは
    narrative×5を共有し候補が大きく重複する。qid=1001実測で延べ36,000件→和集合7,167件）。"""
    q = queries[qid]
    repeated_q = " ".join([q] * QUERY_REPEAT)

    subq_conds = []
    topic_terms, topic_ids = [], []
    seen_terms, seen_ids = set(), set()
    for dq, doc in dq_pairs_structured(webstyle[qid]):
        span_terms = analyze_terms(dq, field="body")
        per_cond = {}
        for mode in MODES:
            t, h, b, logical = field_texts(mode, repeated_q, dq, doc)
            qterms = []
            for field, text in (("title", t), ("headings", h), ("body", b)):
                for term in analyze_terms(text, field=field):
                    if term not in qterms:
                        qterms.append(term)
            qtf = qtf_counts(logical)
            for term in qterms:
                if term not in seen_terms:
                    seen_terms.add(term)
                    topic_terms.append(term)
            for vname, use_pb in VARIANTS:
                cand = candidate_search(t, h, b, span_terms, use_pb, RETRIEVE_K)
                per_cond[f"{mode}_{vname}"] = (cand, qterms, qtf)
                for d in cand:
                    if d not in seen_ids:
                        seen_ids.add(d)
                        topic_ids.append(d)
        subq_conds.append(per_cond)

    lists = {m: [] for m in METHODS if m != "baseline"}
    if topic_ids:
        idf = {t: (lambda df: math.log(1 + (N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0)
               (_bm25f_combined_df(t)) for t in topic_terms}
        tvs = fetch_reduced_termvectors(topic_ids, topic_terms)
        for per_cond in subq_conds:
            for method, (cand, qterms, qtf) in per_cond.items():
                ranked = bm25f_score(cand, tvs, qterms, idf, avgdl, qtf, RETRIEVE_K)
                if ranked:
                    lists[method].append(ranked)

    out = {m: ({d: float(s) for d, s in rrf_fuse(ls, top_n=TOPK)} if ls else {})
           for m, ls in lists.items()}
    # baseline: 疑似文書もフィールドも位置ブーストも使わない素のBM25（narrativeそのまま）
    out["baseline"] = {d: float(s) for d, s in bm25_body(q, k=TOPK)}
    return out, len(topic_ids)


def build_runs(qids, queries, webstyle):
    runs = {m: {} for m in METHODS}
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            cached = json.load(f)
        if cached.get("config") == run_config():
            runs = {m: cached["runs"].get(m, {}) for m in METHODS}
            print(f"チェックポイント復帰: {len(runs['structured_base'])} トピック済み", flush=True)
        else:
            print("チェックポイントは条件が違うため破棄", flush=True)

    avgdl = {f: _bm25f_avgdl(f) for f in ("title", "headings", "body")}
    N = _bm25f_total_docs()

    todo = [q for q in qids if q not in runs["structured_base"]]
    t0 = time.time()
    for n, qid in enumerate(todo, 1):
        out, n_fetched = process_topic(qid, queries, webstyle, avgdl, N)
        for m in METHODS:
            runs[m][qid] = out[m]

        el = time.time() - t0
        eta = el / n * (len(todo) - n)
        print(f"  {n}/{len(todo)}  qid={qid}  fetch={n_fetched}件  "
              f"経過{el/60:.1f}分  残り約{eta/60:.0f}分", flush=True)
        with open(CKPT, "w") as f:
            json.dump({"config": run_config(), "runs": runs}, f)
    return runs


def evaluate(run, qrels, qids):
    target = [q for q in qids if q in qrels and q in run]
    if not target:
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = ev.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    agg["n"] = n
    return agg


def main():
    print("読み込み中...", flush=True)
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}

    qids = sorted(qid for qid in webstyle
                  if qid in queries and webstyle[qid].get("decomposed_queries"))
    print(f"対象トピック: {len(qids)}   RETRIEVE_K={RETRIEVE_K}   条件: {', '.join(METHODS)}",
          flush=True)

    runs = build_runs(qids, queries, webstyle)

    summary = {}
    for name in QREL_SETS:
        for m in METHODS:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], qids)

    with open(OUT, "w") as f:
        json.dump({"n_topics": len(qids), "config": run_config(), "summary": summary},
                   f, indent=2)
    print(f"\n-> {OUT}")

    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print("method".ljust(26) + "".join(labels[k].ljust(14) for k in METRIC_KEYS) + "n")
        for m in METHODS:
            a = summary[qs].get(m)
            if not a:
                print(m.ljust(26) + "-")
                continue
            print(m.ljust(26) + "".join(f"{a[k]:.4f}".ljust(14) for k in METRIC_KEYS) + str(a["n"]))

        # 妥当性チェック: structured_base は既知の k1b3_a0 を再現するはず
        ref, got = KNOWN_K1B3[qs], summary[qs].get("structured_base")
        if got:
            print(f"\n  [妥当性チェック] structured_base − 既知のk1b3_a0（0近傍なら再現）")
            for k in METRIC_KEYS:
                print(f"    {labels[k]:14s} {got[k] - ref[k]:+.4f}")
        # 位置ブーストの効果
        a, b = summary[qs].get("structured_base"), summary[qs].get("structured_posboost")
        if a and b:
            print(f"  [位置ブーストの効果] structured_posboost − structured_base")
            for k in METRIC_KEYS:
                print(f"    {labels[k]:14s} {b[k] - a[k]:+.4f}")


if __name__ == "__main__":
    main()
