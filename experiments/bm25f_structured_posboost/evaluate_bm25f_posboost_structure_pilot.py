"""
2つの問いを同時に検証するパイロット実験（n=20トピック、RETRIEVE_K=1000）。

1. 構造化擬似文書の寄与: 真のBM25F(k1=3.0, b=[0,0.8,1.0], qtf=linear)で、
   title/headings/bodyにLLM生成の構造化疑似文書を入れる（現行best）のと、
   narrative×5+Subqueryだけを3フィールドとも同じテキストで入れる（構造化なし）のを比較する。

2. 位置ブーストの追加: bm25f_prepare()の候補プール取得クエリ（bool/should）に
   span_first節（span_end=100, span_boost=15、championと同じ値）を追加する。
   BM25Fスコア自体（Python側のbm25f_score）は変更せず、候補プールの選抜だけに
   位置ブーストを使う。championのposboostは「検索時に効いて候補集合そのものを
   変える」ことがEXPERIMENT_SUMMARY.md §7.12で確認済みなので、同じ構造をBM25Fの
   候補プール取得側に持ち込めばrecall@1000にも効きうる、という仮説を検証する。

2x2 (structured/narrative_only) x (posboost有無) の4条件。

n=20はパイロット（時間短縮のためRETRIEVE_K=1000。本番のk1b3_a0はRETRIEVE_K=3000
なので数値はそちらと直接比較できない。あくまで4条件間の相対比較用）。

使い方:
    python3 evaluate_bm25f_posboost_structure_pilot.py [N_TOPICS]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import pytrec_eval

from retriever import (client, INDEX, analyze_terms, rrf_fuse,
                        _bm25f_avgdl, _bm25f_total_docs, _bm25f_combined_df)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
CKPT = os.path.join(RAG_DIR, "bm25f_posboost_structure_pilot_ckpt.json")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
N_TOPICS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0
B_TITLE, B_HEADINGS, B_BODY = 0.0, 0.8, 1.0
BM25_K1 = 3.0                 # k1_length_sweepの最良値
SPAN_END, SPAN_BOOST = 100, 15.0   # championと同じ

METHODS = ["structured_base", "structured_posboost",
           "narrowonly_base", "narrowonly_posboost"]

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


def bm25f_prepare_custom(title_text, headings_text, body_text, span_terms,
                          candidate_k, use_posboost):
    """retriever.bm25f_prepare()の候補プール取得クエリに、use_posboost=Trueなら
    span_first節（body、span_terms、SPAN_END/SPAN_BOOST）を追加した版。
    後段のBM25Fスコア計算（Python側）はそのまま bm25f_prepare と同一ロジック。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": TITLE_W}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": HEADINGS_W}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": BODY_W}}})
    if use_posboost and span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": SPAN_END, "boost": SPAN_BOOST}})
    if not should:
        return None

    res = client.search(index=INDEX, body={
        "size": candidate_k, "_source": False,
        "query": {"bool": {"should": should}}
    })
    candidates = [h["_id"] for h in res["hits"]["hits"]]
    if not candidates:
        return None

    query_terms = []
    seen = set()
    for field, text in (("title", title_text), ("headings", headings_text), ("body", body_text)):
        if not text or not text.strip():
            continue
        for t in analyze_terms(text, field=field):
            if t not in seen:
                seen.add(t)
                query_terms.append(t)

    avgdl = {f: _bm25f_avgdl(f) for f in ("title", "headings", "body")}
    N = _bm25f_total_docs()
    idf = {t: (lambda df: math.log(1 + (N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0)
           (_bm25f_combined_df(t))
           for t in query_terms}

    docs = {}
    if query_terms:
        tv_docs = client.mtermvectors(index=INDEX, body={
            "ids": candidates,
            "parameters": {
                "fields": ["title", "headings", "body"],
                "term_statistics": False, "field_statistics": False,
                "positions": False, "offsets": False, "payloads": False,
            }
        })
        for doc in tv_docs.get("docs", []):
            docid = doc.get("_id")
            tvs = doc.get("term_vectors", {})
            field_terms, field_len = {}, {}
            for field in ("title", "headings", "body"):
                terms_info = tvs.get(field, {}).get("terms", {})
                field_terms[field] = terms_info
                field_len[field] = sum(info["term_freq"] for info in terms_info.values())
            docs[docid] = {"field_terms": field_terms, "field_len": field_len}

    return {"candidates": candidates, "query_terms": query_terms, "idf": idf,
            "avgdl": avgdl, "docs": docs}


def qtf_counts(logical_query):
    res = client.indices.analyze(index=INDEX, body={"field": "body", "text": logical_query})
    from collections import Counter
    return Counter(t["token"] for t in res.get("tokens", []))


def bm25f_score_k1(raw, qtf, k):
    if raw is None:
        return []
    if not raw["query_terms"]:
        return [(d, 0.0) for d in raw["candidates"][:k]]
    idf, avgdl, docs = raw["idf"], raw["avgdl"], raw["docs"]
    weights = {"title": TITLE_W, "headings": HEADINGS_W, "body": BODY_W}
    bs = {"title": B_TITLE, "headings": B_HEADINGS, "body": B_BODY}
    scores = {}
    for docid, doc in docs.items():
        field_terms, field_len = doc["field_terms"], doc["field_len"]
        score = 0.0
        for t in raw["query_terms"]:
            t_idf = idf.get(t, 0.0)
            if t_idf <= 0:
                continue
            pseudo_tf = 0.0
            for field in ("title", "headings", "body"):
                info = field_terms[field].get(t)
                if not info:
                    continue
                dl = field_len[field]
                B = (1 - bs[field]) + bs[field] * (dl / avgdl[field]) if avgdl[field] else 1.0
                pseudo_tf += weights[field] * info["term_freq"] / B
            if pseudo_tf > 0:
                n = float(qtf.get(t, 1))
                score += t_idf * pseudo_tf / (BM25_K1 + pseudo_tf) * n
        scores[docid] = score
    return sorted(scores.items(), key=lambda x: -x[1])[:k]


def build_texts(mode, repeated_q, dq, doc):
    """mode: 'structured' (LLM疑似文書) / 'narrowonly' (narrative+Subqueryのみ、3フィールド共通)"""
    if mode == "structured":
        headings_txt = " ".join(doc["headings"])
        return (f"{repeated_q} {dq} {doc['title']}",
                f"{repeated_q} {dq} {headings_txt}",
                f"{repeated_q} {dq} {doc['body']}")
    else:
        plain = f"{repeated_q} {dq}"
        return (plain, plain, plain)


def run_config():
    return {"retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT, "topk": TOPK,
            "weights": [TITLE_W, HEADINGS_W, BODY_W], "b": [B_TITLE, B_HEADINGS, B_BODY],
            "k1": BM25_K1, "span_end": SPAN_END, "span_boost": SPAN_BOOST,
            "n_topics": N_TOPICS}


def build_runs(qids, queries, webstyle):
    runs = {m: {} for m in METHODS}
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            cached = json.load(f)
        if cached.get("config") == run_config():
            runs = {m: cached["runs"].get(m, {}) for m in METHODS}
            print(f"チェックポイント復帰: {len(runs[METHODS[0]])} トピック済み", flush=True)
        else:
            print("チェックポイントは条件が違うため破棄", flush=True)

    todo = [q for q in qids if q not in runs[METHODS[0]]]
    t0 = time.time()
    for n, qid in enumerate(todo, 1):
        q = queries[qid]
        repeated_q = " ".join([q] * QUERY_REPEAT)
        lists = {m: [] for m in METHODS}
        for dq, doc in dq_pairs_structured(webstyle[qid]):
            span_terms = analyze_terms(dq, field="body")
            logical_q_structured = f"{repeated_q} {dq} {doc['title']} {' '.join(doc['headings'])} {doc['body']}"
            logical_q_narrowonly = f"{repeated_q} {dq}"

            for mode, base_qtf in (("structured", logical_q_structured),
                                    ("narrowonly", logical_q_narrowonly)):
                title_q, headings_q, body_q = build_texts(mode, repeated_q, dq, doc)
                qtf = qtf_counts(base_qtf)
                for posboost, suffix in ((False, "base"), (True, "posboost")):
                    method = f"{mode}_{suffix}"
                    raw = bm25f_prepare_custom(title_q, headings_q, body_q, span_terms,
                                                candidate_k=RETRIEVE_K, use_posboost=posboost)
                    ranked = bm25f_score_k1(raw, qtf, RETRIEVE_K)
                    if ranked:
                        lists[method].append(ranked)

        for m in METHODS:
            fused = rrf_fuse(lists[m], top_n=TOPK) if lists[m] else []
            runs[m][qid] = {d: float(s) for d, s in fused}

        el = time.time() - t0
        eta = el / n * (len(todo) - n)
        print(f"  {n}/{len(todo)}  qid={qid}  経過{el/60:.1f}分  残り約{eta/60:.0f}分", flush=True)
        with open(CKPT, "w") as f:
            json.dump({"config": run_config(), "runs": runs}, f)
    return runs


def evaluate(run, qrels, qids, label):
    target = [q for q in qids if q in qrels and q in run]
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = ev.evaluate({q: run[q] for q in target})
    n = len(results)
    if n == 0:
        print(f"[{label}] このqrelsに含まれるトピックが0件（スキップ）")
        return None
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    agg["n"] = n
    return agg


def main():
    print("読み込み中...", flush=True)
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}

    valid_qids = sorted(qid for qid in webstyle
                        if qid in queries and webstyle[qid].get("decomposed_queries"))
    search_qids = valid_qids[:N_TOPICS]
    print(f"全トピック: {len(valid_qids)}   対象: {len(search_qids)}   "
          f"candidate_k={RETRIEVE_K}   条件数: {len(METHODS)}", flush=True)

    runs = build_runs(search_qids, queries, webstyle)

    summary = {}
    for name in QREL_SETS:
        print(f"\n[{name}]")
        for m in METHODS:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], search_qids, f"{m}/{name}")

    out = os.path.join(RAG_DIR, "bm25f_posboost_structure_pilot_result.json")
    with open(out, "w") as f:
        json.dump({"search_qids_n": len(search_qids), "config": run_config(), "summary": summary},
                   f, indent=2)
    print(f"\n-> {out}")

    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        header = "method".ljust(24) + "".join(labels[k].ljust(14) for k in METRIC_KEYS) + "n"
        print(header)
        for m in METHODS:
            a = summary[qs].get(m)
            if not a:
                print(m.ljust(24) + "-")
                continue
            print(m.ljust(24) + "".join(f"{a[k]:.4f}".ljust(14) for k in METRIC_KEYS) + str(a["n"]))


if __name__ == "__main__":
    main()
