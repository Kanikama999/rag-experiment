"""
webstyle_narrative_equalweight_posboost_only（bm25_equalweight_posboost_discourseboost、
markers=[]でdiscourseboostを無効化したposboost_only構成。現状のchampion）の
span_end / span_boostを、search_posboost_params.pyと同じ座標降下法（35トピック
サブセット、1軸ずつ2ラウンド）で再探索する。

既存のsearch_posboost_params.py（posboost_param_search_result_round2.json、
span_end=100, span_boost=15.0を採用）はtitle=2,headings=1,body=1のfielded構成
（bm25_fielded_posboost）で探索したものであり、championのequalweight構成（title=1均等）
では一度も再探索されていない。フィールド重みが変わるとBM25生スコアのスケールも
変わるため、span_boostの固定値の相対的な効きも変わりうる。本スクリプトはその
ギャップを埋めるための再探索。

使い方:
    python search_equalweight_posboost_params.py
"""

from __future__ import annotations

import json
import os
import time

import pytrec_eval

from retriever import bm25_equalweight_posboost_discourseboost, rrf_fuse, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5

SPAN_END_GRID = [10, 30, 50, 100, 200]
SPAN_BOOST_GRID = [3.0, 5.0, 8.0, 15.0, 20.0, 30.0]
DEFAULT_SPAN_BOOST = 15.0  # championの現行値を起点にする

METRIC_SPECS = ["recall.1000", "ndcg_cut.10"]
METRICS = set(METRIC_SPECS)


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
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


_analyze_cache = {}


def cached_span_terms(dq):
    if dq not in _analyze_cache:
        _analyze_cache[dq] = analyze_terms(dq, field="body")
    return _analyze_cache[dq]


class ScoreCache:
    def __init__(self):
        self._cache = {}
        self.calls = self.hits = 0

    def search(self, title_q, headings_q, body_q, span_terms, span_end, span_boost):
        key = (title_q, headings_q, body_q, tuple(span_terms), span_end, span_boost)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        r = bm25_equalweight_posboost_discourseboost(
            title_q, headings_q, body_q, list(span_terms), k=RETRIEVE_K,
            span_end=span_end, span_boost=span_boost, markers=[])
        self._cache[key] = r
        return r


def evaluate_params(span_end, span_boost, qids, queries, webstyle, qrels, cache):
    run = {}
    for qid in qids:
        q = queries[qid]
        entry = webstyle[qid]
        pairs = dq_pairs_structured(entry)
        repeated_q = " ".join([q] * QUERY_REPEAT)
        lists = []
        for dq, doc in pairs:
            title_q = f"{repeated_q} {dq} {doc['title']}"
            headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
            body_q = f"{repeated_q} {dq} {doc['body']}"
            span_terms = cached_span_terms(dq)
            lists.append(cache.search(title_q, headings_q, body_q, span_terms, span_end, span_boost))
        lists = [lst for lst in lists if lst]
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        run[qid] = {docid: float(score) for docid, score in fused}

    target = [q for q in qids if q in qrels]
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    recall_1000 = sum(r["recall_1000"] for r in results.values()) / n
    ndcg_10 = sum(r["ndcg_cut_10"] for r in results.values()) / n
    return recall_1000, ndcg_10


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and qid in qrels and webstyle[qid].get("decomposed_queries")
    )
    search_qids = valid_qids[::3]
    print(f"全トピック: {len(valid_qids)}   探索用サブセット: {len(search_qids)}")

    cache = ScoreCache()
    log = []

    def try_params(span_end, span_boost, tag=""):
        t0 = time.time()
        recall_1000, ndcg_10 = evaluate_params(span_end, span_boost, search_qids, queries, webstyle, qrels, cache)
        dt = time.time() - t0
        print(f"  {tag}span_end={span_end:4d} span_boost={span_boost:.1f}  "
              f"recall@1000={recall_1000:.4f} nDCG@10={ndcg_10:.4f}  ({dt:.0f}s)")
        log.append({"span_end": span_end, "span_boost": span_boost,
                     "recall_1000": recall_1000, "ndcg_10": ndcg_10})
        return recall_1000, ndcg_10

    print(f"\n=== ラウンド1: span_boost={DEFAULT_SPAN_BOOST}固定でspan_endを走査 ===")
    round1 = []
    for span_end in SPAN_END_GRID:
        r, n = try_params(span_end, DEFAULT_SPAN_BOOST)
        round1.append((span_end, r, n))
    best_end_recall = max(round1, key=lambda t: t[1])[0]
    best_end_ndcg = max(round1, key=lambda t: t[2])[0]
    print(f"  -> recall@1000最良: span_end={best_end_recall}   nDCG@10最良: span_end={best_end_ndcg}")

    print(f"\n=== ラウンド2: recall@1000最良のspan_end={best_end_recall}に固定してspan_boostを走査 ===")
    round1_at_best_end = next((r, n) for e, r, n in round1 if e == best_end_recall)
    round2 = []
    for span_boost in SPAN_BOOST_GRID:
        if span_boost == DEFAULT_SPAN_BOOST:
            r, n = round1_at_best_end
            print(f"  (ラウンド1の結果を再利用) span_end={best_end_recall:4d} span_boost={span_boost:.1f}  "
                  f"recall@1000={r:.4f} nDCG@10={n:.4f}")
        else:
            r, n = try_params(best_end_recall, span_boost)
        round2.append((span_boost, r, n))
    best_boost_recall = max(round2, key=lambda t: t[1])[0]
    best_boost_ndcg = max(round2, key=lambda t: t[2])[0]
    print(f"  -> recall@1000最良: span_boost={best_boost_recall}   nDCG@10最良: span_boost={best_boost_ndcg}")

    print("\n" + "=" * 72)
    print(f"最良の組（recall@1000基準、{len(search_qids)}トピックのサブセット）: "
          f"span_end={best_end_recall}, span_boost={best_boost_recall}")
    best_r, best_n = evaluate_params(best_end_recall, best_boost_recall, search_qids, queries, webstyle, qrels, cache)
    print(f"  recall@1000={best_r:.4f}  nDCG@10={best_n:.4f}")
    print(f"（参考: 現行championのデフォルト値 span_end=100, span_boost=15.0）")

    print(f"\nキャッシュ: 呼び出し{cache.calls}回 / ヒット{cache.hits}回")

    out_path = os.path.join(RAG_DIR, "equalweight_posboost_param_search_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "search_qids_n": len(search_qids),
            "round1_span_end_sweep": [{"span_end": e, "recall_1000": r, "ndcg_10": n} for e, r, n in round1],
            "round2_span_boost_sweep": [{"span_boost": b, "recall_1000": r, "ndcg_10": n} for b, r, n in round2],
            "best_span_end": best_end_recall,
            "best_span_boost": best_boost_recall,
            "best_recall_1000": best_r,
            "best_ndcg_10": best_n,
        }, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
