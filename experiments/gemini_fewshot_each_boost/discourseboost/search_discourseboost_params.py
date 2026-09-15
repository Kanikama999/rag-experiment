"""
webstyle_narrative_fielded_discourseboost（bm25_fielded_discourseboost、Subquery本文の語が
要約系discourse markerの直後slop語以内に出現したらmarker_boost倍で加点するspan_near）の
slop / marker_boostを、search_posboost_params.pyと同じ座標降下法で探索する。

posboostのspan_boostは初期値2.0よりずっと高い15.0が最良だった。discourseboostの
marker_boostも初期値3.0のままで、同様の伸びしろが眠っている可能性がある。まず
marker_boost=3.0固定でslopを走査し、最良のslopに固定してmarker_boostを走査する。

探索フェーズは全105トピックの1/3（約35トピック、valid_qids[::3]）だけを使う。
title/headings/bodyのブーストはrecallopt（title=2,headings=1,body=1）に固定する。

使い方:
    python search_discourseboost_params.py
"""

# 2026-09-13: discourseboost関連ファイルをdiscourseboost/サブディレクトリへ移動。
# RAG_DIRが1階層深くなった分、WEBSTYLE_FILE/DATA_DIRの相対パスを調整済み。

from __future__ import annotations

import json
import os
import sys

# 2026-09-13: discourseboost/ サブディレクトリからでも retriever.py（親ディレクトリ）を
# importできるよう、親ディレクトリをsys.pathに追加する。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import pytrec_eval

from retriever import bm25_fielded_discourseboost, rrf_fuse, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "..", "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5

TITLE_BOOST, HEADINGS_BOOST, BODY_BOOST = 2, 1, 1  # recalloptに固定

SLOP_GRID = [10, 20, 40]
MARKER_BOOST_GRID = [1.0, 3.0, 8.0, 15.0]
DEFAULT_SLOP = 20
DEFAULT_MARKER_BOOST = 3.0

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
    """(title_q, headings_q, body_q, span_terms, slop, marker_boost) -> 検索結果 のキャッシュ。"""

    def __init__(self):
        self._cache = {}
        self.calls = self.hits = 0

    def search(self, title_q, headings_q, body_q, span_terms, slop, marker_boost):
        key = (title_q, headings_q, body_q, tuple(span_terms), slop, marker_boost)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        r = bm25_fielded_discourseboost(title_q, headings_q, body_q, list(span_terms), k=RETRIEVE_K,
                                         title_boost=TITLE_BOOST, headings_boost=HEADINGS_BOOST,
                                         body_boost=BODY_BOOST, slop=slop, marker_boost=marker_boost)
        self._cache[key] = r
        return r


def evaluate_params(slop, marker_boost, qids, queries, webstyle, qrels, cache):
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
            lists.append(cache.search(title_q, headings_q, body_q, span_terms, slop, marker_boost))
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
    search_qids = valid_qids[::3]  # 探索フェーズ用の部分集合（約1/3）
    print(f"全トピック: {len(valid_qids)}   探索用サブセット: {len(search_qids)}")

    cache = ScoreCache()
    log = []

    def try_params(slop, marker_boost, tag=""):
        t0 = time.time()
        recall_1000, ndcg_10 = evaluate_params(slop, marker_boost, search_qids, queries, webstyle, qrels, cache)
        dt = time.time() - t0
        print(f"  {tag}slop={slop:4d} marker_boost={marker_boost:.1f}  "
              f"recall@1000={recall_1000:.4f} nDCG@10={ndcg_10:.4f}  ({dt:.0f}s)")
        log.append({"slop": slop, "marker_boost": marker_boost,
                     "recall_1000": recall_1000, "ndcg_10": ndcg_10})
        return recall_1000, ndcg_10

    print("\n=== ラウンド1: marker_boost=3.0固定でslopを走査 ===")
    round1 = []
    for slop in SLOP_GRID:
        r, n = try_params(slop, DEFAULT_MARKER_BOOST)
        round1.append((slop, r, n))
    best_slop_recall = max(round1, key=lambda t: t[1])[0]
    print(f"  -> recall@1000最良: slop={best_slop_recall}")

    print("\n=== ラウンド2: recall@1000最良のslopに固定してmarker_boostを走査 ===")
    round1_at_best_slop = next((r, n) for s, r, n in round1 if s == best_slop_recall)
    round2 = []
    for marker_boost in MARKER_BOOST_GRID:
        if marker_boost == DEFAULT_MARKER_BOOST:
            r, n = round1_at_best_slop
            print(f"  (ラウンド1の結果を再利用) slop={best_slop_recall:4d} marker_boost={marker_boost:.1f}  "
                  f"recall@1000={r:.4f} nDCG@10={n:.4f}")
        else:
            r, n = try_params(best_slop_recall, marker_boost)
        round2.append((marker_boost, r, n))
    best_marker_boost_recall = max(round2, key=lambda t: t[1])[0]
    print(f"  -> recall@1000最良: marker_boost={best_marker_boost_recall}")

    print("\n" + "=" * 72)
    print(f"最良の組（recall@1000基準、{len(search_qids)}トピックのサブセット）: "
          f"slop={best_slop_recall}, marker_boost={best_marker_boost_recall}")
    best_r, best_n = evaluate_params(best_slop_recall, best_marker_boost_recall, search_qids, queries, webstyle, qrels, cache)
    print(f"  recall@1000={best_r:.4f}  nDCG@10={best_n:.4f}")

    print(f"\nキャッシュ: 呼び出し{cache.calls}回 / ヒット{cache.hits}回")

    out_path = os.path.join(RAG_DIR, "discourseboost_param_search_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "search_qids_n": len(search_qids),
            "round1_slop_sweep": [{"slop": s, "recall_1000": r, "ndcg_10": n} for s, r, n in round1],
            "round2_marker_boost_sweep": [{"marker_boost": b, "recall_1000": r, "ndcg_10": n} for b, r, n in round2],
            "best_slop": best_slop_recall,
            "best_marker_boost": best_marker_boost_recall,
            "best_recall_1000": best_r,
            "best_ndcg_10": best_n,
        }, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
