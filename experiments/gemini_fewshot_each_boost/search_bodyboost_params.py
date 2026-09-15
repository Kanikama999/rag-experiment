"""
現状の最良設定（webstyle_narrative_equalweight_posboost_discourseboost、title=headings=body=1
均等、posboost span_end=100/span_boost=15、discourseboost slop=20/marker_boost=3）は
3フィールドを均等重みにしているが、posboost・discourseboostのボーナス加点はどちらも
bodyフィールドに対してだけかかっている。つまりbodyは既に「隠れた優遇」を受けている
状態だが、それでも明示的にbody_boostを上げるとさらに伸びるか、bm25_fielded_posboost_
discourseboost（title_boost/headings_boost/body_boostを個別に指定できる版）で
body_boostだけを走査して確認する。title_boost=headings_boost=1に固定。

探索フェーズは全105トピックの1/3（約35トピック、valid_qids[::3]）だけを使う。

使い方:
    python search_bodyboost_params.py
"""

from __future__ import annotations

import json
import os
import time

import pytrec_eval

from retriever import bm25_fielded_posboost_discourseboost, rrf_fuse, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5

TITLE_BOOST, HEADINGS_BOOST = 1, 1  # equalweight版(title:headings:body=1:1:1)に合わせて固定
SPAN_END, SPAN_BOOST = 100, 15.0  # posboostの既存最良値
SLOP, MARKER_BOOST = 20, 3.0  # discourseboostの既存最良値（≒初期値）

BODY_BOOST_GRID = [1.0, 1.5, 2.0, 3.0, 5.0]
DEFAULT_BODY_BOOST = 1.0

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
    """(title_q, headings_q, body_q, span_terms, body_boost) -> 検索結果 のキャッシュ。"""

    def __init__(self):
        self._cache = {}
        self.calls = self.hits = 0

    def search(self, title_q, headings_q, body_q, span_terms, body_boost):
        key = (title_q, headings_q, body_q, tuple(span_terms), body_boost)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        r = bm25_fielded_posboost_discourseboost(
            title_q, headings_q, body_q, list(span_terms), k=RETRIEVE_K,
            title_boost=TITLE_BOOST, headings_boost=HEADINGS_BOOST, body_boost=body_boost,
            span_end=SPAN_END, span_boost=SPAN_BOOST, slop=SLOP, marker_boost=MARKER_BOOST)
        self._cache[key] = r
        return r


def evaluate_params(body_boost, qids, queries, webstyle, qrels, cache):
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
            lists.append(cache.search(title_q, headings_q, body_q, span_terms, body_boost))
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

    def try_params(body_boost, tag=""):
        t0 = time.time()
        recall_1000, ndcg_10 = evaluate_params(body_boost, search_qids, queries, webstyle, qrels, cache)
        dt = time.time() - t0
        print(f"  {tag}body_boost={body_boost:.1f}  "
              f"recall@1000={recall_1000:.4f} nDCG@10={ndcg_10:.4f}  ({dt:.0f}s)")
        log.append({"body_boost": body_boost, "recall_1000": recall_1000, "ndcg_10": ndcg_10})
        return recall_1000, ndcg_10

    print("\n=== body_boostを走査（title=headings=1固定、posboost/discourseboostは既存最良値） ===")
    results = []
    for body_boost in BODY_BOOST_GRID:
        r, n = try_params(body_boost)
        results.append((body_boost, r, n))
    best_recall = max(results, key=lambda t: t[1])[0]
    best_ndcg = max(results, key=lambda t: t[2])[0]
    print(f"  -> recall@1000最良: body_boost={best_recall}   nDCG@10最良: body_boost={best_ndcg}")

    print(f"\nキャッシュ: 呼び出し{cache.calls}回 / ヒット{cache.hits}回")

    out_path = os.path.join(RAG_DIR, "bodyboost_param_search_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "search_qids_n": len(search_qids),
            "body_boost_sweep": [{"body_boost": b, "recall_1000": r, "ndcg_10": n} for b, r, n in results],
            "best_body_boost_recall": best_recall,
            "best_body_boost_ndcg": best_ndcg,
        }, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
