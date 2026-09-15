"""
webstyle_narrative_fielded_recallopt_posboost（bm25_fielded_posboost、Subquery本文の語が
文書bodyの先頭span_end語以内に出現したらspan_boost倍で加点するspan_first）の
span_end / span_boostを、実際のパイプラインで直接探索する。

search_field_boosts.py と同じ考え方（生スコアの回帰ではなく評価指標そのものを直接
最大化する）を踏襲。ただしspan_end/span_boostは2軸とも「クエリの構造自体」に効くため
（title/headings/bodyのブースト比のように後からスコアを再合成できない）、素朴な
座標降下法（1軸ずつ、2ラウンド）で探索する。まずspan_boost=2.0固定でspan_endを走査し、
最良のspan_endに固定してspan_boostを走査する。

探索フェーズは実行時間短縮のため全105トピックの1/3（約35トピック、valid_qids[::3]で
search_field_boosts.pyと同じ抽出方法）だけを使う。最後に見つかった最良の組を全105トピックで
確認評価する（別途evaluate_decomposed_webstyle_bm25structure.pyのMSM_RATIOと同様、
定数を書き換えて再実行する）。

title/headings/bodyのブーストはrecallopt（title=2,headings=1,body=1）に固定する。

使い方:
    python search_posboost_params.py
"""

from __future__ import annotations

import json
import os
import time

import pytrec_eval

from retriever import bm25_fielded_posboost, rrf_fuse, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5

TITLE_BOOST, HEADINGS_BOOST, BODY_BOOST = 2, 1, 1  # recalloptに固定

SPAN_END_GRID = [100]  # ラウンド1でほぼ無感応(0.268〜0.270)と分かったのでspan_boost探索に絞る
SPAN_BOOST_GRID = [8.0, 10.0, 15.0, 20.0, 30.0, 50.0]
DEFAULT_SPAN_END = 100
DEFAULT_SPAN_BOOST = 8.0  # 前回探索の最良値を起点にする

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
    """(title_q, headings_q, body_q, span_terms, span_end, span_boost) -> 検索結果 のキャッシュ。"""

    def __init__(self):
        self._cache = {}
        self.calls = self.hits = 0

    def search(self, title_q, headings_q, body_q, span_terms, span_end, span_boost):
        key = (title_q, headings_q, body_q, tuple(span_terms), span_end, span_boost)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        r = bm25_fielded_posboost(title_q, headings_q, body_q, list(span_terms), k=RETRIEVE_K,
                                   title_boost=TITLE_BOOST, headings_boost=HEADINGS_BOOST,
                                   body_boost=BODY_BOOST, span_end=span_end, span_boost=span_boost)
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
    search_qids = valid_qids[::3]  # 探索フェーズ用の部分集合（約1/3）
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

    print("\n=== ラウンド1: span_boost=2.0固定でspan_endを走査 ===")
    round1 = []
    for span_end in SPAN_END_GRID:
        r, n = try_params(span_end, DEFAULT_SPAN_BOOST)
        round1.append((span_end, r, n))
    best_end_recall = max(round1, key=lambda t: t[1])[0]
    best_end_ndcg = max(round1, key=lambda t: t[2])[0]
    print(f"  -> recall@1000最良: span_end={best_end_recall}   nDCG@10最良: span_end={best_end_ndcg}")

    print("\n=== ラウンド2: recall@1000最良のspan_endに固定してspan_boostを走査 ===")
    # DEFAULT_SPAN_BOOST(=2.0)の点はラウンド1で既に計算済み（span_boost=2.0固定で
    # span_endを走査したので、best_end_recallの点はそこに含まれている）なので再利用する。
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
    print(f"  -> recall@1000最良: span_boost={best_boost_recall}")

    print("\n" + "=" * 72)
    print(f"最良の組（recall@1000基準、{len(search_qids)}トピックのサブセット）: "
          f"span_end={best_end_recall}, span_boost={best_boost_recall}")
    best_r, best_n = evaluate_params(best_end_recall, best_boost_recall, search_qids, queries, webstyle, qrels, cache)
    print(f"  recall@1000={best_r:.4f}  nDCG@10={best_n:.4f}")
    print(f"（参考: ラウンド1探索(span_end=50, span_boost=2.0)の105トピックでの結果は "
          f"recall@1000=0.2816, nDCG@10=0.5062 だった）")

    print(f"\nキャッシュ: 呼び出し{cache.calls}回 / ヒット{cache.hits}回")

    out_path = os.path.join(RAG_DIR, "posboost_param_search_result_round2.json")
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
