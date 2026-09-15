"""
posboostの効果が「narrativeのような長い深いクエリに特有」なのか、それとも
クエリ長に関係なく効く一般的な手法（FirstPの再実装）なのかを検証する対照実験。

同一コーパス（msmarco-v21-doc）・同一qrels（keystone_qrels_consensus/coverage、
TREC 2025 RAG公式のセグメント単位判定を文書粒度に集約したもの）で、クエリの長さだけを
3段階に変えてposboostの効果を比較する:

- Tier A（短い、平均30語弱）: Subquery単体（narrativeをLLMで分解した簡潔な質問文、
  1トピック平均4〜5個）をbm25_bodyで検索し、Subquery横断でRRF融合
- Tier B（中程度、narrative単体・繰り返しなし、平均40語程度）: narrativeそのまま
  （＝既存のbaseline条件と同じ入力）をbm25_bodyで検索
- Tier C（長い、既存の最良パイプライン、300〜400語規模）: narrative×5+Subquery+
  LLM生成疑似文書のfielded検索（title=2,headings=1,body=1）

各Tierについて、posboostなし/ありを比較し、絶対差・相対改善率を報告する。per-topicの
スコアはJSONに保存し、別途analyze_querylength_confound.pyで対応のあるbootstrap検定
（p値・95%信頼区間）を行う。

Tier間で相対改善率がほぼ一定なら、posboostはクエリ長に関係なく効く一般的な手法
（FirstP相当）であることが示唆される。Tierが長いほど改善率が大きいなら、
narrative特有の効果を主張できる材料になる。

使い方:
    python evaluate_querylength_confound.py [QUERY_REPEAT]   # 省略時は5（Tier Cのみに影響）
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import (bm25_body, bm25_bodyonly_posboost_discourseboost,
                        bm25_fielded, bm25_fielded_posboost, rrf_fuse, analyze_terms)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5

SPAN_END, SPAN_BOOST = 100, 15.0  # posboostの既存最良値（Tier A/B/Cで共通に使う）
TITLE_BOOST, HEADINGS_BOOST, BODY_BOOST = 2, 1, 1  # Tier Cのfieldedブースト（recallopt）

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


def load_qrels(path):
    qrels, skipped = {}, 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                skipped += 1
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


class CachedSimple:
    """(query_text, span_terms) -> bm25_bodyonly_posboost_discourseboost結果 のキャッシュ。
    markers=[]固定でdiscourseboostは常に無効（posboostの有無だけを見る実験のため）。
    span_terms=[]ならposboostも無効（つまり素のbm25_body相当になる）。"""

    def __init__(self, topk):
        self.topk = topk
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, query_text, span_terms):
        key = (query_text.strip(), tuple(span_terms))
        if not key[0]:
            return []
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self._cache[key] = bm25_bodyonly_posboost_discourseboost(
            query_text, list(span_terms), k=self.topk,
            span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
        return self._cache[key]

    def stats(self, label):
        return f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回"


class CachedFielded:
    """(title_q, headings_q, body_q, span_terms, use_posboost) -> 検索結果 のキャッシュ。"""

    def __init__(self, topk):
        self.topk = topk
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, title_q, headings_q, body_q, span_terms, use_posboost):
        key = (title_q.strip(), headings_q.strip(), body_q.strip(), tuple(span_terms), use_posboost)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        if use_posboost:
            self._cache[key] = bm25_fielded_posboost(
                title_q, headings_q, body_q, list(span_terms), k=self.topk,
                title_boost=TITLE_BOOST, headings_boost=HEADINGS_BOOST, body_boost=BODY_BOOST,
                span_end=SPAN_END, span_boost=SPAN_BOOST)
        else:
            self._cache[key] = bm25_fielded(
                title_q, headings_q, body_q, k=self.topk,
                title_boost=TITLE_BOOST, headings_boost=HEADINGS_BOOST, body_boost=BODY_BOOST)
        return self._cache[key]

    def stats(self, label):
        return f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回"


def fuse(lists, topk):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


def build_tierA(qids, queries, webstyle, search_simple, use_posboost):
    """Tier A: Subquery単体、posboostなし/あり。"""
    run = {}
    for qid in qids:
        entry = webstyle[qid]
        dqs = entry["decomposed_queries"]
        lists = []
        for dq in dqs:
            span_terms = analyze_terms(dq, field="body") if use_posboost else []
            lists.append(search_simple(dq, span_terms))
        run[qid] = fuse(lists, TOPK)
    return run


def build_tierB(qids, queries, search_simple, use_posboost):
    """Tier B: narrative単体（繰り返しなし）、posboostなし/あり。"""
    run = {}
    for qid in qids:
        q = queries[qid]
        span_terms = analyze_terms(q, field="body") if use_posboost else []
        run[qid] = fuse([search_simple(q, span_terms)], TOPK)
    return run


def build_tierC(qids, queries, webstyle, search_fielded, use_posboost):
    """Tier C: narrative×5+Subquery+疑似文書のfielded検索、posboostなし/あり。"""
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
            span_terms = analyze_terms(dq, field="body") if use_posboost else []
            lists.append(search_fielded(title_q, headings_q, body_q, span_terms, use_posboost))
        run[qid] = fuse(lists, TOPK)
    return run


def per_topic_scores(run, qrels, eval_qids):
    """{qid: {metric_key: value}} を返す（トピック単位、bootstrap検定用）。"""
    target = [q for q in eval_qids if q in qrels]
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run.get(q, {}) for q in target})
    return {q: {k: results[q][k] for k in METRIC_KEYS} for q in target}


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries
        and webstyle[qid].get("decomposed_queries")
        and any(d and d.strip() for d in webstyle[qid].get("query2doc_docs", []))
    )
    print(f"対象トピック: {len(valid_qids)}")

    search_simple = CachedSimple(RETRIEVE_K)
    search_fielded = CachedFielded(RETRIEVE_K)

    all_scores = {}  # {tier: {posboost_on: {qrelname: {qid: {metric: val}}}}}

    for tier_name, build_fn, args in [
        ("A_subquery", build_tierA, (valid_qids, queries, webstyle, search_simple)),
        ("B_narrative", build_tierB, (valid_qids, queries, search_simple)),
        ("C_fullpipeline", build_tierC, (valid_qids, queries, webstyle, search_fielded)),
    ]:
        all_scores[tier_name] = {}
        for use_posboost in [False, True]:
            t0 = time.time()
            run = build_fn(*args, use_posboost)
            print(f"  {tier_name} posboost={use_posboost}: {time.time()-t0:.0f}s")
            all_scores[tier_name][str(use_posboost)] = {}
            for name in QREL_SETS:
                scores = per_topic_scores(run, qrels[name], valid_qids)
                all_scores[tier_name][str(use_posboost)][name] = scores
                agg = {k: sum(v[k] for v in scores.values()) / len(scores) for k in METRIC_KEYS}
                print(f"    [{name}] n={len(scores)}  " +
                      "  ".join(f"{k}={agg[k]:.4f}" for k in METRIC_KEYS))

    print(f"\n{search_simple.stats('search_simple')}")
    print(search_fielded.stats('search_fielded'))

    out_path = os.path.join(RAG_DIR, "querylength_confound_per_topic_scores.json")
    with open(out_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
