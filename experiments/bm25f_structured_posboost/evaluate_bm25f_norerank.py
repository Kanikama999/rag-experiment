"""
bm25f_k1_length_sweep の k1b3_a0（BM25F再スコアリングあり）に対して、
「Stage 1（候補取得）のOpenSearchネイティブBM25の並びをそのまま使い、
Stage 2の自前BM25F再スコアリングを一切しなかったらどうなるか」を測る。

狙い:
  bm25f_prepare は2段階構成——
    Stage 1: OpenSearchネイティブBM25（k1=1.2既定、bool/should、
             title^2 headings^1 body^1）で候補プールtop RETRIEVE_K件を取得
    Stage 2: 純Pythonの自前BM25F（k1自由）でその候補プールだけ再スコアリング
  ——になっている。k1b3_a0の改善が「k1=3.0の再スコアリング」によるものなのか、
  それとも他の要因（候補プールサイズRETRIEVE_K=3000など）によるものかを切り分けるため、
  Stage 2を無効化しStage 1の順位をそのまま使った場合の性能を測る。

  Stage 1の並びをそのまま使う＝そのSubqueryのランキング = 検索ヒット順（rank順）。
  RRFは順位しか使わないので、スコア自体は不要（rank=1,2,3,...のダミースコアで足りる）。
  _mtermvectorsもIDF/avgdl計算も不要なので、フルスイープよりずっと軽い。

  weights(2:1:1)・candidate_k(=RETRIEVE_K)・クエリ構成（narrative×5+Subquery×1+
  生成テキスト）・RRF(k=60)はk1_length_sweepと完全に揃えてあるので、
  「Stage 2の有無」だけの差分になる。

使い方:
    python3 evaluate_bm25f_norerank.py [LIMIT]
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import client, INDEX, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000   # bm25f_k1_length_sweepのcandidate_kと揃える
QUERY_REPEAT = 5
TITLE_W, HEADINGS_W, BODY_W = 2.0, 1.0, 1.0   # 同上、weightsを揃える
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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


def stage1_only(title_text, headings_text, body_text, k):
    """bm25f_prepareのStage 1と同じ候補取得クエリだけを投げ、ネイティブBM25の
    ヒット順をそのまま返す（Stage 2の自前BM25F再スコアリングは一切行わない）。
    RRFは順位しか使わないため、スコアはダミーで良い。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": TITLE_W}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": HEADINGS_W}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": BODY_W}}})
    if not should:
        return []
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {"bool": {"should": should}},
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def evaluate(run, qrels, qids, label):
    target = [q for q in qids if q in qrels and q in run]
    if not target:
        print(f"[{label}] 採点対象なし")
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = ev.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    agg["n"] = n
    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg


def main():
    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt"))
             for n in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
        and any(d and d.get("body") for d in webstyle[qid].get("query2doc_docs_structured", []))
    )
    if LIMIT:
        valid_qids = valid_qids[:LIMIT]
    print(f"疑似文書がある {len(valid_qids)} クエリ")
    print(f"RETRIEVE_K={RETRIEVE_K} QUERY_REPEAT={QUERY_REPEAT} "
          f"weights=({TITLE_W},{HEADINGS_W},{BODY_W})（Stage 2の再スコアリングなし）")

    run = {}
    t0 = time.time()
    for i, qid in enumerate(valid_qids, 1):
        q = queries[qid]
        repeated_q = " ".join([q] * QUERY_REPEAT)
        lists = []
        for dq, doc in dq_pairs_structured(webstyle[qid]):
            headings_txt = " ".join(doc["headings"])
            title_q = f"{repeated_q} {dq} {doc['title']}"
            headings_q = f"{repeated_q} {dq} {headings_txt}"
            body_q = f"{repeated_q} {dq} {doc['body']}"
            lst = stage1_only(title_q, headings_q, body_q, RETRIEVE_K)
            if lst:
                lists.append(lst)
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        run[qid] = {d: float(s) for d, s in fused}
        if i % 10 == 0 or i == len(valid_qids):
            el = time.time() - t0
            print(f"\r  {i}/{len(valid_qids)} ({el:.0f}s)", end="", flush=True)
    print()

    summary = {}
    for name in QREL_SETS:
        agg = evaluate(run, qrels[name], valid_qids, f"norerank / {name}")
        if agg:
            summary[name] = agg

    out_path = os.path.join(RAG_DIR, "bm25f_norerank_result.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(valid_qids), "retrieve_k": RETRIEVE_K,
                   "weights": [TITLE_W, HEADINGS_W, BODY_W],
                   "summary": {"norerank": summary}}, f, ensure_ascii=False, indent=1)
    print(f"\n結果を保存: {out_path}")


if __name__ == "__main__":
    main()
