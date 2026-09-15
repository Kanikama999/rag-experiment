"""
真のBM25F（retriever.py bm25f()）のフィールド別長さ正規化パラメータ b_title/b_headings/
b_bodyを、実際のパイプライン（bm25f + RRF融合）で直接探索する。

evaluate_webstyle_bm25f.py（35トピックのサブセット、b_title=b_headings=b_body=0.4固定
=このインデックスのbody用similarity設定をそのまま流用）では、真のBM25Fが既存の線形和
方式（webstyle_narrative_fielded_recallopt）に対しnDCG@10で明確に劣るという結果だった
（0.4710→0.3201、consensus qrels）。title/headingsはbodyよりずっと短いテキストなので、
bodyと同じb=0.4を使うと長さ正規化が強すぎる（≒短いフィールド内での語の出現率の違いを
過大評価する）可能性がある——BM25F本来の文献（Robertson et al. 2004）でも
フィールドごとに別のbを使うことが前提とされている。本スクリプトはこれを実際に検証する。

探索方法: search_field_boosts.pyと同じく「探索フェーズは1/3サブセット、グリッドサーチ」
という構成を踏襲する。ただし今回は「候補プール取得＋IDF計算＋_mtermvectors」という
重い（ネットワークI/Oが必要な）処理をbm25f_prepare()で1回だけ行い、b_*を変えるだけの
軽い（純Python）再スコアリングはbm25f_score()で使い回す。b_*を変えても候補プール自体は
変わらない（プール取得はweights依存でb_*非依存）ため、この使い回しは常に正しい。

段階:
  1. title×headings 5x5グリッド（body=0.4固定、GRID=[0.0,0.2,0.4,0.6,0.8]）
  2. 1で見つかった最良(title,headings)を固定し、bodyだけ同じGRIDで振る

フィールド重み（title_weight/headings_weight/body_weight）はevaluate_webstyle_bm25f.py
と同じ既存最良比率（title=2,headings=1,body=1、recallopt）に固定する（重みのチューニングは
本スクリプトの対象外）。

使い方:
    python search_bm25f_field_b.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
from retriever import bm25f_prepare, bm25f_score, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PARENT_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(PARENT_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
QRELS_FILE = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")

TOPK = 1000
# 注意: PrepCacheはbm25f_prepare()の生データ（候補プール全件分のフィールド別term
# vector辞書）をSubquery単位でグリッドサーチ全体を通して保持し続けるため、メモリ使用量は
# ほぼ CANDIDATE_K × キャッシュされたSubquery数 に比例する。実際にCANDIDATE_K=3000
# （evaluate_webstyle_bm25f.pyのRETRIEVE_Kと同じ値）で試したところ、35トピック分
# （約154 Subquery）をキャッシュした時点でプロセスが18GB超のRSSを使い、共有マシンの
# 負荷急増（load average 50〜80、他ユーザーにも影響）を引き起こした。b_*のグリッド
# サーチはCANDIDATE_K自体を全b combo共通で固定していれば相対比較として妥当なので、
# ここでは大幅に小さい値を使う（絶対値としてのrecall@1000等はこの値に事実上律速される
# ため、evaluate_webstyle_bm25f.pyのCANDIDATE_K=3000での数値とは直接比較できない点に注意。
# 見つかったb_*は全105トピックの確認評価でCANDIDATE_K=3000に戻して検証すること）。
CANDIDATE_K = 200
QUERY_REPEAT = 5

TITLE_WEIGHT, HEADINGS_WEIGHT, BODY_WEIGHT = 2.0, 1.0, 1.0  # recallopt比率で固定
BM25_K1 = 0.9

B_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]
BODY_B_DEFAULT = 0.4  # 段階1でbodyを固定する値（このインデックスのsimilarity設定）

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
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


class PrepCache:
    """(title_q, headings_q, body_q) -> bm25f_prepare()の生データ、のキャッシュ。
    weights（title/headings/body_weight）は本スクリプト内で固定なので、b_*を変える
    グリッドサーチ全体を通してprepareは1回で済む（再スコアリングはbm25f_scoreで純Python、
    ネットワークI/O無し）。"""

    def __init__(self):
        self._cache = {}
        self.calls = self.hits = 0
        self.prepare_time = 0.0

    def get(self, title_q, headings_q, body_q):
        key = (title_q, headings_q, body_q)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        t0 = time.time()
        raw = bm25f_prepare(title_q, headings_q, body_q,
                             title_weight=TITLE_WEIGHT, headings_weight=HEADINGS_WEIGHT,
                             body_weight=BODY_WEIGHT, candidate_k=CANDIDATE_K)
        self.prepare_time += time.time() - t0
        self._cache[key] = raw
        return raw


def build_raw_lists(qids, queries, webstyle, cache, log_progress=False):
    """qid -> [(dq, raw), ...] のリスト（bm25f_prepareの結果、b_*非依存）を作る。
    グリッドサーチの前に1回だけ呼び、以降は評価のたびにbm25f_scoreだけ回す。"""
    raw_lists = {}
    t0 = time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        entry = webstyle[qid]
        pairs = dq_pairs_structured(entry)
        repeated_q = " ".join([q] * QUERY_REPEAT)
        raws = []
        for dq, doc in pairs:
            title_q = f"{repeated_q} {dq} {doc['title']}"
            headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
            body_q = f"{repeated_q} {dq} {doc['body']}"
            raws.append(cache.get(title_q, headings_q, body_q))
        raw_lists[qid] = raws
        if log_progress and (i % 5 == 0 or i == len(qids)):
            print(f"\r  prepare: {i}/{len(qids)} ({time.time() - t0:.0f}s)", end="", flush=True)
    if log_progress:
        print()
    return raw_lists


def evaluate_b(b_title, b_headings, b_body, qids, raw_lists, qrels):
    run = {}
    for qid in qids:
        lists = [bm25f_score(raw, k=TOPK, title_weight=TITLE_WEIGHT,
                              headings_weight=HEADINGS_WEIGHT, body_weight=BODY_WEIGHT,
                              b_title=b_title, b_headings=b_headings, b_body=b_body,
                              bm25_k1=BM25_K1)
                 for raw in raw_lists[qid] if raw is not None]
        lists = [lst for lst in lists if lst]
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        run[qid] = {docid: float(score) for docid, score in fused}

    target = [q for q in qids if q in qrels]
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m.replace(".", "_"): sum(r[m.replace(".", "_")] for r in results.values()) / n
           for m in METRIC_SPECS}
    return agg


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
    search_qids = valid_qids[::3]  # search_field_boosts.pyと同じ間引き（約1/3）
    print(f"全トピック: {len(valid_qids)}   探索用サブセット: {len(search_qids)}")
    print(f"weights固定: title={TITLE_WEIGHT}, headings={HEADINGS_WEIGHT}, body={BODY_WEIGHT}"
          f"   candidate_k={CANDIDATE_K}")

    cache = PrepCache()
    print("\n=== 候補プール取得・IDF計算・mtermvectors取得（b_*非依存、1回だけ） ===")
    raw_lists = build_raw_lists(search_qids, queries, webstyle, cache, log_progress=True)
    print(f"prepare: 呼び出し {cache.calls} 回 / キャッシュヒット {cache.hits} 回 / "
          f"合計 {cache.prepare_time:.0f}s")

    log = []

    def try_b(b_title, b_headings, b_body, stage):
        t0 = time.time()
        agg = evaluate_b(b_title, b_headings, b_body, search_qids, raw_lists, qrels)
        dt = time.time() - t0
        print(f"  [{stage}] b_title={b_title} b_headings={b_headings} b_body={b_body}  "
              f"recall@100={agg['recall_100']:.4f} recall@1000={agg['recall_1000']:.4f} "
              f"nDCG@10={agg['ndcg_cut_10']:.4f} P@100={agg['P_100']:.4f}  ({dt:.1f}s)")
        entry = {"stage": stage, "b_title": b_title, "b_headings": b_headings, "b_body": b_body, **agg}
        log.append(entry)
        return entry

    print(f"\n=== 段階1: title×headings {len(B_GRID)}x{len(B_GRID)}グリッド"
          f"（body={BODY_B_DEFAULT}固定） ===")
    for bt in B_GRID:
        for bh in B_GRID:
            try_b(bt, bh, BODY_B_DEFAULT, "stage1_title_headings")

    stage1_log = [r for r in log if r["stage"] == "stage1_title_headings"]
    best_ndcg = max(stage1_log, key=lambda r: r["ndcg_cut_10"])
    best_recall1000 = max(stage1_log, key=lambda r: r["recall_1000"])
    print(f"\n段階1 最良(nDCG@10): b_title={best_ndcg['b_title']} b_headings={best_ndcg['b_headings']}"
          f"  (nDCG@10={best_ndcg['ndcg_cut_10']:.4f})")
    print(f"段階1 最良(recall@1000): b_title={best_recall1000['b_title']} "
          f"b_headings={best_recall1000['b_headings']}  (recall@1000={best_recall1000['recall_1000']:.4f})")

    print(f"\n=== 段階2: bodyのみ{len(B_GRID)}通り "
          f"（b_title={best_ndcg['b_title']}, b_headings={best_ndcg['b_headings']}固定、nDCG@10最良点を採用） ===")
    for bb in B_GRID:
        try_b(best_ndcg["b_title"], best_ndcg["b_headings"], bb, "stage2_body")

    stage2_log = [r for r in log if r["stage"] == "stage2_body"]
    best_overall_ndcg = max(stage2_log, key=lambda r: r["ndcg_cut_10"])
    best_overall_recall1000 = max(stage2_log, key=lambda r: r["recall_1000"])

    print(f"\n{'=' * 72}")
    print("探索結果まとめ:")
    print(f"  nDCG@10最良: b_title={best_overall_ndcg['b_title']}, "
          f"b_headings={best_overall_ndcg['b_headings']}, b_body={best_overall_ndcg['b_body']}"
          f"  (nDCG@10={best_overall_ndcg['ndcg_cut_10']:.4f}, "
          f"recall@1000={best_overall_ndcg['recall_1000']:.4f})")
    print(f"  recall@1000最良: b_title={best_overall_recall1000['b_title']}, "
          f"b_headings={best_overall_recall1000['b_headings']}, b_body={best_overall_recall1000['b_body']}"
          f"  (recall@1000={best_overall_recall1000['recall_1000']:.4f}, "
          f"nDCG@10={best_overall_recall1000['ndcg_cut_10']:.4f})")
    print(f"  参考(デフォルト b=0.4/0.4/0.4): "
          + next((f"nDCG@10={r['ndcg_cut_10']:.4f} recall@1000={r['recall_1000']:.4f}"
                   for r in stage1_log if r["b_title"] == 0.4 and r["b_headings"] == 0.4), "見つからず"))

    out_path = os.path.join(RAG_DIR, "bm25f_field_b_search_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "search_qids": search_qids,
            "grid": B_GRID,
            "weights": {"title": TITLE_WEIGHT, "headings": HEADINGS_WEIGHT, "body": BODY_WEIGHT},
            "candidate_k": CANDIDATE_K,
            "best_stage1_ndcg": best_ndcg,
            "best_stage1_recall1000": best_recall1000,
            "best_overall_ndcg": best_overall_ndcg,
            "best_overall_recall1000": best_overall_recall1000,
            "log": log,
        }, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
