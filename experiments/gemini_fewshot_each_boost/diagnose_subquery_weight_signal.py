"""
RRF融合でSubqueryを等価に扱う設計を見直す前に、その前提を測る診断スクリプト。

背景:
  現行championはSubquery横断のRRF融合を等重み（1/(k+rank)の単純和）で行っている。
  「質のいいランキングと悪いランキングが等価に扱われる」という問題意識から、
  分子にBM25スコアを置く案（BM25_d/(60+rank)）が出た。しかしこの案は
  **「BM25スコアが高いSubquery＝質のいいランキング」という未検証の前提**に乗っている。

  過去に scorerrf（Subquery単位の重み＝上位100件のBM25スコア合計）を試したときは
  効果がほぼ誤差だった（recall@1000 0.5463→0.5474）。ただしあれは旧構成
  （recallopt(2,1,1)、posboostなし）上での測定で、現行champion上では未検証。

  heading role実験の教訓（heading_role_experiment_summary.md 5節）に従い、
  実装の前にシグナルの実在と大きさを測る。

本スクリプトが1回の検索パスで出すもの:

  [診断A] Subquery単体の「実際の質」と「重み候補」の**トピック内**順位相関（Spearman）。
      トピックをまたぐと難易度で交絡するため、必ずトピック内で取り105トピック分を平均する。
      重み候補: 上位100件のBM25スコア合計 / 同平均 / 最大スコア / クエリ語数。
      クエリ語数を混ぜてあるのは「スコアが大きいのは質が高いからではなく単にクエリが
      長いからではないか」という交絡（evaluate_querylength_confound.py と同じ懸念）を
      同時に測るため。

  [診断B] トピック内での重み候補のばらつき（変動係数）。全Subqueryの重みがほぼ同値なら、
      相関が正でも順位を動かす力がない。scorerrfが誤差に終わった説明になりうる。

  [融合式の直接比較] 検索結果をメモリに持っているので、追加検索なしで融合式だけ差し替えて
      実測できる。
        equal_rrf         : 現行champion（1/(k+rank)）
        raw_score_rrf     : 提案式そのまま（BM25_d/(k+rank)）
        norm_score_rrf_t* : リスト内最大スコアで正規化した版（(s_d/s_max)^t/(k+rank)）。
                            t=0 で equal_rrf に一致するので、現行を含む連続族になる。
        listw_sum100      : scorerrf相当（リスト単位の定数重み＝上位100件のスコア合計）

  [オラクル上限] 各Subqueryを「その単体ランキングの実際の質」で重み付けして融合した場合の
      スコア。qrelsを使うので実現不可能だが、**リスト単位の重み付けという方針全体の天井**を
      与える。この天井がequal_rrfとほぼ同じなら、どんな重み付け式を設計しても無駄と分かる。

使い方:
    python diagnose_subquery_weight_signal.py                # 105トピック（約26分）
    python diagnose_subquery_weight_signal.py --limit 2      # 動作確認
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pytrec_eval
from scipy.stats import spearmanr

from retriever import bm25_equalweight_posboost_discourseboost, analyze_terms

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

WEBSTYLE_FILES = {
    "gemini": os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json"),
    "gpt56terra": os.path.join(RAG_DIR,
                               "multi_query2doc_decomposed_webstyle_gpt56terra_L200.json"),
}

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000        # championと揃える
SPAN_END = 100           # championと同じ
SPAN_BOOST = 15.0        # championと同じ
RRF_K = 60               # championと同じ
SCORE_WEIGHT_TOPN = 100  # 重み候補「スコア合計」に使う上位件数（scorerrfと同じ定義）

# 正規化スコアRRFの温度。0 は equal_rrf と数学的に一致するので対照として入れてある。
NORM_TEMPERATURES = [0.0, 0.25, 0.5, 1.0, 2.0]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]

# 診断Aで相関を取る「重み候補」
WEIGHT_CANDIDATES = ["sum_top100", "mean_top100", "max_score", "n_query_terms"]
# 診断Aで相関を取る「単体ランキングの質」
QUALITY_KEYS = ["ndcg_cut_10", "recall_100"]


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4 or line.startswith("#"):
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


class Cache:
    """検索関数の結果をキャッシュする薄いラッパ（2x2スクリプトと同じ方式）。"""

    def __init__(self, fn):
        self.fn = fn
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, key, *args, **kwargs):
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self._cache[key] = self.fn(*args, **kwargs)
        return self._cache[key]

    def stats(self, label):
        return f"  {label}: 検索 {self.calls} 回 / キャッシュヒット {self.hits} 回"


# ============================================================
# 融合式（すべて同じ rank_lists を入力に取る。追加検索は不要）
# ============================================================

def _top(fused, topn):
    return {d: float(s) for d, s in
            sorted(fused.items(), key=lambda x: -x[1])[:topn]}


def fuse_equal(lists, topn=TOPK, k=RRF_K):
    """現行champion。1/(k+rank) の単純和。"""
    fused = defaultdict(float)
    for lst in lists:
        for rank, (docid, _) in enumerate(lst, start=1):
            fused[docid] += 1.0 / (k + rank)
    return _top(fused, topn)


def fuse_listweight(lists, weights, topn=TOPK, k=RRF_K):
    """リスト単位の定数重み（scorerrf方式）。w_i/(k+rank)。"""
    fused = defaultdict(float)
    for w, lst in zip(weights, lists):
        for rank, (docid, _) in enumerate(lst, start=1):
            fused[docid] += float(w) / (k + rank)
    return _top(fused, topn)


def fuse_raw_score(lists, topn=TOPK, k=RRF_K):
    """提案式そのまま。BM25生スコアを分子に置く: s_d/(k+rank)。"""
    fused = defaultdict(float)
    for lst in lists:
        for rank, (docid, score) in enumerate(lst, start=1):
            fused[docid] += float(score) / (k + rank)
    return _top(fused, topn)


def fuse_norm_score(lists, t=1.0, topn=TOPK, k=RRF_K):
    """リスト内最大スコアで正規化した版: (s_d/s_max)^t/(k+rank)。
    t=0 なら全項が1になり fuse_equal と一致する（対照）。"""
    fused = defaultdict(float)
    for lst in lists:
        smax = max((float(s) for _, s in lst), default=0.0)
        if smax <= 0:
            smax = 1.0
        for rank, (docid, score) in enumerate(lst, start=1):
            fused[docid] += ((float(score) / smax) ** t) / (k + rank)
    return _top(fused, topn)


def normalize_weights(ws):
    """平均1に正規化する。全部0なら等重みに落とす。"""
    ws = [max(float(w), 0.0) for w in ws]
    total = sum(ws)
    if total <= 0:
        return [1.0] * len(ws)
    return [w * len(ws) / total for w in ws]


# ============================================================
# 評価
# ============================================================

def score_single(ranking, qrels_one, qid):
    """1本のランキング（[(docid, score), ...]）を qrels で採点して指標dictを返す。
    そのトピックのqrelsが無ければ None。

    注意: pytrec_eval.RelevanceEvaluator は使い回せない。同じインスタンスで
    evaluate() を2回目以降呼ぶと最大カットオフの指標（ここでは recall_1000）が
    結果から落ちる。必ず呼び出しごとに作り直すこと。"""
    if not qrels_one:
        return None
    ev = pytrec_eval.RelevanceEvaluator({qid: qrels_one}, METRICS)
    res = ev.evaluate({qid: {d: float(s) for d, s in ranking}})
    if qid not in res:
        return None
    return {m: res[qid][m] for m in METRIC_KEYS}


def evaluate_run(run, qrels, eval_qids, label):
    target = [q for q in eval_qids if q in qrels and run.get(q)]
    if not target:
        print(f"[{label}] 採点対象なし")
        return None, 0
    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}
    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(WEBSTYLE_FILES), default="gemini")
    ap.add_argument("--query-repeat", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="先頭Nトピックだけ回す（動作確認用）")
    args = ap.parse_args()

    print(f"読み込み中... model={args.model}")
    with open(WEBSTYLE_FILES[args.model]) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
        and any(d and d.get("body") for d in webstyle[qid].get("query2doc_docs_structured", []))
    )
    if args.limit:
        valid_qids = valid_qids[:args.limit]
    print(f"疑似文書がある {len(valid_qids)} クエリ")
    print(f"TOPK={TOPK} RETRIEVE_K={RETRIEVE_K} QUERY_REPEAT={args.query_repeat} "
          f"SPAN_END={SPAN_END} SPAN_BOOST={SPAN_BOOST} RRF_K={RRF_K}")
    print("=" * 72)

    search_champion = Cache(bm25_equalweight_posboost_discourseboost)

    # ------------------------------------------------------------
    # 1パス: Subqueryごとにchampionの一次検索を回し、結果と重み候補を貯める
    # ------------------------------------------------------------
    per_topic = {}   # qid -> {"lists": [...], "weights": {cand: [...]}, "quality": {name: [dict|None]}}
    t0 = time.time()
    for i, qid in enumerate(valid_qids, 1):
        entry = webstyle[qid]
        docs = entry.get("query2doc_docs_structured") or []
        dqs = entry.get("decomposed_queries") or []
        q = queries[qid]
        repeated_q = " ".join([q] * args.query_repeat)

        lists, wcand, quality = [], defaultdict(list), {n: [] for n in QREL_SETS}
        for dq, doc in zip(dqs, docs):
            if not doc or not doc.get("body"):
                continue
            title_q = f"{repeated_q} {dq} {doc['title']}"
            headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
            body_q = f"{repeated_q} {dq} {doc['body']}"
            span_terms = analyze_terms(dq, field="body")
            lst = search_champion(
                (title_q, headings_q, body_q, tuple(span_terms)),
                title_q, headings_q, body_q, span_terms, k=RETRIEVE_K,
                span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
            if not lst:
                continue
            lists.append(lst)

            scores = [float(s) for _, s in lst[:SCORE_WEIGHT_TOPN]]
            wcand["sum_top100"].append(sum(scores))
            wcand["mean_top100"].append(sum(scores) / max(len(scores), 1))
            wcand["max_score"].append(max(scores) if scores else 0.0)
            # 交絡チェック用: このSubqueryの検索クエリの語数
            wcand["n_query_terms"].append(float(len(analyze_terms(body_q, field="body"))))

            for name in QREL_SETS:
                quality[name].append(
                    score_single(lst, qrels[name].get(qid), qid))

        if lists:
            per_topic[qid] = {"lists": lists, "weights": dict(wcand), "quality": quality}

        if i % 10 == 0 or i == len(valid_qids):
            print(f"\r  検索: {i}/{len(valid_qids)} ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()
    print(search_champion.stats("bm25_equalweight_posboost_discourseboost"))

    eval_qids = [q for q in valid_qids if q in per_topic]
    n_sub = sum(len(per_topic[q]["lists"]) for q in eval_qids)
    print(f"トピック {len(eval_qids)} / Subquery {n_sub} 本 "
          f"（平均 {n_sub / max(len(eval_qids), 1):.1f} 本/トピック）")

    out = {"model": args.model, "n_topics": len(eval_qids), "n_subqueries": n_sub,
           "query_repeat": args.query_repeat, "rrf_k": RRF_K}

    # ------------------------------------------------------------
    # 診断A: トピック内でのSpearman相関（重み候補 vs 単体の質）
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[診断A] Subquery単体の質と重み候補の トピック内 順位相関（Spearman）")
    print("  ※トピック内で相関を取り、トピック間で平均する（難易度交絡を避けるため）")
    print("  ※Subqueryが2本未満のトピックは相関が定義できないので除外")
    diagA = {}
    for name in QREL_SETS:
        diagA[name] = {}
        print(f"\n  --- qrels={name} ---")
        print(f"    {'重み候補':<16}{'質指標':<16}{'平均ρ':>9}{'中央値ρ':>10}"
              f"{'ρ>0の割合':>11}{'n_topics':>10}")
        for cand in WEIGHT_CANDIDATES:
            for qk in QUALITY_KEYS:
                rhos = []
                for qid in eval_qids:
                    d = per_topic[qid]
                    xs = d["weights"][cand]
                    ys = [m[qk] if m else None for m in d["quality"][name]]
                    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
                    if len(pairs) < 2:
                        continue
                    xa = [p[0] for p in pairs]
                    ya = [p[1] for p in pairs]
                    if len(set(xa)) < 2 or len(set(ya)) < 2:
                        continue  # 定数列は相関が定義できない
                    rho = spearmanr(xa, ya).statistic
                    if rho is not None and not np.isnan(rho):
                        rhos.append(float(rho))
                if rhos:
                    mean_rho = float(np.mean(rhos))
                    med_rho = float(np.median(rhos))
                    pos = float(np.mean([r > 0 for r in rhos]))
                else:
                    mean_rho = med_rho = pos = float("nan")
                diagA[name][f"{cand}|{qk}"] = {
                    "mean_rho": mean_rho, "median_rho": med_rho,
                    "frac_positive": pos, "n_topics": len(rhos)}
                print(f"    {cand:<16}{qk:<16}{mean_rho:>9.4f}{med_rho:>10.4f}"
                      f"{pos:>11.3f}{len(rhos):>10}")
    out["diagnosis_A_spearman"] = diagA

    # ------------------------------------------------------------
    # 診断B: トピック内での重み候補のばらつき（変動係数）
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[診断B] トピック内での重み候補のばらつき（変動係数 CV = 標準偏差/平均）")
    print("  ※CVが小さいほど「どのSubqueryも同じ重み」＝重み付けしても順位が動かない")
    diagB = {}
    print(f"    {'重み候補':<16}{'平均CV':>10}{'中央値CV':>11}{'最大/最小の中央値':>20}")
    for cand in WEIGHT_CANDIDATES:
        cvs, ratios = [], []
        for qid in eval_qids:
            xs = [float(x) for x in per_topic[qid]["weights"][cand]]
            if len(xs) < 2:
                continue
            m = float(np.mean(xs))
            if m > 0:
                cvs.append(float(np.std(xs) / m))
            lo = min(xs)
            if lo > 0:
                ratios.append(max(xs) / lo)
        diagB[cand] = {
            "mean_cv": float(np.mean(cvs)) if cvs else float("nan"),
            "median_cv": float(np.median(cvs)) if cvs else float("nan"),
            "median_max_min_ratio": float(np.median(ratios)) if ratios else float("nan"),
            "n_topics": len(cvs)}
        b = diagB[cand]
        print(f"    {cand:<16}{b['mean_cv']:>10.4f}{b['median_cv']:>11.4f}"
              f"{b['median_max_min_ratio']:>20.3f}")
    out["diagnosis_B_dispersion"] = diagB

    # ------------------------------------------------------------
    # 融合式の直接比較（追加検索なし）+ オラクル上限
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("[融合式の比較] 同じ検索結果を融合式だけ差し替えて実測")

    runs = {}
    runs["equal_rrf"] = {q: fuse_equal(per_topic[q]["lists"]) for q in eval_qids}
    runs["raw_score_rrf"] = {q: fuse_raw_score(per_topic[q]["lists"]) for q in eval_qids}
    for t in NORM_TEMPERATURES:
        runs[f"norm_score_rrf_t{t}"] = {
            q: fuse_norm_score(per_topic[q]["lists"], t=t) for q in eval_qids}
    runs["listw_sum100"] = {
        q: fuse_listweight(per_topic[q]["lists"],
                           normalize_weights(per_topic[q]["weights"]["sum_top100"]))
        for q in eval_qids}

    # オラクル: そのSubquery単体の実際の質を重みにする（qrels使用＝実現不可能な上限）
    oracle_names = []
    for name in QREL_SETS:
        for qk in QUALITY_KEYS:
            label = f"oracle_{name}_{qk}"
            oracle_names.append((label, name))
            run = {}
            for q in eval_qids:
                d = per_topic[q]
                ws = [(m[qk] if m else 0.0) for m in d["quality"][name]]
                if not any(w > 0 for w in ws):
                    run[q] = fuse_equal(d["lists"])   # qrelsが無い/全0なら等重み
                else:
                    run[q] = fuse_listweight(d["lists"], normalize_weights(ws))
            runs[label] = run

    summary = {}
    for method, run in runs.items():
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            # オラクルは自分を作るのに使ったqrelsでのみ意味があるので、そちらだけ出す
            if method.startswith("oracle_") and not method.startswith(f"oracle_{name}_"):
                continue
            agg, _ = evaluate_run(run, qrels[name], eval_qids, f"{method} / {name}")
            summary.setdefault(name, {})[method] = agg
    out["fusion_comparison"] = summary

    out_path = os.path.join(
        RAG_DIR, f"subquery_weight_signal_diagnosis_{args.model}_rep{args.query_repeat}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n診断結果を保存: {out_path}")


if __name__ == "__main__":
    main()
