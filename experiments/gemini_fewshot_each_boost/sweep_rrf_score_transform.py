"""
RRF融合の分子に置く関数 f(score) を掃引する実験。

背景:
  現行championのRRFは 1/(k+rank) の等重み和で、Subqueryごとのランキングの質を区別しない。
  「BM25_d/(k+rank) にすればスコアの高い文書を重く扱える」という案を診断した結果
  （subquery_weight_signal_diagnosis_gemini_rep5.json）、champion構成の上では
  全指標で現行をわずかに上回った（consensus recall@1000 0.3095 -> 0.3101、
  coverage 0.5995 -> 0.6042）。ただしBM25の生スコアはリスト間でスケールが揃っておらず、
  正規化や非線形変換を挟む余地が大きい。

  本スクリプトは融合式を
        fused[d] += f(score_d) / (k + rank_d)
  と一般化し、f を「正規化 x 非線形変換」の組み合わせで掃引する。

  検索は1パスで済む（融合式を変えても一次検索の結果は同じ）ため、f を何十通り試しても
  追加の検索コストはゼロ。

f の構成:
  [正規化] リスト（Subquery）内で計算する
    const      : f=1。数学的に現行の等重みRRFと一致するはずの対照（正当性チェック）
    none       : 正規化なし。生のBM25スコア（＝最初の提案そのまま）
    max        : s / s_max
    minmax     : (s - s_min) / (s_max - s_min)
    l2         : s / sqrt(sum s_i^2)
    zscore     : (s - mu) / sigma          ※負値を取りうる（下記の注意参照）
    zscore_shift : zscoreを最小値0までシフトした非負版
    mad        : (s - median) / (1.4826 * MAD)   ※負値を取りうる
    mad_shift  : madの非負版
    percentile : リスト内での上位割合 (N - rank + 1)/N
                 ※これはrankのみの関数なので、スコアの情報を完全に捨てた対照になる。
                   percentileがequal_rrfと大差なければ「スコアには順位以上の情報がある」
                   という主張の反証になる。

  [非線形変換] 0〜1に正規化された値に対して掛ける（max / minmax の上でのみ掃引）
    identity   : そのまま
    pow{g}     : x^g            高スコアの強調(g>1)・圧縮(g<1)
    log        : log(1+x)       極端な差を圧縮
    sigmoid{k} : 1/(1+exp(-k(x-0.5)))  中央0.5を境に滑らかに強調
    tanh{k}    : tanh(k x)
    tan{a}     : tan(a x)       閾値付近で急激に強調（a<pi/2、発散を防ぐため上限クリップ）
    sat{c}     : x/(x+c)        BM25と同じ飽和の発想
    exp{lam}   : 1-exp(-lam x)  同上

注意（負値になる正規化について）:
  zscore / mad は平均以下の文書に負の寄与を与える。RRFは本来「ヒットした文書は必ず
  加点される」融合なので、負の寄与は設計思想と衝突する。比較のため素の版も回すが、
  非負版（*_shift）の方が解釈しやすい。

過学習への対策:
  45通り前後を掃引して最大値を拾うと、差が0.005程度である以上ほぼ確実にノイズを選ぶ。
  そこでトピックを2分割し、各fについて「両方の半分で equal_rrf を上回ったか」を判定する。
  全体で1位でも片方の半分で負けているfは信用しない。

使い方:
    python sweep_rrf_score_transform.py                          # n5_s5 x idfweighted（recall最良構成）
    python sweep_rrf_score_transform.py --retrieval champion --subquery-repeat 1
    python sweep_rrf_score_transform.py --limit 3                # 動作確認
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import pytrec_eval

from retriever import (bm25_equalweight_posboost_discourseboost,
                       bm25_fielded_weighted_posboost, analyze_terms)
from term_weights import idf_term_weights

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
RETRIEVE_K = 1000
SPAN_END = 100
SPAN_BOOST = 15.0
RRF_K = 60

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]

# 主指標（recall@1000最優先の方針に合わせる）
PRIMARY = "recall_1000"


# ============================================================
# f = 非線形変換 ∘ 正規化
# ============================================================

def normalize(scores, kind):
    """リスト内スコア配列を正規化する。scoresは降順（rank順）のnumpy配列。"""
    n = len(scores)
    if kind == "const":
        return np.ones(n)
    if kind == "none":
        return scores.copy()
    if kind == "percentile":
        # rank のみの関数。スコア情報を捨てた対照。
        return (n - np.arange(n)) / n
    if kind == "max":
        m = scores.max()
        return scores / m if m > 0 else np.ones(n)
    if kind == "minmax":
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo) if hi > lo else np.ones(n)
    if kind == "l2":
        nrm = math.sqrt(float((scores ** 2).sum()))
        return scores / nrm if nrm > 0 else np.ones(n)
    if kind in ("zscore", "zscore_shift"):
        mu, sd = scores.mean(), scores.std()
        z = (scores - mu) / sd if sd > 0 else np.zeros(n)
        return z - z.min() if kind == "zscore_shift" else z
    if kind in ("mad", "mad_shift"):
        med = np.median(scores)
        mad = np.median(np.abs(scores - med))
        v = (scores - med) / (1.4826 * mad) if mad > 0 else np.zeros(n)
        return v - v.min() if kind == "mad_shift" else v
    raise ValueError(f"未知の正規化: {kind}")


def transform(x, kind):
    """0〜1に正規化された値への非線形変換。"""
    if kind == "identity":
        return x
    if kind.startswith("pow"):
        g = float(kind[3:])
        return np.power(np.clip(x, 0.0, None), g)
    if kind == "log":
        return np.log1p(np.clip(x, 0.0, None))
    if kind.startswith("sigmoid"):
        k = float(kind[7:])
        return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
    if kind.startswith("tanh"):
        k = float(kind[4:])
        return np.tanh(k * x)
    if kind.startswith("tan"):
        a = float(kind[3:])
        # a*x が pi/2 に近づくと発散するので、tan(1.5) 相当で頭打ちにする
        return np.tan(np.clip(a * x, 0.0, 1.5))
    if kind.startswith("sat"):
        c = float(kind[3:])
        return x / (x + c)
    if kind.startswith("exp"):
        lam = float(kind[3:])
        return 1.0 - np.exp(-lam * x)
    raise ValueError(f"未知の変換: {kind}")


# 掃引するf。(正規化, 変換) の組。
NORMS_ONLY = ["const", "none", "max", "minmax", "l2",
              "zscore", "zscore_shift", "mad", "mad_shift", "percentile"]
TRANSFORMS = ["pow0.25", "pow0.5", "pow2.0", "pow4.0", "log",
              "sigmoid4.0", "sigmoid8.0", "sigmoid12.0",
              "tanh1.0", "tanh3.0", "tan1.0", "tan1.4",
              "sat0.1", "sat0.3", "sat1.0",
              "exp1.0", "exp3.0", "exp5.0"]
# 変換は0〜1に収まる正規化の上でのみ意味があるので max / minmax に掛ける
TRANSFORM_BASES = ["max", "minmax"]


def build_variants():
    out = [(nm, "identity") for nm in NORMS_ONLY]
    for base in TRANSFORM_BASES:
        for tr in TRANSFORMS:
            out.append((base, tr))
    return out


def fuse_with_f(lists, norm_kind, tr_kind, k=RRF_K, topn=TOPK):
    """fused[d] += f(score_d)/(k+rank_d)。listsは[(docid, score), ...]のリスト。"""
    fused = defaultdict(float)
    for lst in lists:
        if not lst:
            continue
        scores = np.array([float(s) for _, s in lst], dtype=float)
        vals = transform(normalize(scores, norm_kind), tr_kind)
        for rank, ((docid, _), v) in enumerate(zip(lst, vals), start=1):
            fused[docid] += float(v) / (k + rank)
    return {d: float(s) for d, s in
            sorted(fused.items(), key=lambda x: -x[1])[:topn]}


# ============================================================
# 入出力・評価
# ============================================================

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


def eval_per_topic(run, qrels, qids):
    """トピック別の指標を返す。RelevanceEvaluatorは使い回すと最大カットオフの指標
    （recall_1000）が2回目以降落ちるので、呼び出しごとに作り直す。"""
    target = [q for q in qids if q in qrels and run.get(q)]
    if not target:
        return {}
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
    return {q: {m: res[q][m] for m in METRIC_KEYS} for q in res}


def agg(per_topic, qids):
    sel = [per_topic[q] for q in qids if q in per_topic]
    if not sel:
        return None
    return {m: sum(r[m] for r in sel) / len(sel) for m in METRIC_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(WEBSTYLE_FILES), default="gemini")
    ap.add_argument("--retrieval", choices=["champion", "idfweighted"], default="idfweighted",
                    help="検索構成。idfweighted=recall@1000基準の最良")
    ap.add_argument("--narrative-repeat", type=int, default=5)
    ap.add_argument("--subquery-repeat", type=int, default=5,
                    help="Subqueryの繰り返し回数。5=n5_s5（recall最良）、1=現champion相当")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = (f"{args.retrieval}_n{args.narrative_repeat}_s{args.subquery_repeat}"
           f"_{args.model}")
    print(f"読み込み中... model={args.model} retrieval={args.retrieval} "
          f"narrative x{args.narrative_repeat} / subquery x{args.subquery_repeat}")
    with open(WEBSTYLE_FILES[args.model]) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt"))
             for n in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
        and any(d and d.get("body") for d in webstyle[qid].get("query2doc_docs_structured", []))
    )
    if args.limit:
        valid_qids = valid_qids[:args.limit]
    print(f"疑似文書がある {len(valid_qids)} クエリ")

    search_champ = Cache(bm25_equalweight_posboost_discourseboost)
    search_weighted = Cache(bm25_fielded_weighted_posboost)

    # ---- 一次検索（1パスだけ） ----
    per_topic_lists = {}
    t0 = time.time()
    for i, qid in enumerate(valid_qids, 1):
        entry = webstyle[qid]
        docs = entry.get("query2doc_docs_structured") or []
        dqs = entry.get("decomposed_queries") or []
        q = queries[qid]
        narr = " ".join([q] * args.narrative_repeat) if args.narrative_repeat else ""
        lists = []
        for dq, doc in zip(dqs, docs):
            if not doc or not doc.get("body"):
                continue
            sub = " ".join([dq] * args.subquery_repeat)
            prefix = f"{narr} {sub}".strip()
            span_terms = analyze_terms(dq, field="body")
            if args.retrieval == "champion":
                title_q = f"{prefix} {doc['title']}"
                headings_q = f"{prefix} {' '.join(doc['headings'])}"
                body_q = f"{prefix} {doc['body']}"
                lst = search_champ((title_q, headings_q, body_q, tuple(span_terms)),
                                   title_q, headings_q, body_q, span_terms, k=RETRIEVE_K,
                                   span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
            else:
                w = idf_term_weights(doc)
                key = (prefix, tuple(w["title"]), tuple(w["headings"]),
                       tuple(w["body"]), tuple(span_terms))
                lst = search_weighted(key, prefix, w["title"], w["headings"], w["body"],
                                      span_terms, k=RETRIEVE_K,
                                      title_boost=1, headings_boost=1, body_boost=1,
                                      span_end=SPAN_END, span_boost=SPAN_BOOST)
            if lst:
                lists.append(lst)
        if lists:
            per_topic_lists[qid] = lists
        if i % 10 == 0 or i == len(valid_qids):
            print(f"\r  検索: {i}/{len(valid_qids)} ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()
    print(search_champ.stats("champion検索"))
    print(search_weighted.stats("idf重み付き検索"))

    eval_qids = [q for q in valid_qids if q in per_topic_lists]
    n_sub = sum(len(per_topic_lists[q]) for q in eval_qids)
    print(f"トピック {len(eval_qids)} / Subquery {n_sub} 本")

    # 過学習チェック用にトピックを2分割（qidのソート順で交互に振る＝決定的）
    half_a = [q for i, q in enumerate(eval_qids) if i % 2 == 0]
    half_b = [q for i, q in enumerate(eval_qids) if i % 2 == 1]

    # ---- f を掃引（追加検索なし） ----
    variants = build_variants()
    print(f"\nf を {len(variants)} 通り評価します（追加の検索は発生しません）")
    results = {}
    t0 = time.time()
    for j, (nm, tr) in enumerate(variants, 1):
        label = f"{nm}|{tr}"
        run = {q: fuse_with_f(per_topic_lists[q], nm, tr) for q in eval_qids}
        entry = {}
        for name in QREL_SETS:
            pt = eval_per_topic(run, qrels[name], eval_qids)
            entry[name] = {
                "all": agg(pt, eval_qids),
                "half_a": agg(pt, half_a),
                "half_b": agg(pt, half_b),
            }
        results[label] = entry
        print(f"\r  {j}/{len(variants)} ({time.time() - t0:.0f}s) {label:<24}",
              end="", flush=True)
    print()

    # ---- 正当性チェック ----
    # const|identity は f=1 なので、定義上 1/(k+rank) の等重みRRF（現行champion）と
    # 完全に一致しなければならない。独立に計算して突き合わせる。
    base = results["const|identity"]
    ref_run = {}
    for q in eval_qids:
        fused = defaultdict(float)
        for lst in per_topic_lists[q]:
            for rank, (docid, _) in enumerate(lst, start=1):
                fused[docid] += 1.0 / (RRF_K + rank)
        ref_run[q] = {d: float(s) for d, s in
                      sorted(fused.items(), key=lambda x: -x[1])[:TOPK]}
    print("\n[正当性チェック] const|identity vs 独立に計算した等重みRRF")
    all_ok = True
    for name in QREL_SETS:
        ref = agg(eval_per_topic(ref_run, qrels[name], eval_qids), eval_qids)
        got = base[name]["all"]
        if ref is None or got is None:
            print(f"  {name}: 採点対象なし（スキップ）")
            continue
        diffs = {m: abs(ref[m] - got[m]) for m in METRIC_KEYS}
        worst = max(diffs.values())
        status = "一致" if worst < 1e-9 else f"不一致（最大差 {worst:.2e}）"
        if worst >= 1e-9:
            all_ok = False
        print(f"  {name}: {status}  " +
              "  ".join(f"{m}={got[m]:.4f}" for m in METRIC_KEYS))
    if not all_ok:
        raise SystemExit("正当性チェックに失敗しました。融合の実装を確認してください。")

    # ---- 出力 ----
    print("\n" + "=" * 100)
    print(f"[結果] 主指標 {PRIMARY}。const|identity（現行の等重みRRF）との差で表示")
    print("  half_a / half_b は105トピックを交互に2分割したもの。")
    print("  ★ = 両方の半分で現行を上回った（＝ノイズで拾った可能性が低い）")
    for name in QREL_SETS:
        b = base[name]
        if b["all"] is None:
            continue
        print(f"\n--- qrels={name}  現行の{PRIMARY} = {b['all'][PRIMARY]:.4f} ---")
        rows = []
        for label, e in results.items():
            if e[name]["all"] is None:
                continue
            d_all = e[name]["all"][PRIMARY] - b["all"][PRIMARY]
            d_a = e[name]["half_a"][PRIMARY] - b["half_a"][PRIMARY]
            d_b = e[name]["half_b"][PRIMARY] - b["half_b"][PRIMARY]
            both = (d_a > 0) and (d_b > 0)
            rows.append((d_all, d_a, d_b, both, label,
                         e[name]["all"]["ndcg_cut_10"] - b["all"]["ndcg_cut_10"]))
        rows.sort(key=lambda r: -r[0])
        print(f"    {'f':<24}{'Δ' + PRIMARY:>13}{'Δhalf_a':>10}{'Δhalf_b':>10}"
              f"{'両方勝':>7}{'ΔnDCG@10':>11}")
        for d_all, d_a, d_b, both, label, d_ndcg in rows:
            mark = " ★" if both else "  "
            print(f"    {label:<24}{d_all:>+13.4f}{d_a:>+10.4f}{d_b:>+10.4f}"
                  f"{mark:>7}{d_ndcg:>+11.4f}")

    out = {"model": args.model, "retrieval": args.retrieval,
           "narrative_repeat": args.narrative_repeat,
           "subquery_repeat": args.subquery_repeat,
           "n_topics": len(eval_qids), "n_subqueries": n_sub,
           "rrf_k": RRF_K, "primary_metric": PRIMARY,
           "half_a_qids": half_a, "half_b_qids": half_b,
           "results": results}
    out_path = os.path.join(RAG_DIR, f"rrf_score_transform_sweep_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n結果を保存: {out_path}")


if __name__ == "__main__":
    main()
