"""
文献で確立している融合手法を掃引する実験。

背景:
  sweep_rrf_score_transform.py では融合式を fused[d] += f(score_d)/(k+rank_d) と置き、
  f（分子）を46通り掃引した。結果、正規化も非線形変換も効かず、最も単純な f(s)=s
  （生スコア）が最良だった（consensus recall@1000 +0.0007、両半分で勝ち）。

  しかしあの掃引は **分子だけ** を動かし、分母 (k+rank) と k=60 を固定していた。
  データ融合の文献が重視しているのは、むしろ以下の2軸で、どちらも未検証だった:

    1. ヒット数の乗数（CombMNZ）
       「複数のリストに共通して出現する文書を強調する」という、f では表現できない軸。
       Fox & Shaw (1994)。CombSUM に「その文書がヒットしたリスト数」を掛ける。
    2. rank減衰の形と k
       Cormack et al. (2009) の RRF は k=60 が既定だが、Bruch et al. (TOIS 2023,
       arXiv:2210.11934) は RRF がパラメータに敏感であることを指摘している。

  また前回の「正規化が全て悪化した」という結果は、Montague & Aslam (2001) の
  「融合アルゴリズムより正規化が重要」という知見と一見矛盾する。ただしあの知見は
  スコアのスケールが比較不能な **異種システム** の融合を対象にしている。本実験では
  全リストが同一インデックス上の同一BM25関数で、違うのはクエリだけなのでスケールは
  最初から揃っている。正規化は「揃える」効果を持たず、リスト間の本物の差を消すだけ
  だった、と解釈できる。この解釈が正しければ CombSUM 系でも無正規化が強いはず。

掃引する手法:

  [順位ベース]
    rrf_k{k}    : sum 1/(k+rank)                 k を掃引（k=60 が現行champion）
    rr          : sum 1/rank                     RRFのk=0
    isr         : sum 1/rank^2                   Inverse Square Rank
    rbc_p{phi}  : sum (1-phi) phi^(rank-1)       Rank-Biased Centroids 相当の幾何減衰
    borda       : sum (N-rank+1)/N               Borda count

  [スコアベース: Fox & Shaw (1994)]
    combsum_{norm} : sum norm(s)
    combmnz_{norm} : sum norm(s) * (その文書がヒットしたリスト数)
    combanz_{norm} : sum norm(s) / (同上)
    combmax_{norm} : リスト横断の最大値
    norm は none / minmax / zscore / max / sumnorm(s/sum s)

  [ハイブリッド: 前回の勝者とMNZの組み合わせ]
    score_rrf_k{k} : sum s/(k+rank)              前回の最良（k=60）。kを掃引
    score_mnz_k{k} : (sum s/(k+rank)) * ヒット数  上記にMNZの乗数を掛けた版

  convex combination (Bruch et al.) について:
    あれは異種の2リスト（lexical/semantic）を alpha で重み付けする手法。本実験の
    リストは全て同質なSubqueryなので、対応物は「リスト重み付きCombSUM」になる。
    重みを一様にすれば CombSUM そのもの（＝下記で掃引済み）であり、重みを最適化した
    場合の上限は診断済み（オラクルで consensus recall@1000 +0.0022 しかない）。
    したがって新規条件としては追加しない。

検索結果のキャッシュ:
  融合式を変えても一次検索の結果は変わらないので、初回だけ検索してディスクに保存し、
  2回目以降は再利用する。以降この手の実験は数秒で回せる。

使い方:
    python sweep_fusion_methods.py                       # n5_s5 x idfweighted（recall最良構成）
    python sweep_fusion_methods.py --retrieval champion --subquery-repeat 1
    python sweep_fusion_methods.py --limit 3             # 動作確認
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

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]
PRIMARY = "recall_1000"

RRF_KS = [10, 20, 40, 60, 100, 200]
RBC_PHIS = [0.90, 0.95, 0.98, 0.99]
NORMS = ["none", "minmax", "zscore", "max", "sumnorm"]


# ============================================================
# 正規化
# ============================================================

def normalize(scores, kind):
    n = len(scores)
    if kind == "none":
        return scores.copy()
    if kind == "max":
        m = scores.max()
        return scores / m if m > 0 else np.ones(n)
    if kind == "minmax":
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo) if hi > lo else np.ones(n)
    if kind == "zscore":
        mu, sd = scores.mean(), scores.std()
        return (scores - mu) / sd if sd > 0 else np.zeros(n)
    if kind == "sumnorm":
        t = scores.sum()
        return scores / t if t > 0 else np.ones(n) / n
    raise ValueError(f"未知の正規化: {kind}")


# ============================================================
# 融合手法
# ============================================================

def fuse_rank(lists, kind, topn=TOPK):
    """順位のみを使う融合。"""
    fused = defaultdict(float)
    for lst in lists:
        n = len(lst)
        for rank, (docid, _) in enumerate(lst, start=1):
            if kind.startswith("rrf_k"):
                w = 1.0 / (float(kind[5:]) + rank)
            elif kind == "rr":
                w = 1.0 / rank
            elif kind == "isr":
                w = 1.0 / (rank * rank)
            elif kind.startswith("rbc_p"):
                phi = float(kind[5:])
                w = (1.0 - phi) * (phi ** (rank - 1))
            elif kind == "borda":
                w = (n - rank + 1) / n
            else:
                raise ValueError(f"未知の順位融合: {kind}")
            fused[docid] += w
    return _top(fused, topn)


def fuse_score(lists, family, norm_kind, topn=TOPK):
    """スコアベースの融合（Fox & Shaw 系）。"""
    acc = defaultdict(float)
    hits = defaultdict(int)
    best = defaultdict(lambda: -math.inf)
    for lst in lists:
        if not lst:
            continue
        vals = normalize(np.array([float(s) for _, s in lst], dtype=float), norm_kind)
        for (docid, _), v in zip(lst, vals):
            acc[docid] += float(v)
            hits[docid] += 1
            if float(v) > best[docid]:
                best[docid] = float(v)
    fused = {}
    for d, s in acc.items():
        if family == "combsum":
            fused[d] = s
        elif family == "combmnz":
            fused[d] = s * hits[d]
        elif family == "combanz":
            fused[d] = s / hits[d]
        elif family == "combmax":
            fused[d] = best[d]
        else:
            raise ValueError(f"未知のスコア融合: {family}")
    return _top(fused, topn)


def fuse_score_rrf(lists, k, mnz=False, topn=TOPK):
    """前回の勝者 sum s/(k+rank)。mnz=Trueならヒット数の乗数を掛ける。"""
    acc = defaultdict(float)
    hits = defaultdict(int)
    for lst in lists:
        for rank, (docid, score) in enumerate(lst, start=1):
            acc[docid] += float(score) / (k + rank)
            hits[docid] += 1
    if mnz:
        return _top({d: s * hits[d] for d, s in acc.items()}, topn)
    return _top(acc, topn)


def _top(fused, topn):
    return {d: float(s) for d, s in
            sorted(fused.items(), key=lambda x: -x[1])[:topn]}


def build_methods():
    """(ラベル, 融合関数) のリストを返す。"""
    ms = []
    for k in RRF_KS:
        ms.append((f"rrf_k{k}", lambda L, k=k: fuse_rank(L, f"rrf_k{k}")))
    ms.append(("rr", lambda L: fuse_rank(L, "rr")))
    ms.append(("isr", lambda L: fuse_rank(L, "isr")))
    for p in RBC_PHIS:
        ms.append((f"rbc_p{p}", lambda L, p=p: fuse_rank(L, f"rbc_p{p}")))
    ms.append(("borda", lambda L: fuse_rank(L, "borda")))
    for fam in ["combsum", "combmnz", "combanz"]:
        for nm in NORMS:
            ms.append((f"{fam}_{nm}",
                       lambda L, f=fam, n=nm: fuse_score(L, f, n)))
    for nm in ["none", "minmax"]:
        ms.append((f"combmax_{nm}", lambda L, n=nm: fuse_score(L, "combmax", n)))
    for k in RRF_KS:
        ms.append((f"score_rrf_k{k}", lambda L, k=k: fuse_score_rrf(L, k)))
        ms.append((f"score_mnz_k{k}", lambda L, k=k: fuse_score_rrf(L, k, mnz=True)))
    return ms


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
    """注意: RelevanceEvaluator は使い回すと2回目以降 recall_1000 が落ちるので毎回作る。"""
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


def retrieve_lists(args, webstyle, queries, valid_qids):
    """一次検索。結果はディスクにキャッシュし、2回目以降は再利用する。"""
    tag = (f"{args.retrieval}_n{args.narrative_repeat}_s{args.subquery_repeat}"
           f"_{args.model}" + (f"_limit{args.limit}" if args.limit else ""))
    cache_path = os.path.join(RAG_DIR, f"_cache_lists_{tag}.json")
    if os.path.exists(cache_path) and not args.refresh:
        print(f"検索結果キャッシュを読み込み: {os.path.basename(cache_path)}")
        with open(cache_path) as f:
            raw = json.load(f)
        return {q: [[(d, float(s)) for d, s in lst] for lst in lists]
                for q, lists in raw.items()}, cache_path

    search_champ = Cache(bm25_equalweight_posboost_discourseboost)
    search_weighted = Cache(bm25_fielded_weighted_posboost)
    out, t0 = {}, time.time()
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
                lists.append([(d, float(s)) for d, s in lst])
        if lists:
            out[qid] = lists
        if i % 10 == 0 or i == len(valid_qids):
            print(f"\r  検索: {i}/{len(valid_qids)} ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()
    print(search_champ.stats("champion検索"))
    print(search_weighted.stats("idf重み付き検索"))
    with open(cache_path, "w") as f:
        json.dump(out, f)
    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"検索結果を保存: {os.path.basename(cache_path)} ({size_mb:.1f} MB)")
    return out, cache_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(WEBSTYLE_FILES), default="gemini")
    ap.add_argument("--retrieval", choices=["champion", "idfweighted"], default="idfweighted")
    ap.add_argument("--narrative-repeat", type=int, default=5)
    ap.add_argument("--subquery-repeat", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh", action="store_true", help="検索キャッシュを無視して引き直す")
    args = ap.parse_args()

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

    per_topic_lists, _ = retrieve_lists(args, webstyle, queries, valid_qids)
    eval_qids = [q for q in valid_qids if q in per_topic_lists]
    n_sub = sum(len(per_topic_lists[q]) for q in eval_qids)
    print(f"トピック {len(eval_qids)} / Subquery {n_sub} 本")

    # CombMNZ の乗数が働く余地があるかを見る。ほとんどの文書が1本のリストにしか
    # 出現しないなら、MNZの乗数はほぼ定数1になり CombSUM と区別がつかなくなる。
    hist = defaultdict(int)
    for q in eval_qids:
        c = defaultdict(int)
        for lst in per_topic_lists[q]:
            for d, _ in lst:
                c[d] += 1
        for v in c.values():
            hist[v] += 1
    total = sum(hist.values())
    print("\n[診断] 文書が何本のSubqueryリストに出現したか（CombMNZの乗数の分布）")
    for h in sorted(hist):
        print(f"    {h}本: {hist[h]:>8} 件 ({100.0 * hist[h] / total:5.2f}%)")

    half_a = [q for i, q in enumerate(eval_qids) if i % 2 == 0]
    half_b = [q for i, q in enumerate(eval_qids) if i % 2 == 1]

    methods = build_methods()
    print(f"\n融合手法 {len(methods)} 通りを評価します（追加の検索は発生しません）")
    results, t0 = {}, time.time()
    for j, (label, fn) in enumerate(methods, 1):
        run = {q: fn(per_topic_lists[q]) for q in eval_qids}
        entry = {}
        for name in QREL_SETS:
            pt = eval_per_topic(run, qrels[name], eval_qids)
            entry[name] = {"all": agg(pt, eval_qids),
                           "half_a": agg(pt, half_a),
                           "half_b": agg(pt, half_b)}
        results[label] = entry
        print(f"\r  {j}/{len(methods)} ({time.time() - t0:.0f}s) {label:<22}",
              end="", flush=True)
    print()

    # 基準は rrf_k60 = 現行champion の等重みRRF
    base = results["rrf_k60"]
    print("\n" + "=" * 104)
    print(f"[結果] 主指標 {PRIMARY}。rrf_k60（現行の等重みRRF）との差")
    print("  ★ = 105トピックを交互2分割した両方の半分で現行を上回った")
    for name in QREL_SETS:
        b = base[name]
        if b["all"] is None:
            continue
        print(f"\n--- qrels={name}  現行(rrf_k60)の{PRIMARY} = {b['all'][PRIMARY]:.4f} ---")
        rows = []
        for label, e in results.items():
            if e[name]["all"] is None:
                continue
            d_all = e[name]["all"][PRIMARY] - b["all"][PRIMARY]
            d_a = e[name]["half_a"][PRIMARY] - b["half_a"][PRIMARY]
            d_b = e[name]["half_b"][PRIMARY] - b["half_b"][PRIMARY]
            rows.append((d_all, d_a, d_b, (d_a > 0 and d_b > 0), label,
                         e[name]["all"]["ndcg_cut_10"] - b["all"]["ndcg_cut_10"],
                         e[name]["all"][PRIMARY]))
        rows.sort(key=lambda r: -r[0])
        print(f"    {'融合手法':<22}{PRIMARY:>12}{'Δ':>10}{'Δhalf_a':>10}"
              f"{'Δhalf_b':>10}{'両方勝':>7}{'ΔnDCG@10':>11}")
        for d_all, d_a, d_b, both, label, d_ndcg, absval in rows:
            mark = " ★" if both else "  "
            print(f"    {label:<22}{absval:>12.4f}{d_all:>+10.4f}{d_a:>+10.4f}"
                  f"{d_b:>+10.4f}{mark:>7}{d_ndcg:>+11.4f}")

    tag = (f"{args.retrieval}_n{args.narrative_repeat}_s{args.subquery_repeat}_{args.model}")
    out_path = os.path.join(RAG_DIR, f"fusion_methods_sweep_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "retrieval": args.retrieval,
                   "narrative_repeat": args.narrative_repeat,
                   "subquery_repeat": args.subquery_repeat,
                   "n_topics": len(eval_qids), "n_subqueries": n_sub,
                   "primary_metric": PRIMARY,
                   "half_a_qids": half_a, "half_b_qids": half_b,
                   "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\n結果を保存: {out_path}")


if __name__ == "__main__":
    main()
