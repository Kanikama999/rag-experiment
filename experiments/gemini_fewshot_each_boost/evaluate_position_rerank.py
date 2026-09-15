"""
位置ブーストが効いている理由は「位置」なのか「窓による希釈低減」なのかを切り分ける。

## 問題

現行の位置ブーストは span_first(end=100, boost=15)、つまり**文書先頭の窓**に加点する。
これが効く理由には2つの説明があり、まだ区別できていない。

  説A（位置）: 要点は文書の冒頭に書かれやすいから、冒頭を見ると当たる（FirstP の直感）
  説B（希釈）: 窓で区切ると長い文書での語の希釈が減るから当たる。**冒頭である必要はない**

論文の主張は説A（「文書の要点は冒頭に書かれやすい」）に依拠しているので、この切り分けは
避けて通れない。傍証として、多段窓の実験（§7.10）では減衰なしの flat が最良で、
冒頭を強く優遇するほど悪化した。これは説B寄りの結果である。

## 切り分け方

FirstP（先頭の窓のみ）と MaxP（最良の窓を文書のどこからでも探す）を**同じ機構で**計算し、
比較する。MaxP は位置を一切見ない（どこにあってもよい）ので:

  MaxP ≈ FirstP  → 説B。効いているのは窓であって位置ではない
  FirstP > MaxP  → 説A。位置が本当に効いている
  MaxP > FirstP  → 窓を冒頭に固定しているのが損。最良箇所を探す方が良い

なお **MaxP は位置の手法ではない**。文書を固定長パッセージに分割して各々を独立に採点し、
その最大値を文書スコアにする手法（Dai & Callan 2019 ほか）で、解いているのは
「長い文書ではクエリ語が希釈される」問題である。位置から独立なので、本検証の対照として
ちょうどよい。

## 実装

位置は `_mtermvectors` の `positions: True` で取得する（語ごとの全出現位置が返る）。
painless の `_index` API（expert scripting）は ES7 以降で削除されており使えないため、
サーバ側スクリプトでの位置参照は不可。Python 側でのリランクが唯一の経路。

ペイロードが重い（位置ありで 78.7 KB/文書）ので、次の3点で軽くする:
  - リランク対象を上位 RERANK_N 件に絞る（それ以下の順位は基礎スコアのまま残す）
  - body フィールドのみ取得
  - docid 単位で位置をキャッシュ（同一トピック内の Subquery は narrative×5 を共有するため
    候補が大きく重複する）

## 条件

すべて同一の候補プール（位置ブーストなしの fielded 検索で上位 RETRIEVE_K 件）に対する
リランクで、加点の式だけが違う。最終スコア = 基礎BM25スコア + BOOST × 窓スコア。

  nopos      : 加点なし（リランクの土台）
  firstp     : 窓 [0,W) のみ。champion の span_first と同じ発想
  maxp       : 幅Wの窓を STRIDE ずつずらし、最良の窓を採用（位置は問わない）
  maxp_wlen  : maxp と同じだが、tf正規化を文書長でなく窓長で行う（本来のパッセージ検索寄り）
  decay      : 窓を使わず、初出位置の連続減衰 1/(1+p/DECAY_C) で加点（おまけ）
  champion_ref: 実際の span_first クエリ（外部参照。firstp がこれを再現するかの妥当性確認）

窓スコアは span_first の採点式に合わせ、窓内の出現回数を tf として
`Σ_t idf(t) · tf/(tf + k1·(1-b+b·dl/avgdl))` で計算する（span_first も文書長で正規化して
いることを explain で確認済み）。maxp_wlen だけ dl を窓長Wに置き換える。

使い方:
    python evaluate_position_rerank.py            # 35トピック（既定）
    python evaluate_position_rerank.py --full     # 105トピック
    python evaluate_position_rerank.py --limit 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import pytrec_eval

from retriever import (client, INDEX, bm25_equalweight_posboost_discourseboost,
                       bm25_fielded, analyze_terms, rrf_fuse,
                       _bm25f_avgdl, _bm25f_total_docs, _bm25f_combined_df)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
RERANK_N = 100          # 位置を取ってリランクする件数（ペイロード抑制）
QUERY_REPEAT = 5
W = 100                 # 窓幅。champion の span_end に合わせる
STRIDE = 50             # maxp の窓のずらし幅
BOOST = 15.0            # champion の span_boost に合わせる
BM25_K1, BM25_B = 0.9, 0.4
DECAY_C = 100.0         # decay 条件の減衰定数

METHODS = ["nopos", "firstp", "maxp", "maxp_wlen", "decay", "champion_ref"]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]

_pos_cache = {}          # docid -> {term: [positions]} / dl
_idf_cache = {}


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


def idf_of(term):
    if term not in _idf_cache:
        df = _bm25f_combined_df(term)
        N = _bm25f_total_docs()
        _idf_cache[term] = math.log(1 + (N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0
    return _idf_cache[term]


def fetch_positions(docids):
    """未キャッシュの docid について body の語→位置リストを取得してキャッシュへ。"""
    need = [d for d in docids if d not in _pos_cache]
    if not need:
        return
    for i in range(0, len(need), 100):
        chunk = need[i:i + 100]
        res = client.mtermvectors(index=INDEX, body={
            "ids": chunk,
            "parameters": {"fields": ["body"], "positions": True, "offsets": False,
                           "payloads": False, "term_statistics": False,
                           "field_statistics": False},
        })
        for doc in res.get("docs", []):
            terms = doc.get("term_vectors", {}).get("body", {}).get("terms", {})
            pos = {t: [tok["position"] for tok in info.get("tokens", [])]
                   for t, info in terms.items()}
            dl = sum(info["term_freq"] for info in terms.values())
            _pos_cache[doc.get("_id")] = {"pos": pos, "dl": dl}
        for d in chunk:                      # 取得できなかった分も空で埋めて再取得を防ぐ
            _pos_cache.setdefault(d, {"pos": {}, "dl": 0})


def window_score(entry, span_terms, avgdl, mode):
    """窓スコア。mode: firstp / maxp / maxp_wlen / decay"""
    pos, dl = entry["pos"], entry["dl"]
    if not pos or dl <= 0:
        return 0.0
    norm_dl = dl

    if mode == "decay":
        # 窓を使わず、語ごとの初出位置で連続減衰
        s = 0.0
        for t in span_terms:
            p = pos.get(t)
            if p:
                s += idf_of(t) / (1.0 + min(p) / DECAY_C)
        return s

    def score_window(lo, hi, length_for_norm):
        s = 0.0
        for t in span_terms:
            p = pos.get(t)
            if not p:
                continue
            tf = sum(1 for x in p if lo <= x < hi)
            if tf <= 0:
                continue
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * length_for_norm / avgdl)
            s += idf_of(t) * tf / denom
        return s

    if mode == "firstp":
        return score_window(0, W, norm_dl)

    best = 0.0
    for lo in range(0, max(dl - W, 0) + 1, STRIDE):
        ln = W if mode == "maxp_wlen" else norm_dl
        best = max(best, score_window(lo, lo + W, ln))
    return best


def build_run(method, qids, queries, prepared, avgdl, base_cache):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        rep = " ".join([q] * QUERY_REPEAT)
        lists = []
        for dq, doc, span_terms in prepared[qid]:
            key = (qid, dq)
            if method == "champion_ref":
                lists.append(bm25_equalweight_posboost_discourseboost(
                    f"{rep} {dq} {doc['title']}",
                    f"{rep} {dq} {' '.join(doc['headings'])}",
                    f"{rep} {dq} {doc['body']}", span_terms, k=RETRIEVE_K,
                    span_end=W, span_boost=BOOST, markers=[]))
                continue

            base = base_cache[key]
            if method == "nopos":
                lists.append(base)
                continue

            head = [d for d, _ in base[:RERANK_N]]
            fetch_positions(head)
            rescored = []
            for rank, (docid, sc) in enumerate(base):
                if rank < RERANK_N:
                    e = _pos_cache.get(docid)
                    bonus = (BOOST * window_score(e, span_terms, avgdl, method)) if e else 0.0
                    rescored.append((docid, sc + bonus))
                else:
                    rescored.append((docid, sc))
            rescored.sort(key=lambda x: -x[1])
            lists.append(rescored)

        run[qid] = ({d: float(s) for d, s in rrf_fuse(lists, top_n=TOPK)}
                    if lists else {})
        if i % 5 == 0 or i == len(qids):
            print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s, "
                  f"位置キャッシュ {len(_pos_cache)}件)", end="", flush=True)
    print()
    return run


def evaluate(run, qrels, qids):
    target = [q for q in qids if q in qrels and run.get(q)]
    if not target:
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
    n = len(res)
    out = {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}
    out["n"] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print("読み込み中...")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid = sorted(
        qid for qid in webstyle
        if qid in queries and webstyle[qid].get("decomposed_queries")
        and any(d and d.get("body") for d in webstyle[qid].get("query2doc_docs_structured", []))
    )
    qids = valid if args.full else valid[::3]
    if args.limit:
        qids = qids[:args.limit]

    prepared = {}
    for qid in qids:
        e = webstyle[qid]
        prepared[qid] = [(dq, doc, analyze_terms(dq, field="body"))
                         for dq, doc in zip(e["decomposed_queries"],
                                            e["query2doc_docs_structured"])
                         if doc and doc.get("body")]
    qids = [q for q in qids if prepared[q]]
    avgdl = _bm25f_avgdl("body")

    print(f"全 {len(valid)} / 対象 {len(qids)} トピック   "
          f"W={W} STRIDE={STRIDE} BOOST={BOOST} RERANK_N={RERANK_N} "
          f"DECAY_C={DECAY_C} avgdl(body)={avgdl:.0f}")
    print(f"条件: {', '.join(METHODS)}")
    print("=" * 72)

    # 位置ブーストなしの基礎ランキングを1回だけ取って全条件で使い回す
    print("基礎ランキング取得中...")
    base_cache, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        rep = " ".join([queries[qid]] * QUERY_REPEAT)
        for dq, doc, _st in prepared[qid]:
            base_cache[(qid, dq)] = bm25_fielded(
                f"{rep} {dq} {doc['title']}",
                f"{rep} {dq} {' '.join(doc['headings'])}",
                f"{rep} {dq} {doc['body']}", k=RETRIEVE_K,
                title_boost=1, headings_boost=1, body_boost=1)
        if i % 5 == 0 or i == len(qids):
            print(f"\r  {i}/{len(qids)} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    runs = {m: build_run(m, qids, queries, prepared, avgdl, base_cache) for m in METHODS}

    summary = {}
    for name in QREL_SETS:
        for m in METHODS:
            summary.setdefault(name, {})[m] = evaluate(runs[m], qrels[name], qids)

    labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
              "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
    print("\n" + "=" * 72)
    print(f"SUMMARY  対象 {len(qids)} トピック")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        header = "method".ljust(15) + "".join(labels[k].ljust(15) for k in METRIC_KEYS) + "n"
        print(header)
        print("-" * len(header))
        for m in METHODS:
            agg = summary[name].get(m)
            row = m.ljust(15)
            if not agg:
                print(row + "-")
                continue
            print(row + "".join(f"{agg[k]:.4f}".ljust(15) for k in METRIC_KEYS) + str(agg["n"]))

        f, mx = summary[name].get("firstp"), summary[name].get("maxp")
        if f and mx:
            print(f"\n  [{name}] FirstP vs MaxP（正なら FirstP=位置が有利、"
                  f"ゼロ近傍なら位置は効いていない）")
            for k in METRIC_KEYS:
                print(f"    {labels[k]:14s} {f[k] - mx[k]:+.4f}")
        ch, no = summary[name].get("champion_ref"), summary[name].get("nopos")
        if ch and f:
            print(f"  [{name}] 妥当性チェック firstp − champion_ref（0近傍なら再現）")
            for k in METRIC_KEYS:
                print(f"    {labels[k]:14s} {f[k] - ch[k]:+.4f}")
    print("=" * 72)

    out_path = os.path.join(RAG_DIR, "position_rerank_result.json")
    with open(out_path, "w") as fo:
        json.dump({"n_queries": len(qids), "qids": qids, "full": args.full,
                   "config": {"W": W, "STRIDE": STRIDE, "BOOST": BOOST,
                              "RERANK_N": RERANK_N, "RETRIEVE_K": RETRIEVE_K,
                              "DECAY_C": DECAY_C, "k1": BM25_K1, "b": BM25_B},
                   "summary": summary}, fo, ensure_ascii=False, indent=2)
    print(f"\n集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
