"""
クエリ成分の寄与を指標別に分解する（2^4 完全要因計画）。

背景:
これまでのアブレーションは「検索側」（どのフィールドで引くか・位置ブーストを足すか）は
切り分けてあるが、**クエリ本文の成分**を分離した実験が無い。champion は
「narrative×5 + Subquery×1 + 疑似参照文書」の全部入りで、各成分の取り分が不明のまま。
「無駄に全部入りにせず、どの成分が recall に効き、どの成分が precision に効くのかを
分離せよ」という指摘に答える。

設計（繰り返しは×5に固定。4要因すべて2水準の完全要因）:
    N: narrative×5        なし / あり
    S: Subquery×5         なし / あり
    P: 疑似参照文書        なし / あり   （文章として生成。title/headings/body）
    E: 拡張語リスト        なし / あり   （語の列挙。title 8 / heading 20 / body 40）

各フィールド f のクエリ本文は
    q_f = narrative×5 + Subquery×5 + 疑似文書のf相当部分 + 拡張語リストのf相当部分
で組む（narrative/Subquery は3フィールド共通、P と E はフィールドごとに対応部分）。
champion と同じ均等重み(1,1,1)・Subquery単位検索・RRF(k=60)・RETRIEVE_K=1000。

P と E は元々「代替案」の関係（同じサブクエリから同じモデルが生成し、フィールド対応も
語数もほぼ同じ。文章か語の列挙かだけが違う）だが、実測すると語彙の重なりは44%程度しか
なく、半分以上はお互いに無い語。したがって P=E=あり は単なる冗長ではなく**語彙倍増条件**
であり、両者が飽和し合うか加算的かを測る意味がある。

2つのarmを回す:
- posboost なし … クエリ成分の純粋な寄与。**主結果**（15条件。全offは空クエリなので除外）
- posboost あり … champion と同じ機構下での寄与（16条件。全offでも「疑似文書を使わない
                    位置ブーストのみ」という §3.1 の条件になるので残す）

【重要な注意】posboost arm では span_terms = analyze_terms(Subquery) を champion と同じく
常に使う。つまり **S=なし の条件でも Subquery の情報が span_terms 経由で入る**ので、
posboost arm における S の主効果は過小評価（下限）になる。成分の帰属を論じるときは
posboost なし arm を使うこと。

検算用の既知条件:
    N1_S1_P1_E0_pb1 = champion + Subquery×5（§7.7）  consensus nDCG@10 = 0.5177
    N0_S1_P1_E0_pb1 = §7.7「narrative なし + Subquery×5」
    N0_S0_P0_E0_pb1 = 疑似文書を使わない位置ブーストのみ（§3.1）

使い方:
    python3 evaluate_query_component_grid.py [SUBSET_STRIDE] [LIMIT]
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import (bm25_fielded, bm25_equalweight_posboost_discourseboost,
                        analyze_terms, rrf_fuse)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
TERMLIST_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_termlist_T8_H20_B40.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000
QUERY_REPEAT = 5
SPAN_END, SPAN_BOOST = 100, 15.0
N_WORKERS = 8      # トピック単位の検索は互いに独立なので並列化できる（結果は逐次実行と同一）
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

FACTORS = ["N", "S", "P", "E"]
METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]

CKPT = os.path.join(RAG_DIR, "_cache_query_component_grid.json")


def cond_name(n, s, p, e, pb):
    return f"N{n}_S{s}_P{p}_E{e}_pb{int(pb)}"


def conditions():
    out = []
    for pb in (False, True):
        for n in (0, 1):
            for s in (0, 1):
                for p in (0, 1):
                    for e in (0, 1):
                        if not (n or s or p or e) and not pb:
                            continue        # 空クエリ
                        out.append((n, s, p, e, pb))
    return out


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4 or line.startswith("#"):
                continue
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    return qrels


def load_queries(path):
    q = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                q[o["id"]] = o["title"]
    return q


def build_field_texts(narrative, dq, pdoc, terms, n_on, s_on, p_on, e_on):
    """各フィールドのクエリ本文を組む。N/S は3フィールド共通、P/E はフィールド対応部分。"""
    common = []
    if n_on:
        common += [narrative] * QUERY_REPEAT
    if s_on:
        common += [dq] * QUERY_REPEAT
    base = " ".join(common)

    per_field = []
    for f, pkey, ekey in (("title", "title", "title_terms"),
                           ("headings", "headings", "heading_terms"),
                           ("body", "body", "body_terms")):
        parts = [base]
        if p_on:
            v = pdoc[pkey]
            parts.append(" ".join(v) if isinstance(v, list) else v)
        if e_on:
            parts.append(" ".join(terms[ekey]))
        per_field.append(" ".join(x for x in parts if x).strip())
    return tuple(per_field)


def build_span_cache(qids, webstyle):
    """位置ブーストの判定語を全サブクエリぶん先に作っておく（並列実行前に一度だけ。
    analyze_terms は検索と違い結果を共有辞書に書くので、並列化の前に潰しておく）。"""
    cache = {}
    for qid in qids:
        for i, dq in enumerate(webstyle[qid]["decomposed_queries"]):
            cache[(qid, i)] = analyze_terms(dq)
    return cache


def _run_one_topic(qid, n_on, s_on, p_on, e_on, use_pb, queries, webstyle, termlist, span_cache):
    narrative = queries[qid]
    we, te = webstyle[qid], termlist[qid]
    lists = []
    for i, dq in enumerate(we["decomposed_queries"]):
        pdoc = we["query2doc_docs_structured"][i]
        if not pdoc or not pdoc.get("body"):
            continue
        t, h, b = build_field_texts(narrative, dq, pdoc,
                                     te["query2doc_docs_terms"][i],
                                     n_on, s_on, p_on, e_on)
        if use_pb:
            lists.append(bm25_equalweight_posboost_discourseboost(
                t, h, b, span_cache[(qid, i)], k=RETRIEVE_K,
                span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[]))
        else:
            lists.append(bm25_fielded(t, h, b, k=RETRIEVE_K,
                                       title_boost=1, headings_boost=1, body_boost=1))
    lists = [l for l in lists if l]
    return qid, {d: float(sc) for d, sc in (rrf_fuse(lists, top_n=TOPK) if lists else [])}


def build_run(n_on, s_on, p_on, e_on, use_pb, qids, queries, webstyle, termlist, span_cache):
    """トピック単位で並列に検索する。各トピックの RRF 融合は他トピックに依存しないので、
    逐次実行と完全に同じ結果になる（検証済み）。"""
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_run_one_topic, qid, n_on, s_on, p_on, e_on, use_pb,
                           queries, webstyle, termlist, span_cache) for qid in qids]
        return dict(f.result() for f in futs)


def evaluate(run, qrels, qids):
    target = [q for q in qids if q in qrels and q in run]
    if not target:
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    res = ev.evaluate({q: run[q] for q in target})
    if not res:
        return None
    n = len(res)
    agg = {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}
    agg["n"] = n
    return agg


def found_relevant(run, qrels, qids):
    """[固有貢献の逆引き用] 各トピックについて「qrelsの正解文書のうち、どれを何位で
    拾ったか」を記録する。ランキング全体（105×1000件）を保存すると数百MBになるが、
    正解文書だけなら桁違いに小さく、かつ後段の分析

        - この正解文書はどの素材が見つけたか
        - ある素材だけが見つけた正解は何件か（＝固有貢献）
        - 素材間で見つけた正解はどれだけ重なるか

    に必要な情報はすべてこれで足りる。**追加検索なしで③が出せるようにするための記録。**
    rel>0 の文書のみ対象（rel=0 は非正解なので固有貢献の議論に無関係）。"""
    out = {}
    for qid in qids:
        if qid not in qrels or qid not in run:
            continue
        ranked = sorted(run[qid].items(), key=lambda x: -x[1])
        rank_of = {d: r for r, (d, _) in enumerate(ranked, 1)}
        hits = {d: rank_of[d] for d, rel in qrels[qid].items()
                if rel > 0 and d in rank_of}
        if hits:
            out[qid] = hits
    return out


def main():
    print("読み込み中...")
    webstyle = json.load(open(WEBSTYLE_FILE))["results"]
    termlist = json.load(open(TERMLIST_FILE))["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}

    valid = sorted(q for q in webstyle
                    if q in queries and q in termlist and webstyle[q].get("decomposed_queries"))
    qids = valid[::SUBSET_STRIDE]
    if LIMIT:
        qids = qids[:LIMIT]

    conds = conditions()
    print(f"トピック {len(qids)}   条件 {len(conds)}   RETRIEVE_K={RETRIEVE_K}  QUERY_REPEAT={QUERY_REPEAT}")

    print("位置ブーストの判定語を準備中...")
    span_cache = build_span_cache(qids, webstyle)
    results = {}
    if os.path.exists(CKPT):
        c = json.load(open(CKPT))
        if c.get("n_topics") == len(qids) and c.get("design") == "2^4":
            results = c["results"]
            print(f"チェックポイント復帰: {len(results)} 条件済み")

    t0 = time.time()
    for i, (n, s, p, e, pb) in enumerate(conds, 1):
        name = cond_name(n, s, p, e, pb)
        if name in results:
            continue
        ts = time.time()
        run = build_run(n, s, p, e, pb, qids, queries, webstyle, termlist, span_cache)
        rec = {"config": {"narrative": n, "subquery": s, "pseudo": p,
                           "expansion": e, "posboost": int(pb)},
               "search_sec": round(time.time() - ts, 1)}
        for qs in QREL_SETS:
            rec[qs] = evaluate(run, qrels[qs], qids)
            rec[f"found_{qs}"] = found_relevant(run, qrels[qs], qids)
        results[name] = rec
        con = rec["consensus"]
        print(f"  [{i}/{len(conds)}] {name}  {time.time()-ts:.0f}s  "
              f"nDCG@10={con['ndcg_cut_10']:.4f}  R@1000={con['recall_1000']:.4f}  "
              f"(経過{(time.time()-t0)/60:.0f}分)", flush=True)
        json.dump({"n_topics": len(qids), "design": "2^4", "results": results}, open(CKPT, "w"))

    suffix = "" if SUBSET_STRIDE == 1 and not LIMIT else f"_stride{SUBSET_STRIDE}_n{len(qids)}"
    meta = {"n_topics": len(qids), "qids": qids, "design": "2^4",
            "retrieve_k": RETRIEVE_K, "query_repeat": QUERY_REPEAT,
            "span_end": SPAN_END, "span_boost": SPAN_BOOST}

    # 集計値だけの軽いファイル（①②の分析用）
    slim = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("found_")}
            for k, v in results.items()}
    out = os.path.join(RAG_DIR, f"query_component_grid{suffix}.json")
    json.dump({**meta, "results": slim}, open(out, "w"), indent=2)
    print(f"\n-> {out}")

    # 正解文書の逆引き記録（③の分析用）
    found = {k: {kk: vv for kk, vv in v.items() if kk.startswith("found_")}
             for k, v in results.items()}
    out2 = os.path.join(RAG_DIR, f"query_component_found_rels{suffix}.json")
    json.dump({**meta, "found": found}, open(out2, "w"))
    print(f"-> {out2}")


if __name__ == "__main__":
    main()
