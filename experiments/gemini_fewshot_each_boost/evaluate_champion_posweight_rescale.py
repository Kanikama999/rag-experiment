"""
championのposition boost（span_first、先頭span_end語以内に出現すれば一律ボーナス）を、
「位置そのものに応じて重みを変える」方式に差し替えて検証する。

背景（explain()で確認した現状の正体）:
    score_posboost = boost × idf(span_terms) × [freq_in_window / (freq_in_window + k1・B)]
"先頭100語以内に出た回数"をBM25飽和にかけているだけで、**位置の早さそのものは
見ていない**（先頭5語目の出現も99語目の出現も、回数としては同じ1回）。

提案: 出現位置xそのものから重みを計算し、合算する。
    weight(x) = 100 - 7・sqrt(x - 1)
例: 位置5の出現 → 100-7√4=86。位置18の出現 → 100-7√17≈71.1。早いほど高得点、
かつBM25のような飽和ではなく単純合算（複数回早く出るほど積み上がる）。

候補生成はchampion（bm25_equalweight_posboost_discourseboost、markers=[]、
span_end=100、span_boost=15）と完全に同一にする（同じ候補プールで比較するため）。
スコアは「championと同じtitle/headings/body通常matchの合計」+
「span_first の代わりに weight(x) の合算 × scale」で計算し直す。

scale（新ボーナスの規模合わせ）は複数振って比較する: {0.1, 0.3, 0.5, 1.0, 2.0}
（weight(x)の生の合計はchampionのspan_first寄与よりずっと大きいスケールになりうる
ため、スケール自体も探索対象にする）。

使い方:
    python3 evaluate_champion_posweight_rescale.py [SUBSET_STRIDE] [LIMIT]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import (client, INDEX, bm25_fielded,
                        bm25_equalweight_posboost_discourseboost,
                        analyze_terms, rrf_fuse)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
SPAN_END, SPAN_BOOST = 100, 15.0   # championと同一（候補生成をそろえるため）
N_WORKERS = 3
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

SCALES = [0.1, 0.3, 0.5, 1.0, 2.0]
METHODS = ["champion"] + [f"posweight_scale{s:g}" for s in SCALES]

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


def pos_weight(x):
    """weight(x) = 100 - 7*sqrt(x-1)。x<1は起きない前提だがmax(x,1)で保険。"""
    x = max(x, 1)
    return 100.0 - 7.0 * math.sqrt(x - 1)


def subquery_rankings(title_text, headings_text, body_text, span_terms, k):
    """championと同じ候補プール（champion本体のスコアも含む）+ 通常match単体スコア
    + 候補文書のbody position情報、を1回のI/Oセットで返す。"""
    # 1) championそのもの（候補プール + championのスコア）
    champ = bm25_equalweight_posboost_discourseboost(
        title_text, headings_text, body_text, span_terms, k=k,
        span_end=SPAN_END, span_boost=SPAN_BOOST, markers=[])
    if not champ:
        return None
    candidates = [d for d, _ in champ]

    # 2) 通常match単体（position boostなしの基礎スコア）
    base = dict(bm25_fielded(title_text, headings_text, body_text, k=k,
                              title_boost=1, headings_boost=1, body_boost=1))

    # 3) 候補文書のbody位置情報（span_termsのみ、positions=Trueで取得）
    tv = client.mtermvectors(index=INDEX, body={
        "ids": candidates,
        "parameters": {"fields": ["body"], "positions": True,
                        "term_statistics": False, "field_statistics": False}})
    span_set = set(span_terms)
    pos_bonus_raw = {}
    for doc in tv.get("docs", []):
        docid = doc.get("_id")
        terms = doc.get("term_vectors", {}).get("body", {}).get("terms", {})
        total = 0.0
        for t in span_set:
            info = terms.get(t)
            if not info:
                continue
            for tk in info.get("tokens", []):
                p = tk.get("position", 10**9)
                if p < SPAN_END:
                    total += pos_weight(p)
        pos_bonus_raw[docid] = total

    return {"champ": dict(champ), "base": base, "pos_bonus_raw": pos_bonus_raw,
            "candidates": candidates}


def scored_lists(raw, k):
    if raw is None:
        return {m: [] for m in METHODS}
    out = {"champion": sorted(raw["champ"].items(), key=lambda x: -x[1])[:k]}
    for s in SCALES:
        scores = {}
        for d in raw["candidates"]:
            scores[d] = raw["base"].get(d, 0.0) + s * raw["pos_bonus_raw"].get(d, 0.0)
        out[f"posweight_scale{s:g}"] = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return out


def run_topic(qid, narrative, entry):
    rep = " ".join([narrative] * QUERY_REPEAT)
    lists = {m: [] for m in METHODS}
    for dq, doc in dq_pairs_structured(entry):
        hd = " ".join(doc["headings"])
        t = f"{rep} {dq} {doc['title']}"
        h = f"{rep} {dq} {hd}"
        b = f"{rep} {dq} {doc['body']}"
        span_terms = analyze_terms(dq)   # championと同じ
        raw = subquery_rankings(t, h, b, span_terms, RETRIEVE_K)
        sl = scored_lists(raw, RETRIEVE_K)
        for m in METHODS:
            if sl[m]:
                lists[m].append(sl[m])
    return qid, {m: {d: float(s) for d, s in (rrf_fuse(lists[m], top_n=TOPK) if lists[m] else [])}
                 for m in METHODS}


def evaluate(run, qrels, qids):
    tg = [q for q in qids if q in qrels and run.get(q)]
    if not tg:
        return None
    ev = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in tg}, METRICS)
    res = ev.evaluate({q: run[q] for q in tg})
    if not res:
        return None
    n = len(res)
    a = {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}
    a["n"] = n
    return a


def main():
    print("読み込み中...")
    webstyle = json.load(open(WEBSTYLE_FILE))["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}
    qids = sorted(q for q in webstyle if q in queries and webstyle[q].get("decomposed_queries"))
    qids = qids[::SUBSET_STRIDE]
    if LIMIT:
        qids = qids[:LIMIT]
    print(f"{len(qids)}トピック  scale候補={SCALES}")

    t0 = time.time()
    runs = {m: {} for m in METHODS}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_topic, q, queries[q], webstyle[q]) for q in qids]
        for n, f in enumerate(futs, 1):
            qid, out = f.result()
            for m in METHODS:
                runs[m][qid] = out[m]
            if n % 10 == 0 or n == len(qids):
                el = time.time() - t0
                print(f"  {n}/{len(qids)}  {el/60:.1f}分  残り約{el/n*(len(qids)-n)/60:.0f}分", flush=True)

    results = {}
    for qs in QREL_SETS:
        for m in METHODS:
            results.setdefault(qs, {})[m] = evaluate(runs[m], qrels[qs], qids)

    suffix = "" if SUBSET_STRIDE == 1 and not LIMIT else f"_stride{SUBSET_STRIDE}_n{len(qids)}"
    out = os.path.join(RAG_DIR, f"champion_posweight_rescale_result{suffix}.json")
    json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print(f"{'手法':<22}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for m in METHODS:
            a = results[qs][m]
            if a: print(f"{m:<22}"+"".join(f"{a[k]:>10.4f}" for k in METRIC_KEYS))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
