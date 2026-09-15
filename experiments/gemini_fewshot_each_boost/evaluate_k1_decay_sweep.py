"""
「出現回数が増えるほど、後の1回1回の影響を弱くする」を BM25 の k1 パラメータの
スイープとして検証する。

背景:
evaluate_coverage_bm25.py（TFを完全に無視し「出現したか0/1」だけで加点）は35トピックで
champion に全指標・両qrelsで惨敗した。「本当に濃く関連している文書と、ただ話題がかすった
だけの文書を区別できなくなった」のが敗因だった。

指摘: TFを捨てるのではなく、TFが増えるほど後の1回の影響を減衰させるべき。これは
BM25のk1パラメータが既にやっていること。k1→0 の極限が「1回でも出ればidf、それ以上は
変わらない」＝案1そのものと一致する（実測: tf=1で1.000、tf=167でもk1=0.1なら1.099まで
しか伸びない。k1=0.9(現行)なら1.890まで伸びる）。champion(k1=0.9)と案1(k1≈0)の間に
落とし所があるはず。

設計（2026-09-11 メモリ問題を受けて再設計）:
bm25f_prepare()の_mtermvectorsは、フィールドを指定すると**クエリ語だけでなく文書内の
全ての語**のterm vectorを返す（1文書で1336語返ってきた実測あり）。これを105トピック
全部ぶんメモリに保持したままk1をスイープしようとした結果、生データだけで数GB規模に
なりOOM Killerに落とされた（チェックポイントJSONが446MBで壊れていたことから推定）。

そこで**トピックごとに「取得→即座に全k1でスコア化→生データは破棄」のストリーミング
構成**に変更。保持するのは最終的な順位（qid→{docid:score}の軽い辞書、k1の数だけ）
だけで、重いterm vectorデータを溜め込まない。

使い方:
    python3 evaluate_k1_decay_sweep.py [SUBSET_STRIDE]
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import bm25f_prepare, bm25f_score, rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
B_TITLE, B_HEADINGS, B_BODY = 0.6, 0.2, 0.2
N_WORKERS = 2   # 2026-09-11: メモリ逼迫でOOM Killerに2回落とされたため保守的に設定
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CKPT = os.path.join(RAG_DIR, "_cache_k1sweep_runs.json")

K1_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.2, 1.5, 2.0]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


def load_qrels(p):
    q = {}
    for line in open(p):
        s = line.split()
        if len(s) == 4 and not line.startswith("#"):
            q.setdefault(s[0], {})[s[2]] = int(s[3])
    return q


def load_queries(p):
    q = {}
    for line in open(p):
        line = line.strip()
        if line:
            o = json.loads(line)
            q[o["id"]] = o["title"]
    return q


def topic_rankings_all_k1(qid, narrative, entry):
    """候補プール・idf・tf/文書長を1回だけ取得し、その場で全k1のランキングを作って
    返す（生データはこの関数のスコープを抜けると同時にGC対象になる）。"""
    rep = " ".join([narrative] * QUERY_REPEAT)
    raws = []
    for i, dq in enumerate(entry["decomposed_queries"]):
        pdoc = entry["query2doc_docs_structured"][i]
        if not (pdoc and pdoc.get("body")):
            continue
        t = f"{rep} {dq} {pdoc['title']}"
        h = f"{rep} {dq} {' '.join(pdoc['headings'])}"
        b = f"{rep} {dq} {pdoc['body']}"
        for attempt in range(3):
            try:
                raw = bm25f_prepare(t, h, b, title_weight=1.0, headings_weight=1.0, body_weight=1.0,
                                     candidate_k=RETRIEVE_K, rerank_n=RETRIEVE_K)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        if raw is not None:
            raws.append(raw)

    out = {}
    for k1 in K1_VALUES:
        lists = [bm25f_score(raw, k=RETRIEVE_K, title_weight=1.0, headings_weight=1.0,
                               body_weight=1.0, b_title=B_TITLE, b_headings=B_HEADINGS,
                               b_body=B_BODY, bm25_k1=k1) for raw in raws]
        lists = [l for l in lists if l]
        fused = rrf_fuse(lists, top_n=TOPK) if lists else []
        out[str(k1)] = {d: float(s) for d, s in fused}
    return qid, out


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
    print(f"{len(qids)}トピック  ストリーミング構成（トピックごとに即スコア化・破棄）")

    runs_by_k1 = {str(k1): {} for k1 in K1_VALUES}
    if os.path.exists(CKPT):
        c = json.load(open(CKPT))
        if c.get("n_topics") == len(qids):
            runs_by_k1 = c["runs_by_k1"]
            done_qids = set(next(iter(runs_by_k1.values())).keys()) if runs_by_k1.get(str(K1_VALUES[0])) else set()
            print(f"チェックポイント復帰: {len(done_qids)}/{len(qids)} トピック済み")

    done_qids = set(runs_by_k1[str(K1_VALUES[0])].keys())
    todo = [q for q in qids if q not in done_qids]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(topic_rankings_all_k1, q, queries[q], webstyle[q]): q for q in todo}
        n_done = 0
        for f in futs:
            qid, out = f.result()
            for k1s, run in out.items():
                runs_by_k1[k1s][qid] = run
            n_done += 1
            if n_done % 5 == 0 or n_done == len(todo):
                el = time.time() - t0
                print(f"  {len(done_qids)+n_done}/{len(qids)}  今回{n_done}/{len(todo)}  "
                      f"{el/60:.1f}分  残り約{el/max(n_done,1)*(len(todo)-n_done)/60:.0f}分", flush=True)
                json.dump({"n_topics": len(qids), "runs_by_k1": runs_by_k1}, open(CKPT, "w"))

    results = {}
    for qs in QREL_SETS:
        for k1 in K1_VALUES:
            results.setdefault(qs, {})[str(k1)] = evaluate(runs_by_k1[str(k1)], qrels[qs], qids)

    suffix = "" if SUBSET_STRIDE == 1 else f"_stride{SUBSET_STRIDE}_n{len(qids)}"
    out_path = os.path.join(RAG_DIR, f"k1_decay_sweep_result{suffix}.json")
    json.dump({"n_topics": len(qids), "k1_values": K1_VALUES, "results": results},
              open(out_path, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print(f"{'k1':>6}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for k1 in K1_VALUES:
            a = results[qs][str(k1)]
            if a: print(f"{k1:>6}"+"".join(f"{a[m]:>10.4f}" for m in METRIC_KEYS))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
