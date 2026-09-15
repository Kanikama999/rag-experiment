"""
「冒頭だけを見て公平にBM25を計算する」位置ブーストの検証。

背景（ユーザーの指摘）:
現行の span_first 位置ブースト（bm25_equalweight_posboost_discourseboost）は、
「文書全体でのBM25（tf/文書長で正規化）」に対する**固定の加点ボーナス**でしかない。
短い文書は本当に関連していても、クエリの語のうち一致する絶対数が少ないため基礎スコアが
低く出る。長い文書は逆にtermの絶対数が多いため、大して関連していなくてもスコアが積み
上がる。固定ボーナスでは、この基礎スコアの差（文書長に起因する不公平）を埋めきれない。

提案: 文書の**冒頭lead_n語だけ**を文書として扱い、文書長もmin(実長,lead_n)として
正規化したBM25を計算する。これなら短い文書（≒全体が冒頭）と長い文書の冒頭部分が、
対等な土俵で比較される。長い文書の後半（無関係かもしれない内容）は一切スコアに寄与しない。

実装は bm25f_prepare/bm25f_score と同じ2段構成（候補プール取得 → _mtermvectorsで
positions=Trueのtermvector取得 → Python側でスコア計算）。title/headingsは短いフィールド
なので制限せず通常のBM25、bodyだけ冒頭制限を適用する。

事前検証（grade>=2 未発見文書25件サンプル）:
    position boost有効(span_end=200,span_boost=30)の候補プールと照合したところ、
    27.8%(140/503件)は既に拾えていた。残り363件のうちサンプル25件をk=5000まで広げて
    確認すると、8件(32%)は深い順位（1678〜4928位）で存在（=基礎スコアが低いだけ）、
    17件(68%)はk=5000でも一切マッチしない（=候補生成自体が届かない）。
    本実験は前者（基礎スコアが低いだけの文書）を狙う。後者は候補プール自体を
    広げないと解決しない別問題。

条件:
    champion_full : 現行championと同じ（body全体のBM25 + span_first定数ボーナス）
    leadonly_N200 : body を冒頭200語制限のBM25に置き換え（位置ブーストなし、
                    それ自体が「冒頭を見る」スコアなので不要）
    leadonly_N100 : 同、冒頭100語
candidate_k は championのRETRIEVE_K=1000と揃えて比較する（拾えなかった17/25=68%の
問題は解決しないが、まずはメカニズムの効果だけを切り分ける）。

使い方:
    python3 evaluate_leadonly_bm25.py [LIMIT]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytrec_eval

from retriever import (client, INDEX, bm25_equalweight_posboost_discourseboost,
                        analyze_terms, rrf_fuse, _bm25f_avgdl, _bm25f_total_docs,
                        _bm25f_combined_df)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK, RETRIEVE_K, QUERY_REPEAT = 1000, 1000, 5
SPAN_END, SPAN_BOOST = 200, 30.0
BM25_K1 = 0.9
B_TITLE, B_HEADINGS, B_BODY = 0.75, 0.75, 0.75
N_WORKERS = 6
SUBSET_STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LEAD_NS = [100, 200]

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


def leadbody_prepare(title_text, headings_text, body_text, candidate_k):
    should = []
    if title_text.strip():
        should.append({"match": {"title": {"query": title_text}}})
    if headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text}}})
    if body_text.strip():
        should.append({"match": {"body": {"query": body_text}}})
    if not should:
        return None
    res = client.search(index=INDEX, body={
        "size": candidate_k, "_source": False, "query": {"bool": {"should": should}}})
    candidates = [h["_id"] for h in res["hits"]["hits"]]
    if not candidates:
        return None

    query_terms, seen = [], set()
    for field, text in (("title", title_text), ("headings", headings_text), ("body", body_text)):
        if not text.strip():
            continue
        for t in analyze_terms(text, field=field):
            if t not in seen:
                seen.add(t)
                query_terms.append(t)

    avgdl = {f: _bm25f_avgdl(f) for f in ("title", "headings", "body")}
    N = _bm25f_total_docs()
    idf = {t: (lambda df: math.log(1 + (N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0)
           (_bm25f_combined_df(t)) for t in query_terms}

    tv = client.mtermvectors(index=INDEX, body={
        "ids": candidates,
        "parameters": {"fields": ["title", "headings", "body"], "positions": True,
                        "term_statistics": False, "field_statistics": False}})
    docs = {}
    for doc in tv.get("docs", []):
        docid = doc.get("_id")
        tvs = doc.get("term_vectors", {})
        ti = tvs.get("title", {}).get("terms", {})
        hi = tvs.get("headings", {}).get("terms", {})
        bi = tvs.get("body", {}).get("terms", {})
        docs[docid] = {
            "title_terms": ti, "title_len": sum(v["term_freq"] for v in ti.values()),
            "headings_terms": hi, "headings_len": sum(v["term_freq"] for v in hi.values()),
            "body_terms": bi, "body_len": sum(v["term_freq"] for v in bi.values()),
        }
    return {"candidates": candidates, "query_terms": query_terms, "idf": idf,
            "avgdl": avgdl, "docs": docs}


def score_full_vs_lead(raw, k, lead_n):
    """同じ生データから、body全体版とbody冒頭lead_n語版の両方を計算する
    （mtermvectorsの再取得なしで比較できるようにするため）。"""
    if raw is None:
        return [], []
    idf, avgdl, docs = raw["idf"], raw["avgdl"], raw["docs"]

    def field_score(t, ti, info, dl, adl, b):
        tf = info["term_freq"]
        B = (1 - b) + b * (dl / adl) if adl else 1.0
        return idf.get(t, 0.0) * tf * (BM25_K1 + 1) / (tf + BM25_K1 * B)

    full_scores, lead_scores = {}, {}
    for docid, doc in docs.items():
        sf = sl = 0.0
        for t in raw["query_terms"]:
            ti = idf.get(t, 0.0)
            if ti <= 0:
                continue
            info = doc["title_terms"].get(t)
            if info:
                s = field_score(t, ti, info, doc["title_len"], avgdl["title"], B_TITLE)
                sf += s; sl += s
            info = doc["headings_terms"].get(t)
            if info:
                s = field_score(t, ti, info, doc["headings_len"], avgdl["headings"], B_HEADINGS)
                sf += s; sl += s
            info = doc["body_terms"].get(t)
            if info:
                dl = doc["body_len"]
                tf = info["term_freq"]
                # 通常版: 文書全体の長さで正規化（bm25f_score()と同じ形。idfも同じものを使う
                # ので、lead版との差は「tf/dlをどこまでの範囲で数えるか」だけに絞られる）。
                Bf = (1 - B_BODY) + B_BODY * (dl / avgdl["body"]) if avgdl["body"] else 1.0
                sf += ti * tf * (BM25_K1 + 1) / (tf + BM25_K1 * Bf)
                tokens = info.get("tokens", [])
                tf_lead = sum(1 for tk in tokens if tk.get("position", 10**9) < lead_n)
                if tf_lead > 0:
                    dl_lead = min(dl, lead_n)
                    Bl = (1 - B_BODY) + B_BODY * (dl_lead / lead_n)
                    sl += ti * tf_lead * (BM25_K1 + 1) / (tf_lead + BM25_K1 * Bl)
        full_scores[docid] = sf
        lead_scores[docid] = sl
    full = sorted(full_scores.items(), key=lambda x: -x[1])[:k]
    lead = sorted(lead_scores.items(), key=lambda x: -x[1])[:k]
    return full, lead


def run_topic(qid, narrative, entry, lead_ns, candidate_k):
    rep = " ".join([narrative] * QUERY_REPEAT)
    out = {f"lead{n}": [] for n in lead_ns}
    out["fullbody"] = []
    for i, dq in enumerate(entry["decomposed_queries"]):
        pdoc = entry["query2doc_docs_structured"][i]
        if not (pdoc and pdoc.get("body")):
            continue
        hd = " ".join(pdoc["headings"])
        t = f"{rep} {dq} {pdoc['title']}"
        h = f"{rep} {dq} {hd}"
        b = f"{rep} {dq} {pdoc['body']}"
        raw = leadbody_prepare(t, h, b, candidate_k)
        for n in lead_ns:
            full, lead = score_full_vs_lead(raw, candidate_k, n)
            out[f"lead{n}"].append(lead)
            if n == lead_ns[0]:
                out["fullbody"].append(full)
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

    print(f"{len(qids)}トピック  candidate_k={RETRIEVE_K}  lead_ns={LEAD_NS}")
    t0 = time.time()
    rankings = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_topic, q, queries[q], webstyle[q], LEAD_NS, RETRIEVE_K) for q in qids]
        for n, f in enumerate(futs, 1):
            qid, out = f.result()
            rankings[qid] = out
            if n % 10 == 0 or n == len(qids):
                el = time.time() - t0
                print(f"  {n}/{len(qids)}  {el/60:.1f}分  残り約{el/n*(len(qids)-n)/60:.0f}分", flush=True)

    METHODS = ["fullbody"] + [f"lead{n}" for n in LEAD_NS]
    runs = {m: {} for m in METHODS}
    for qid, out in rankings.items():
        for m in METHODS:
            lists = out.get(m, [])
            fused = rrf_fuse(lists, top_n=TOPK) if lists else []
            runs[m][qid] = {d: float(s) for d, s in fused}

    results = {}
    for qs in QREL_SETS:
        for m in METHODS:
            results.setdefault(qs, {})[m] = evaluate(runs[m], qrels[qs], qids)

    suffix = "" if SUBSET_STRIDE == 1 else f"_stride{SUBSET_STRIDE}_n{len(qids)}"
    out_path = os.path.join(RAG_DIR, f"leadonly_bm25_result{suffix}.json")
    json.dump({"n_topics": len(qids), "candidate_k": RETRIEVE_K, "lead_ns": LEAD_NS,
                "results": results}, open(out_path, "w"), indent=2)

    for qs in QREL_SETS:
        print(f"\n=== {qs} ===")
        print(f"{'手法':<12}"+"".join(f"{k:>10}" for k in ("R@100","R@1000","nDCG@10","P@100")))
        for m in METHODS:
            a = results[qs][m]
            if a: print(f"{m:<12}"+"".join(f"{a[k]:>10.4f}" for k in METRIC_KEYS))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
