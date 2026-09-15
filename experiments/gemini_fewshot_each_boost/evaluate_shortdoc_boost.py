"""
narr3x6+sub+termlist（現行recall記録）の候補プールに、文書長ベースの「短さボーナス」を
後乗せしたらrecallが伸びるか検証する。

背景:
インデックスのBM25設定は b=0.4（Lucene標準0.75より弱い長さ正規化）で、長文書に有利。
実文書分析（§ titleありなのに拾えなかった503件）でも短文書（<800語）の比率が未発見群で
2.2倍高いことが分かっている。bを直接変えるとインデックス全体（82GB・1096万件）に
影響するため触らず、検索後の候補プールに後乗せで文書長ボーナスを掛ける。

    new_score = rrf_score × (avgdl / max(doc_len, 1))^alpha

alpha=0 は現行と同一（検算用）。alpha を振って効果を見る。
doc_len は bm25f_prepare と同じ方法（_mtermvectors のterm_freq合計）で取得する。

使い方:
    python3 evaluate_shortdoc_boost.py [LIMIT]
"""
import gzip, json, os, sys, time
import pytrec_eval
from retriever import client, INDEX, rrf_fuse, _bm25f_avgdl

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
ALPHAS = [0, 0.25, 0.5, 0.75, 1.0, 1.5]
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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

def load(fn):
    with gzip.open(fn, "rt") as f:
        return json.load(f)["rankings"]

RS = load("_cache_recall_shrink.json.gz")
MAT = load("_cache_material_rankings.json.gz")
LONG = RS["narr3"]
SUBQ = MAT["0"]
TERML = MAT["0"]

qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}
qids = sorted(LONG.keys())
if LIMIT:
    qids = qids[:LIMIT]

# ① まず narr3x6+sub+termlist を再現（既存の勝ちレシピ）
print("融合ランキングを再構築中...")
runs = {}
for qid in qids:
    lists, ws = [], []
    for i, ids in LONG.get(qid, {}).items():
        lists.append([(d, 1.0) for d in ids]); ws.append(6.0)
    for i, d in SUBQ.get(qid, {}).items():
        if "subquery" in d:
            lists.append([(x, 1.0) for x in d["subquery"]]); ws.append(1.0)
    for i, d in TERML.get(qid, {}).items():
        if "termlist" in d:
            lists.append([(x, 1.0) for x in d["termlist"]]); ws.append(1.0)
    runs[qid] = rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else []

# ② 候補プール全文書のbody長を取得
all_ids = sorted({d for r in runs.values() for d, _ in r})
print(f"候補文書数（重複除去後）: {len(all_ids)}   body長を取得中...")
doc_len = {}
t0 = time.time()
for i in range(0, len(all_ids), 500):
    chunk = all_ids[i:i+500]
    res = client.mtermvectors(index=INDEX, body={
        "ids": chunk,
        "parameters": {"fields": ["body"], "term_statistics": False, "field_statistics": False,
                        "positions": False, "offsets": False, "payloads": False}})
    for doc in res.get("docs", []):
        docid = doc.get("_id")
        terms = doc.get("term_vectors", {}).get("body", {}).get("terms", {})
        doc_len[docid] = sum(v["term_freq"] for v in terms.values()) if terms else 0
    if (i // 500) % 20 == 0:
        print(f"  {i}/{len(all_ids)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"-> body長取得完了（{(time.time()-t0)/60:.1f}分）")

avgdl = _bm25f_avgdl("body")
print(f"avgdl(body) = {avgdl:.1f}")

def evaluate(run, qr):
    tg = [q for q in qids if q in qr and run.get(q)]
    if not tg: return None
    ev = pytrec_eval.RelevanceEvaluator({q: qr[q] for q in tg}, METRICS)
    res = ev.evaluate({q: run[q] for q in tg})
    if not res: return None
    n = len(res)
    a = {m: sum(r[m] for r in res.values()) / n for m in METRIC_KEYS}
    a["n"] = n
    return a

results = {}
for alpha in ALPHAS:
    run = {}
    for qid, ranked in runs.items():
        rescored = []
        for d, s in ranked:
            dl = max(doc_len.get(d, avgdl), 1)
            boost = (avgdl / dl) ** alpha if alpha else 1.0
            rescored.append((d, s * boost))
        rescored.sort(key=lambda x: -x[1])
        run[qid] = {d: float(s) for d, s in rescored[:TOPK]}
    rec = {}
    for qs in QREL_SETS:
        rec[qs] = evaluate(run, qrels[qs])
    results[f"alpha{alpha}"] = rec

out = os.path.join(RAG_DIR, "shortdoc_boost_result.json")
json.dump({"n_topics": len(qids), "avgdl_body": avgdl, "results": results}, open(out, "w"), indent=2)

for qs in QREL_SETS:
    print(f"\n=== {qs} ===")
    print(f"{'alpha':<10}{'R@100':>8}{'R@1000':>9}{'nDCG@10':>9}{'P@100':>8}")
    for a_key in [f"alpha{a}" for a in ALPHAS]:
        v = results[a_key].get(qs)
        if v: print(f"{a_key:<10}{v['recall_100']:>8.4f}{v['recall_1000']:>9.4f}{v['ndcg_cut_10']:>9.4f}{v['P_100']:>8.4f}")
print(f"\n-> {out}")
