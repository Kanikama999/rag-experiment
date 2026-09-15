"""
sweep_narr3_offline.py の勝ちパターン（long + sub_pb0 + termlist_pb0）を土台に、
①narr2でも同じレシピを試す ②long重みを5〜8の間で細かく振る。すべて既存キャッシュのみ。
"""
import gzip, json, os
import pytrec_eval
from retriever import rrf_fuse

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
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

SUBQ = MAT["0"]      # sub_pb0
TERML = MAT["0"]     # termlist_pb0

qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}

def evaluate(run, qr, qids):
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
for narr in ("narr2", "narr3"):
    LONG = RS[narr]
    qids = sorted(LONG.keys())
    for lw in (4, 5, 6, 7, 8):
        run = {}
        for qid in qids:
            lists, ws = [], []
            for i, ids in LONG.get(qid, {}).items():
                lists.append([(d, 1.0) for d in ids]); ws.append(float(lw))
            for i, d in SUBQ.get(qid, {}).items():
                if "subquery" in d:
                    lists.append([(x, 1.0) for x in d["subquery"]]); ws.append(1.0)
            for i, d in TERML.get(qid, {}).items():
                if "termlist" in d:
                    lists.append([(x, 1.0) for x in d["termlist"]]); ws.append(1.0)
            run[qid] = {dd: float(s) for dd, s in
                        (rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else [])}
        key = f"{narr}x{lw}+sub_pb0+termlist_pb0"
        rec = {}
        for qs in QREL_SETS:
            rec[qs] = evaluate(run, qrels[qs], qids)
        results[key] = rec

out = os.path.join(RAG_DIR, "sweep_narr23_fine_result.json")
json.dump({"results": results}, open(out, "w"), indent=2)

for qs in QREL_SETS:
    rows = [(k, v) for k, v in results.items() if v.get(qs)]
    print(f"\n=== {qs} ===")
    print(f"{'条件':<32}{'R@100':>8}{'R@1000':>9}{'nDCG@10':>9}{'P@100':>8}")
    for k, v in sorted(rows, key=lambda x: -x[1][qs]["recall_1000"]):
        a = v[qs]
        print(f"{k:<32}{a['recall_100']:>8.4f}{a['recall_1000']:>9.4f}{a['ndcg_cut_10']:>9.4f}{a['P_100']:>8.4f}")
print(f"\n-> {out}")
