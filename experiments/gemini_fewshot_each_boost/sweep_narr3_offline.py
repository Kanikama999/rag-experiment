"""
narr3（recall優先の現行最良）を土台に、既存キャッシュだけで（追加検索ゼロで）
より強い組み合わせが無いか総当たりする。

使う素材（すべて既存）:
  long  : narr3 のランキング（_cache_recall_shrink.json.gz）固定
  short : Subquery単独の全バリエーション
            - 位置ブーストなし（material_fusion pb0）  ← 現行narr3|fuse3が使っているもの
            - 位置ブーストあり se∈{25,50,100,150,200,300}（span_longshort, sb=15）
            - 位置ブーストあり se∈{400,600,800}（recall_max_short_ext, sb=15）
            - 位置ブーストあり se=100（material_fusion pb1）
  third : 拡張語リスト（termlist、位置ブーストあり pb1 / なし pb0）を第3の腕として on/off

long重み [1,3,5,8] × short候補 × termlist on/off をすべてオフラインでRRF融合・評価する。
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
SPAN = load("_cache_span_longshort.json.gz")
EXT = load("_cache_recall_max_short_ext.json.gz")

LONG = RS["narr3"]

SHORT_CANDS = {
    "sub_pb0(現行)": MAT["0"],
    "sub_pb1_se100": MAT["1"],
}
for se in (25, 50, 100, 150, 200, 300):
    SHORT_CANDS[f"sub_se{se}"] = {"__flat__": SPAN[f"short_{se}"]}
for se in (400, 600, 800):
    SHORT_CANDS[f"sub_se{se}"] = {"__flat__": EXT[str(se)]}

def get_short_ids(cand, qid):
    if "__flat__" in cand:
        return cand["__flat__"].get(qid, {})
    return {i: d.get("subquery", []) for i, d in cand.get(qid, {}).items()}

TERMLIST_CANDS = {"なし": None, "termlist_pb1": MAT["1"], "termlist_pb0": MAT["0"]}

def get_termlist_ids(cand, qid):
    if cand is None:
        return {}
    return {i: d.get("termlist", []) for i, d in cand.get(qid, {}).items()}

qrels = {n: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{n}.txt")) for n in QREL_SETS}
qids = sorted(LONG.keys())

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
for lw in (1, 3, 5, 8):
    for sname, scand in SHORT_CANDS.items():
        for tname, tcand in TERMLIST_CANDS.items():
            run = {}
            for qid in qids:
                lists, ws = [], []
                for i, ids in LONG.get(qid, {}).items():
                    lists.append([(d, 1.0) for d in ids]); ws.append(float(lw))
                sids = get_short_ids(scand, qid)
                for i, ids in sids.items():
                    lists.append([(d, 1.0) for d in ids]); ws.append(1.0)
                tids = get_termlist_ids(tcand, qid)
                for i, ids in tids.items():
                    if ids:
                        lists.append([(d, 1.0) for d in ids]); ws.append(1.0)
                run[qid] = {dd: float(s) for dd, s in
                            (rrf_fuse(lists, top_n=TOPK, weights=ws) if lists else [])}
            key = f"narr3x{lw}+{sname}+{tname}"
            rec = {}
            for qs in QREL_SETS:
                rec[qs] = evaluate(run, qrels[qs])
            results[key] = rec

out = os.path.join(RAG_DIR, "sweep_narr3_offline_result.json")
json.dump({"n_topics": len(qids), "results": results}, open(out, "w"), indent=2)

for qs in QREL_SETS:
    rows = [(k, v) for k, v in results.items() if v.get(qs)]
    print(f"\n=== {qs}  recall@1000 上位15 ===")
    print(f"{'条件':<40}{'R@100':>8}{'R@1000':>9}{'nDCG@10':>9}{'P@100':>8}")
    for k, v in sorted(rows, key=lambda x: -x[1][qs]["recall_1000"])[:15]:
        a = v[qs]
        print(f"{k:<40}{a['recall_100']:>8.4f}{a['recall_1000']:>9.4f}{a['ndcg_cut_10']:>9.4f}{a['P_100']:>8.4f}")
print(f"\n-> {out}")
