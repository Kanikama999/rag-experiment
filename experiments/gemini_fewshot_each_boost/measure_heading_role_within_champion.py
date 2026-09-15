"""
見出し役割シグナルが「チャンピオンのランキングに対して追加情報を持つか」の検証。

measure_heading_role_conditional/_alignment.py では負例を『narrativeそのままのBM25
上位』から取っていた。そこでは aligned の lift が 4.90 に達したが、実際に
チャンピオン（構造化疑似文書+fielded+posboost+RRF）へ足すと nDCG@10 は
単調に悪化した（evaluate_headingrole_rerank.py）。

仮説: あの lift はチャンピオンが既に captured 済みの情報だった。つまり負例が
弱すぎた。チャンピオンの上位はそもそも「クエリ話題の見出しを持つ文書」で
占められているので、その中では役割シグナルが relevant/nonrelevant を分けない。

ここでは負例をチャンピオン上位N件の非relevant文書に置き換えて同じ lift を測り直す。
lift が 1.0 付近まで落ちれば仮説が確認され、null result の理由が確定する。

使い方:
    python measure_heading_role_within_champion.py [--rerank-n N]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from retriever import analyze_terms, heading_roles
from measure_heading_role_distribution import load_qrels, fetch_headings
from evaluate_headingrole_rerank import (doc_features, FEATURES, ROLE_TERMS_TEXT,
                                         RUN_CACHE, WEBSTYLE_FILE, DATA_DIR, RAG_DIR)

OUT_FILE = os.path.join(RAG_DIR, "heading_role_within_champion.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-n", type=int, default=100)
    args = ap.parse_args()

    if not os.path.exists(RUN_CACHE):
        raise SystemExit("チャンピオンrunのキャッシュが無い。先に "
                         "evaluate_headingrole_rerank.py を実行すること。")
    runs = json.load(open(RUN_CACHE))
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    qrels = load_qrels(os.path.join(DATA_DIR, "keystone_qrels_consensus.txt"))
    role_terms = set(analyze_terms(ROLE_TERMS_TEXT, field="headings"))

    qids = sorted(q for q in runs if q in qrels and q in webstyle)
    print(f"{len(qids)} トピック / チャンピオン上位{args.rerank_n}件を対象")

    counts = {"pos": Counter(), "neg": Counter()}
    totals = {"pos": 0, "neg": 0}
    per_doc = {}
    t0 = time.time()
    for i, qid in enumerate(qids, 1):
        sq_units = []
        for dq in webstyle[qid]["decomposed_queries"]:
            terms = set(analyze_terms(dq, field="headings")) - role_terms
            if terms:
                sq_units.append((terms, heading_roles(dq)))
        top = [d for d, _ in runs[qid][:args.rerank_n]]
        heads = fetch_headings(top)
        per_doc[qid] = {}
        for d in top:
            feat = doc_features(heads.get(d, ""), sq_units, role_terms)
            per_doc[qid][d] = feat
            group = "pos" if qrels[qid].get(d, 0) >= 1 else "neg"
            totals[group] += 1
            for k, v in feat.items():
                if v:
                    counts[group][k] += 1
        print(f"\r  {i}/{len(qids)} ({time.time()-t0:.0f}s)", end="", flush=True)
    print()

    print(f"\n=== チャンピオン上位{args.rerank_n}件内での lift   "
          f"pos n={totals['pos']}  neg n={totals['neg']} ===")
    print("signal".ljust(12) + "relevant".rjust(10) + "nonrel".rjust(10) + "  lift")
    result = {}
    for k in FEATURES:
        p = 100 * counts["pos"][k] / max(totals["pos"], 1)
        n = 100 * counts["neg"][k] / max(totals["neg"], 1)
        lift = (p / n) if n > 0 else float("nan")
        result[k] = {"rel_pct": round(p, 2), "nonrel_pct": round(n, 2), "lift": round(lift, 3)}
        print(f"{k.ljust(12)}{p:10.2f}{n:10.2f}{lift:8.3f}")

    with open(OUT_FILE, "w") as f:
        json.dump({"rerank_n": args.rerank_n, "n_topics": len(qids), "totals": totals,
                   "result": result, "per_doc": per_doc}, f)
    print(f"\n-> {OUT_FILE}")


if __name__ == "__main__":
    main()
