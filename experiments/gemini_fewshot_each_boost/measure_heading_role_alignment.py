"""
見出し役割仮説の検証その3: 「Subqueryの意図」と「見出しの役割」の対応（alignment）。

その2（measure_heading_role_conditional.py）では、話題一致した見出しがどの役割かで
relevant/nonrelevantの分離度が変わることを確認した。ただしそこでは役割を
クエリと無関係に集計しており、「definitionを聞いていないクエリでもdefinition見出しが
効いている」だけの可能性が残る。

ここでは同じ役割語彙をSubquery側にも当てて意図ラベルを付け、
    aligned : 話題一致見出しの役割が、そのSubqueryの意図と一致
    crossed : 話題一致見出しに役割はあるが、意図とは不一致
    plain   : 話題一致見出しはあるが役割語を持たない
の3条件でrelevant/nonrelevantの割合を比較する。alignedがcrossedを明確に上回るなら
「見出しの役割とクエリの意図を対応付ける」という設計に意味があり、差が無いなら
役割は単なる文書品質の指標（意図とは無関係）だったことになる。

使い方:
    python measure_heading_role_alignment.py [--topics coverage|consensus] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from retriever import bm25_body, analyze_terms
from measure_heading_role_distribution import ROLE_RE, ROLE_PATTERNS, load_qrels, fetch_headings
from measure_heading_role_conditional import (doc_heading_units, load_queries, QUERIES_FILE,
                                              DECOMPOSED_FILE, QRELS, POS_CAP, NEG_CAP,
                                              NEG_POOL_K, MIN_OVERLAP, SEED, RAG_DIR)

ROLE_TERMS_TEXT = (
    "what is are definition defined meaning mean how it works explained detail details "
    "overview about history origin origins etymology background faq frequently asked "
    "questions summary conclusion key takeaways short bottom line recap warning caution "
    "risk risks side effects precautions important note vs versus difference between "
    "compared comparison steps guide instructions tutorial why cause causes reason reasons")


def intent_roles(text):
    """Subquery文字列に同じ役割語彙を当てて意図ラベル集合を返す。"""
    return {r for r, rx in ROLE_RE.items() if rx.search(text)}


def doc_alignment_signals(headings_text, sq_units, role_terms):
    """1文書について aligned/crossed/plain のフラグと、意図別alignedフラグを返す。
    sq_units = [(subqueryの内容語set, subqueryの意図role set), ...]"""
    units = doc_heading_units(headings_text)
    sig = Counter()
    for raw, stems in units:
        content = stems - role_terms
        hroles = {r for r, rx in ROLE_RE.items() if rx.search(raw)}
        for terms, intents in sq_units:
            if len(content & terms) < MIN_OVERLAP:
                continue
            sig["any"] = 1
            if not hroles:
                sig["plain"] = 1
                continue
            if hroles & intents:
                sig["aligned"] = 1
                for r in hroles & intents:
                    sig[f"aligned:{r}"] = 1
            else:
                sig["crossed"] = 1
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="coverage", choices=["coverage", "consensus"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(SEED)
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS[args.topics])
    with open(DECOMPOSED_FILE) as f:
        decomposed = json.load(f)["results"]
    role_terms = set(analyze_terms(ROLE_TERMS_TEXT, field="headings"))

    topics = sorted(t for t in qrels if t in queries and t in decomposed)
    if args.limit:
        topics = topics[:args.limit]

    intent_hist = Counter()
    n_sq = 0
    counts = {"pos": Counter(), "neg": Counter()}
    totals = {"pos": 0, "neg": 0}

    for i, qid in enumerate(topics, 1):
        sqs = decomposed[qid].get("decomposed_queries") or []
        sq_units = []
        for s in sqs:
            terms = set(analyze_terms(s, field="headings")) - role_terms
            if not terms:
                continue
            intents = intent_roles(s)
            n_sq += 1
            intent_hist.update(intents or {"(none)"})
            sq_units.append((terms, intents))
        if not sq_units:
            continue

        pos_ids = [d for d, g in qrels[qid].items() if g >= 1]
        judged_neg = [d for d, g in qrels[qid].items() if g == 0]
        if len(pos_ids) > POS_CAP:
            pos_ids = rng.sample(pos_ids, POS_CAP)
        neg_pool = list(judged_neg)
        if len(neg_pool) < NEG_CAP:
            neg_pool += [d for d, _ in bm25_body(queries[qid], k=NEG_POOL_K) if d not in qrels[qid]]
        neg_ids = neg_pool[:NEG_CAP * 3]
        if len(neg_ids) > NEG_CAP:
            neg_ids = rng.sample(neg_ids, NEG_CAP)

        heads = fetch_headings(pos_ids + neg_ids)
        for group, ids in (("pos", pos_ids), ("neg", neg_ids)):
            for d in ids:
                if d not in heads:
                    continue
                totals[group] += 1
                for k, v in doc_alignment_signals(heads[d], sq_units, role_terms).items():
                    counts[group][k] += v
        print(f"  [{i}/{len(topics)}] {qid}", flush=True)

    print(f"\n=== Subquery意図の内訳 (n={n_sq}) ===")
    for r, c in intent_hist.most_common():
        print(f"  {r.ljust(12)}{c:6d}  ({100*c/max(n_sq,1):5.1f}%)")

    rows = ["any", "plain", "crossed", "aligned"] + [f"aligned:{r}" for r in ROLE_PATTERNS]
    print(f"\n=== 該当見出しを持つ文書の割合(%)   pos n={totals['pos']}  neg n={totals['neg']} ===")
    print("signal".ljust(20) + "relevant".rjust(10) + "nonrel".rjust(10) + "  lift")
    result = {}
    for r in rows:
        p = 100 * counts["pos"][r] / max(totals["pos"], 1)
        n = 100 * counts["neg"][r] / max(totals["neg"], 1)
        lift = (p / n) if n > 0 else float("nan")
        result[r] = {"rel_pct": round(p, 2), "nonrel_pct": round(n, 2), "lift": round(lift, 3)}
        print(f"{r.ljust(20)}{p:10.2f}{n:10.2f}{lift:8.3f}")

    out = os.path.join(RAG_DIR, f"heading_role_alignment_{args.topics}.json")
    with open(out, "w") as f:
        json.dump({"topics": args.topics, "n_topics": len(topics), "n_subqueries": n_sq,
                   "intent_hist": dict(intent_hist), "totals": totals, "result": result},
                  f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
