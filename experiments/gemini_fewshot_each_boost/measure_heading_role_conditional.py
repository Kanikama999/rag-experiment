"""
見出し役割仮説の本命検証（premise check その2）。

measure_heading_role_distribution.py は「文書が役割見出しを持つか」という
クエリ非依存の事前分布を測った。しかしそこでは relevant と judged-nonrelevant の差が
ほとんど無く、役割の無条件事前確率には識別力が無いことが分かった。

こちらで測るのは条件付きの方、つまり画像の表が本来言っている

    「クエリの話題を扱っている見出し」が、どの役割の見出しか

である。具体的には1本の見出しについて
    (a) Subqueryの内容語がその見出しに何語含まれるか（＝その見出しがクエリ話題か）
    (b) その見出しが持つ役割ラベル
を出し、relevant文書 と 非relevant文書 で

    P(「話題一致かつ役割r」の見出しを持つ | relevant)  vs  同 | nonrelevant

を比較する。any（役割を問わず話題一致見出しを持つ）に対する上乗せがあるかどうかが要点で、
上乗せが無ければ「役割」という軸自体が効かない（＝話題一致だけで説明がつく）ことになる。

使い方:
    python measure_heading_role_conditional.py [--topics coverage|consensus] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

from retriever import client, INDEX, bm25_body, analyze_terms
from measure_heading_role_distribution import (ROLE_RE, ROLE_PATTERNS, load_qrels,
                                               split_headings, fetch_headings)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")
DECOMPOSED_FILE = os.path.join(RAG_DIR, "decomposed_queries.json")
QRELS = {"coverage": os.path.join(DATA_DIR, "keystone_qrels_coverage.txt"),
         "consensus": os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")}

POS_CAP = 150          # 1トピックあたりの正例上限
NEG_CAP = 150          # 1トピックあたりの負例上限
NEG_POOL_K = 400       # 負例を取るBM25ベースラインの深さ
MIN_OVERLAP = 2        # 「その見出しはクエリ話題」と見なす内容語一致数
SEED = 0


def load_queries(path):
    q = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                q[o["id"]] = o["title"]
    return q


def analyze_with_offsets(text):
    """headings全文をenglish analyzerにかけ、(token, start_offset)のリストを返す。"""
    res = client.indices.analyze(index=INDEX, body={"analyzer": "english_search", "text": text})
    return [(t["token"], t["start_offset"]) for t in res["tokens"]]


def doc_heading_units(headings_text):
    """headings全文を『1見出し = (raw文字列, 内容語stemのset)』の単位に分解する。
    改行位置の文字オフセットでトークンをバケツ分けするので、索引側のanalyzerと
    完全に同じトークン化のまま見出し単位に戻せる。"""
    if not headings_text or not headings_text.strip():
        return []
    bounds, pos = [], 0
    for line in headings_text.split("\n"):
        if line.strip():
            bounds.append((pos, pos + len(line), line.strip()))
        pos += len(line) + 1
    if not bounds:
        return []
    try:
        toks = analyze_with_offsets(headings_text)
    except Exception:
        return []
    units = [(raw, set()) for _, _, raw in bounds]
    bi = 0
    for tok, off in toks:
        while bi < len(bounds) and off >= bounds[bi][1]:
            bi += 1
        if bi >= len(bounds):
            break
        if off >= bounds[bi][0]:
            units[bi][1].add(tok)
    return units


def doc_signals(headings_text, sq_term_sets, role_terms):
    """1文書について、any / role別 の『話題一致見出しあり』フラグを返す。
    role_terms = 役割語そのもののstem集合。内容語一致の計算からは除外する。"""
    units = doc_heading_units(headings_text)
    sig = {"any": False}
    sig.update({r: False for r in ROLE_PATTERNS})
    for raw, stems in units:
        content = stems - role_terms
        if not any(len(content & s) >= MIN_OVERLAP for s in sq_term_sets):
            continue
        sig["any"] = True
        for r, rx in ROLE_RE.items():
            if rx.search(raw):
                sig[r] = True
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="coverage", choices=["coverage", "consensus"])
    ap.add_argument("--limit", type=int, default=0, help="トピック数の上限（0=全部）")
    args = ap.parse_args()

    rng = random.Random(SEED)
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS[args.topics])
    with open(DECOMPOSED_FILE) as f:
        decomposed = json.load(f)["results"]

    # 役割語そのもののstem（内容語一致から除外するため）
    role_terms = set(analyze_terms(
        "what is are definition defined meaning mean how it works explained detail details "
        "overview about history origin origins etymology background faq frequently asked "
        "questions summary conclusion key takeaways short bottom line recap warning caution "
        "risk risks side effects precautions important note vs versus difference between "
        "compared comparison steps guide instructions tutorial why cause causes reason reasons",
        field="headings"))

    topics = sorted(t for t in qrels if t in queries and t in decomposed)
    if args.limit:
        topics = topics[:args.limit]
    print(f"{args.topics}: {len(topics)} topics  (role_terms={len(role_terms)})")

    counts = {"pos": Counter(), "neg": Counter()}
    totals = {"pos": 0, "neg": 0}

    for i, qid in enumerate(topics, 1):
        sqs = decomposed[qid].get("decomposed_queries") or []
        sq_term_sets = [set(analyze_terms(s, field="headings")) - role_terms for s in sqs]
        sq_term_sets = [s for s in sq_term_sets if s]
        if not sq_term_sets:
            continue
        pos_ids = [d for d, g in qrels[qid].items() if g >= 1]
        judged_neg = [d for d, g in qrels[qid].items() if g == 0]
        if len(pos_ids) > POS_CAP:
            pos_ids = rng.sample(pos_ids, POS_CAP)
        neg_pool = list(judged_neg)
        if len(neg_pool) < NEG_CAP:  # 判定済み負例が足りない分はBM25上位の未判定文書で補う
            retrieved = [d for d, _ in bm25_body(queries[qid], k=NEG_POOL_K)]
            neg_pool += [d for d in retrieved if d not in qrels[qid]]
        neg_ids = neg_pool[:NEG_CAP * 3]
        if len(neg_ids) > NEG_CAP:
            neg_ids = rng.sample(neg_ids, NEG_CAP)

        heads = fetch_headings(pos_ids + neg_ids)
        for group, ids in (("pos", pos_ids), ("neg", neg_ids)):
            for d in ids:
                if d not in heads:
                    continue
                totals[group] += 1
                sig = doc_signals(heads[d], sq_term_sets, role_terms)
                for k, v in sig.items():
                    if v:
                        counts[group][k] += 1
        print(f"  [{i}/{len(topics)}] {qid}: pos={len(pos_ids)} neg={len(neg_ids)}", flush=True)

    rows = ["any"] + list(ROLE_PATTERNS)
    print(f"\n=== 話題一致見出しを持つ文書の割合(%)   pos n={totals['pos']}  neg n={totals['neg']} ===")
    print("signal".ljust(12) + "relevant".rjust(10) + "nonrel".rjust(10) + "  lift(rel/nonrel)")
    result = {}
    for r in rows:
        p = 100 * counts["pos"][r] / max(totals["pos"], 1)
        n = 100 * counts["neg"][r] / max(totals["neg"], 1)
        lift = (p / n) if n > 0 else float("nan")
        result[r] = {"rel_pct": round(p, 2), "nonrel_pct": round(n, 2), "lift": round(lift, 3)}
        print(f"{r.ljust(12)}{p:10.2f}{n:10.2f}{lift:19.3f}")

    out = os.path.join(RAG_DIR, f"heading_role_conditional_{args.topics}.json")
    with open(out, "w") as f:
        json.dump({"topics": args.topics, "n_topics": len(topics), "min_overlap": MIN_OVERLAP,
                   "totals": totals, "result": result}, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
