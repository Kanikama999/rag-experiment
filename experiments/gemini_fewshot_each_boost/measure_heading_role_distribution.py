"""
「見出しには役割がある」という仮説の事前検証（retrieval実装より前の premise check）。

MS MARCO v2.1 doc の headings フィールドを改行で分割して1見出しずつに戻し、
役割ラベル（定義/詳細/起源/QA/要約/注意/比較/手順/原因）を正規表現で付与する。

比較する2集団:
  - relevant : coverage qrels（TREC公式・セグメント判定を文書に集約）で rel>=1 の文書
  - control  : コーパスからのランダムサンプル（random_score）

見たいこと:
  (1) そもそも役割見出しはどのくらいの文書に存在するか（存在しないなら手法が成立しない）
  (2) relevant の方が control より役割見出しを持ちやすいか（持ちやすいならブースト根拠になる）

使い方:
    python measure_heading_role_distribution.py
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict

from retriever import client, INDEX

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
COVERAGE_QRELS = os.path.join(DATA_DIR, "keystone_qrels_coverage.txt")
CONSENSUS_QRELS = os.path.join(DATA_DIR, "keystone_qrels_consensus.txt")
OUT_FILE = os.path.join(RAG_DIR, "heading_role_distribution_result.json")

CONTROL_N = 5000
CONSENSUS_SAMPLE_N = 5000
SEED = 0

# 画像の「見出し→役割」表を英語Webページ向けに移したもの。
# 見出し1本（改行区切りの1行）に対して部分一致で判定する。
ROLE_PATTERNS = {
    "definition": r"\bwhat (is|are|was|were)\b|\bdefinition\b|\bdefined\b|\bmeaning\b|\bwhat does .* mean\b",
    "detail":     r"\bhow (it|they|does|do) work|\bexplained\b|\bin detail\b|\bdetails\b|\boverview\b|\babout\b",
    "origin":     r"\bhistory\b|\borigin(s)?\b|\betymolog|\bwhere .* (come|comes) from\b|\bbackground\b",
    "qa":         r"\bfaq(s)?\b|frequently asked|\bq ?& ?a\b|\bquestions\b|\bpeople also ask\b",
    "summary":    r"\bsummary\b|\bconclusion(s)?\b|key takeaway|\bin short\b|bottom line|\btl;?dr\b|\brecap\b",
    "caution":    r"\bwarning(s)?\b|\bcaution\b|\brisk(s)?\b|side effect|\bprecaution|\bimportant\b|\bnote\b|\bbeware\b|\bdanger",
    # 追加ロール（narrativeクエリで頻出しそうなもの。表には無いが同じ枠組みで測る）
    "comparison": r"\bvs\.?\b|\bversus\b|difference(s)? between|\bcompared? (to|with)\b|\bcomparison\b",
    "procedure":  r"\bhow to\b|\bsteps?\b|\bguide\b|\binstructions\b|\btutorial\b",
    "cause":      r"\bwhy\b|\bcause(s|d)?\b|\breason(s)?\b|\bdue to\b",
}
ROLE_RE = {r: re.compile(p, re.I) for r, p in ROLE_PATTERNS.items()}


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qrels[parts[0]][parts[2]] = int(parts[3])
    return qrels


def split_headings(text):
    if not text:
        return []
    return [h.strip() for h in text.split("\n") if h.strip()]


def classify(heading):
    return [r for r, rx in ROLE_RE.items() if rx.search(heading)]


def fetch_headings(docids, batch=500):
    """mgetでheadingsだけ取る。存在しないIDはスキップ。"""
    out = {}
    docids = list(docids)
    for i in range(0, len(docids), batch):
        chunk = docids[i:i + batch]
        res = client.mget(body={"ids": chunk}, index=INDEX,
                          params={"_source_includes": "headings"})
        for d in res["docs"]:
            if d.get("found"):
                out[d["_id"]] = d["_source"].get("headings", "")
    return out


def sample_corpus(n):
    """random_scoreでコーパスからn件サンプリング（1リクエスト上限10000）。"""
    res = client.search(index=INDEX, body={
        "size": n,
        "_source": ["headings"],
        "query": {"function_score": {"query": {"match_all": {}},
                                     "random_score": {"seed": SEED, "field": "_seq_no"}}},
    })
    return {h["_id"]: h["_source"].get("headings", "") for h in res["hits"]["hits"]}


def profile(name, headings_by_doc):
    n_docs = len(headings_by_doc)
    docs_with_any_heading = 0
    docs_with_role = Counter()      # roleを1本以上持つ文書数
    role_heading_count = Counter()  # role見出しの延べ本数
    total_headings = 0
    uniq_total = 0
    for text in headings_by_doc.values():
        hs = split_headings(text)
        uniq = list(dict.fromkeys(hs))  # 同一見出しの重複を除いた本数も見る
        total_headings += len(hs)
        uniq_total += len(uniq)
        if hs:
            docs_with_any_heading += 1
        seen = set()
        for h in uniq:
            for r in classify(h):
                role_heading_count[r] += 1
                seen.add(r)
        for r in seen:
            docs_with_role[r] += 1
    return {
        "name": name,
        "n_docs": n_docs,
        "docs_with_any_heading_pct": round(100 * docs_with_any_heading / max(n_docs, 1), 2),
        "mean_headings_per_doc": round(total_headings / max(n_docs, 1), 2),
        "mean_uniq_headings_per_doc": round(uniq_total / max(n_docs, 1), 2),
        "pct_docs_with_role": {r: round(100 * docs_with_role[r] / max(n_docs, 1), 2)
                               for r in ROLE_PATTERNS},
        "mean_role_headings_per_doc": {r: round(role_heading_count[r] / max(n_docs, 1), 3)
                                       for r in ROLE_PATTERNS},
    }


def main():
    rng = random.Random(SEED)

    cov = load_qrels(COVERAGE_QRELS)
    cov_rel = sorted({d for t in cov.values() for d, g in t.items() if g >= 1})
    cov_nonrel = sorted({d for t in cov.values() for d, g in t.items() if g == 0})
    con = load_qrels(CONSENSUS_QRELS)
    con_rel = sorted({d for t in con.values() for d, g in t.items() if g >= 1})
    if len(con_rel) > CONSENSUS_SAMPLE_N:
        con_rel = rng.sample(con_rel, CONSENSUS_SAMPLE_N)

    print(f"coverage rel>=1: {len(cov_rel)}  coverage rel==0: {len(cov_nonrel)}  "
          f"consensus rel>=1(sampled): {len(con_rel)}")

    groups = {}
    print("取得中: coverage relevant ...")
    groups["coverage_relevant"] = fetch_headings(cov_rel)
    print("取得中: coverage judged-nonrelevant ...")
    groups["coverage_nonrelevant"] = fetch_headings(cov_nonrel)
    print("取得中: consensus relevant ...")
    groups["consensus_relevant"] = fetch_headings(con_rel)
    print("取得中: corpus random ...")
    groups["corpus_random"] = sample_corpus(CONTROL_N)

    profiles = {k: profile(k, v) for k, v in groups.items()}

    order = ["corpus_random", "coverage_nonrelevant", "coverage_relevant", "consensus_relevant"]
    roles = list(ROLE_PATTERNS)
    w = max(len(r) for r in roles) + 1
    print("\n=== 役割見出しを1本以上持つ文書の割合(%) ===")
    print("role".ljust(w) + "".join(k[:14].rjust(16) for k in order))
    for r in roles:
        print(r.ljust(w) + "".join(f"{profiles[k]['pct_docs_with_role'][r]:16.2f}" for k in order))
    print("\n" + "n_docs".ljust(w) + "".join(f"{profiles[k]['n_docs']:16d}" for k in order))
    print("uniq_head/doc".ljust(w) + "".join(f"{profiles[k]['mean_uniq_headings_per_doc']:16.2f}" for k in order))

    with open(OUT_FILE, "w") as f:
        json.dump({"role_patterns": ROLE_PATTERNS, "profiles": profiles}, f, indent=2)
    print(f"\n-> {OUT_FILE}")


if __name__ == "__main__":
    main()
