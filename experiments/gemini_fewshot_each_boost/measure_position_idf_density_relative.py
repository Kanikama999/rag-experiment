"""
measure_position_idf_density.py（絶対位置＝先頭から何語目か、で平均IDFを集計）の
相対位置版。絶対位置での測定では「100〜150語がむしろ谷、500語以降で冒頭と同水準まで
回復するU字カーブ」という結果だったが、文書長は中央値610語・p90で1993語・p99で
10102語と裾の長い分布（bm25_fielded_posboost_relativeのdocstring参照）でバラバラなため、
「絶対位置100語」が短い文書では本文の大半、長い文書では冒頭のごく一部、と意味が
文書ごとに違う状態で平均を取っていた可能性がある。本スクリプトは同じ測定を
「文書長で正規化した相対位置（0.0=先頭、1.0=末尾）」でやり直し、絶対位置版のU字が
文書長のばらつきによる見かけ上のものだったのか、正規化しても残る本質的な構造なのかを
切り分ける。

qrelsは一切使わない（measure_position_idf_density.pyと同じ設計方針）。極端に短い文書
（本文がMIN_DOC_LEN語未満）は相対位置が意味を持ちにくいため除外する。

使い方:
    python measure_position_idf_density_relative.py [SAMPLE_N]   # 省略時は800
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

from retriever import client, INDEX, _bm25f_total_docs

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
BATCH_SIZE = 100
RANDOM_SEED = 42
MIN_DOC_LEN = 20  # これ未満のbody語数の文書は相対位置が意味を持ちにくいため除外

# 絶対位置版で効果が冒頭10〜20語(中央値610語の文書で約1.6〜3.3%)に集中していたため、
# 冒頭を特に細かく刻む
FRAC_EDGES = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20,
              0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0001]


def sample_doc_ids(n, seed=RANDOM_SEED):
    res = client.search(index=INDEX, body={
        "size": n,
        "_source": False,
        "query": {
            "function_score": {
                "query": {"match_all": {}},
                "random_score": {"seed": seed, "field": "_seq_no"},
            }
        },
    })
    return [h["_id"] for h in res["hits"]["hits"]]


def bin_index(frac):
    for i in range(len(FRAC_EDGES) - 1):
        if FRAC_EDGES[i] <= frac < FRAC_EDGES[i + 1]:
            return i
    return None


def main():
    print(f"サンプル数: {SAMPLE_N}   バッチサイズ: {BATCH_SIZE}   seed: {RANDOM_SEED}   "
          f"MIN_DOC_LEN: {MIN_DOC_LEN}")
    print(f"相対位置ビン: {FRAC_EDGES}")

    N = _bm25f_total_docs()
    print(f"インデックス全体の文書数 N={N}")

    doc_ids = sample_doc_ids(SAMPLE_N)
    print(f"サンプリングした文書数: {len(doc_ids)}")

    bin_idf_sum = [0.0] * (len(FRAC_EDGES) - 1)
    bin_count = [0] * (len(FRAC_EDGES) - 1)
    docs_used = 0
    docs_skipped_short = 0
    total_tokens_seen = 0
    doc_lens = []

    t0 = time.time()
    for i in range(0, len(doc_ids), BATCH_SIZE):
        batch = doc_ids[i:i + BATCH_SIZE]
        tv = client.mtermvectors(index=INDEX, body={
            "ids": batch,
            "parameters": {
                "fields": ["body"],
                "term_statistics": True,
                "field_statistics": False,
                "positions": True,
                "offsets": False,
                "payloads": False,
            },
        })
        for doc in tv.get("docs", []):
            body_tv = doc.get("term_vectors", {}).get("body")
            if not body_tv or not body_tv.get("terms"):
                continue
            terms = body_tv["terms"]
            dl = sum(info.get("term_freq", 0) for info in terms.values())
            if dl < MIN_DOC_LEN:
                docs_skipped_short += 1
                continue
            docs_used += 1
            doc_lens.append(dl)
            for term, info in terms.items():
                df = info.get("doc_freq", 0)
                if df <= 0:
                    continue
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                for tok in info.get("tokens", []):
                    frac = tok["position"] / dl
                    b = bin_index(frac)
                    if b is None:
                        continue
                    bin_idf_sum[b] += idf
                    bin_count[b] += 1
                    total_tokens_seen += 1
        print(f"\r  {min(i + BATCH_SIZE, len(doc_ids))}/{len(doc_ids)} "
              f"({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    doc_lens.sort()
    n_dl = len(doc_lens)
    median_dl = doc_lens[n_dl // 2] if n_dl else float("nan")
    print(f"\n使用文書: {docs_used}   短すぎて除外: {docs_skipped_short}   "
          f"集計トークン数: {total_tokens_seen}")
    print(f"body語数の中央値(使用文書内): {median_dl}")

    print("\n相対位置ビンごとの平均IDF（qrels不使用、文書長で正規化）:")
    print(f"{'frac range':<20}{'n_tokens':<12}{'avg_idf':<10}")
    print("-" * 42)
    rows = []
    for i in range(len(FRAC_EDGES) - 1):
        lo, hi = FRAC_EDGES[i], FRAC_EDGES[i + 1]
        avg_idf = bin_idf_sum[i] / bin_count[i] if bin_count[i] else float("nan")
        rows.append({"lo": lo, "hi": hi, "n_tokens": bin_count[i], "avg_idf": avg_idf})
        print(f"[{lo:.4f},{hi:.4f})   {bin_count[i]:<12}{avg_idf:<10.4f}")

    out_path = os.path.join(RAG_DIR, "position_idf_density_relative_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "sample_n": SAMPLE_N, "docs_used": docs_used, "docs_skipped_short": docs_skipped_short,
            "median_doc_len": median_dl, "total_tokens_seen": total_tokens_seen,
            "frac_edges": FRAC_EDGES, "bins": rows,
        }, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
