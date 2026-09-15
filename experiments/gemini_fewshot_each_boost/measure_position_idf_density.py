"""
posboost（文書bodyの先頭N語以内に出現したら加点）が理論上妥当かどうかを、qrels
（正解判定）を一切使わずに検証する。「クエリ語がどこにあれば正解になりやすいか」を
qrelsから学習するのはテストラベルのリーク（過学習）になるため、代わりに「情報量の多い
（コーパス全体でレアな＝IDFが高い）語が、文書内のどの位置に集中して出現しやすいか」という、
特定のクエリにも正解判定にも依存しないコーパス全体の書き方の性質（inverted pyramid的な
書き方の慣習があるなら、冒頭ほど固有名詞・専門用語のような高IDF語が多いはず、という仮説）
を測定する。

方法:
1. msmarco-v21-docからランダムにSAMPLE_N文書をサンプリング（function_score+random_score、
   クエリともqrelsとも無関係）。
2. 各文書のbody term vectors（positions=True, term_statistics=True）を_mtermvectorsで
   バッチ取得する。term_statisticsを実文書に対して取得すると、その文書に実際に出現する
   語についてグローバルなdoc_freqが一緒に返ってくるため、語ごとに別クエリを投げる必要が
   ない（1バッチ=1回のAPI呼び出しで位置とIDFの両方が手に入る）。
   注意: このdoc_freqは実際には1シャードだけの集計値だが（retriever.py term_idf()の
   docstring参照）、ランダムサンプリングでシャードにほぼ均等に散らばるため、位置ビン間の
   相対比較（本測定の目的）としては十分実用的。
3. 各語の出現位置をビン（POSITION_EDGES）に振り分け、ビンごとに出現した語のIDFの平均を
   集計する。「冒頭のビンほど平均IDFが高い」という傾向が見えれば、posboost的な位置減衰は
   コーパスの一般的性質として正当化できる。qrelsはこの測定に一切使わない
   （このスクリプトはload_qrelsすら呼ばない）。

使い方:
    python measure_position_idf_density.py [SAMPLE_N]   # 省略時は800
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict

from retriever import client, INDEX, _bm25f_total_docs

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
BATCH_SIZE = 100
RANDOM_SEED = 42

# 冒頭ほど細かく、遠方は粗く（posboostのspan_end=100付近の解像度を優先）
POSITION_EDGES = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]


def sample_doc_ids(n, seed=RANDOM_SEED):
    """クエリ・qrelsと無関係にランダムにn件の文書idをサンプリングする。"""
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


def bin_index(pos):
    for i in range(len(POSITION_EDGES) - 1):
        if POSITION_EDGES[i] <= pos < POSITION_EDGES[i + 1]:
            return i
    return None  # POSITION_EDGES[-1]以降は捨てる


def main():
    print(f"サンプル数: {SAMPLE_N}   バッチサイズ: {BATCH_SIZE}   seed: {RANDOM_SEED}")
    print(f"位置ビン: {POSITION_EDGES}")

    N = _bm25f_total_docs()
    print(f"インデックス全体の文書数 N={N}")

    doc_ids = sample_doc_ids(SAMPLE_N)
    print(f"サンプリングした文書数: {len(doc_ids)}")

    bin_idf_sum = [0.0] * (len(POSITION_EDGES) - 1)
    bin_count = [0] * (len(POSITION_EDGES) - 1)
    docs_with_body = 0
    total_tokens_seen = 0

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
            docs_with_body += 1
            doc_freq_field = body_tv.get("field_statistics", {}).get("doc_count", N)
            for term, info in body_tv["terms"].items():
                df = info.get("doc_freq", 0)
                if df <= 0:
                    continue
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                for tok in info.get("tokens", []):
                    pos = tok["position"]
                    b = bin_index(pos)
                    if b is None:
                        continue
                    bin_idf_sum[b] += idf
                    bin_count[b] += 1
                    total_tokens_seen += 1
        print(f"\r  {min(i + BATCH_SIZE, len(doc_ids))}/{len(doc_ids)} "
              f"({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    print(f"\nbodyを持つ文書: {docs_with_body}/{len(doc_ids)}   集計トークン数: {total_tokens_seen}")
    print("\n位置ビンごとの平均IDF（qrels不使用、コーパス全体からのランダムサンプル）:")
    print(f"{'position range':<20}{'n_tokens':<12}{'avg_idf':<10}")
    print("-" * 42)
    rows = []
    for i in range(len(POSITION_EDGES) - 1):
        lo, hi = POSITION_EDGES[i], POSITION_EDGES[i + 1]
        avg_idf = bin_idf_sum[i] / bin_count[i] if bin_count[i] else float("nan")
        rows.append({"lo": lo, "hi": hi, "n_tokens": bin_count[i], "avg_idf": avg_idf})
        print(f"[{lo:4d},{hi:5d})     {bin_count[i]:<12}{avg_idf:<10.4f}")

    out_path = os.path.join(RAG_DIR, "position_idf_density_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "sample_n": SAMPLE_N, "docs_with_body": docs_with_body,
            "total_tokens_seen": total_tokens_seen, "position_edges": POSITION_EDGES,
            "bins": rows,
        }, f, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
