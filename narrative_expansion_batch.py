"""
TREC 2025 RAG のナラティブから HyDE 文書をまとめて生成する（バッチ内多様性版）。

narrative_expansion.py は 1トピック x N_POOL本 を「1文書=1リクエスト」で
独立に生成していたが、温度1.0でも同じテンプレートに収束しやすいことが分かった。
このスクリプトは 1トピック=1リクエスト にして、N_POOL本すべてを同じコンテキスト内で
まとめて書かせる。モデルが「自分が既に書いた内容と重複しないように」書き分けることで
プール全体の多様性が上がるか比較するのが狙い。

出力フォーマットは narrative_expansion.py と同じ（config + results、
results[qid].hyde_results.hyde_k）なので、evaluate.py の HYDE_FILE を
差し替えるだけでそのまま比較評価できる。

使い方:
    python narrative_expansion_batch.py smoke    # 2トピック x 5本だけ試す
    python narrative_expansion_batch.py submit   # 本番投入
    python narrative_expansion_batch.py poll     # 進捗確認
    python narrative_expansion_batch.py fetch    # 結果回収
"""

import json
import os
import re
import sys
import time
import hashlib

import requests

# ============================================================
# 設定
# ============================================================
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/beta/batches"

MODEL = "openai/gpt-5.6-terra"
REASONING = {"effort": "none"}
TEMPERATURE = 1.0

WORDS_PER_DOC = 200
N_POOL = 30                      # 1トピックあたり生成する本数（1リクエストにまとめて生成）
K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30]

SMOKE_N = 5                      # smoke で試す本数（本番より少なくして安く確認する）

QUERY_FIELD = "title"
INPUT_FILE = os.path.expanduser("~/data/trec_rag_2025_queries.jsonl")

CHUNK = 1000
IDS_FILE = "batch_ids_batch.json"
OUT_FILE = f"multi_hyde_batch_L{WORDS_PER_DOC}.json"

DELIM = "@@@---@@@"               # 文書同士の区切り。本文に出てくる見込みはほぼ無い記号列

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# プロンプト
# ============================================================
PROMPT = """Write {n} independent hypothetical documents that would each answer the
following search query. Each document should be a single paragraph of approximately
{words} words, written in English as declarative sentences of fact. Do not include a
title, heading, or any label ending in a colon, and do not use numbering or bullets
within a document.

Make the {n} documents meaningfully different from one another: vary the opening
angle, sentence structure, vocabulary, and which facts are emphasized first, so that
together they cover the topic from diverse perspectives rather than repeating the
same template. Do not summarize or refer to the other documents.

Separate the documents with a line that contains only this exact marker and nothing
else: {delim}
Do not put the marker before the first document or after the last document.

Search query: {query}"""

def max_tokens_for(n):
    return int(n * WORDS_PER_DOC * 1.6) + 500

CONFIG = {
    "model": MODEL,
    "reasoning": REASONING,
    "temperature": TEMPERATURE,
    "words_per_doc": WORDS_PER_DOC,
    "n_pool": N_POOL,
    "query_field": QUERY_FIELD,
    "generation_mode": "batched_single_request",
    "delim": DELIM,
    "prompt_hash": hashlib.sha256(PROMPT.encode()).hexdigest()[:16],
}


# ============================================================
# 入力の読み込み
# ============================================================
def load_queries():
    """[(qid, ナラティブ本文), ...] を返す"""
    out = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                qid = str(d["id"])
                assert "__" not in qid, f"qid に '__' が含まれる: {qid}"
                out.append((qid, d[QUERY_FIELD]))
    return out


def make_body(query, n):
    return {
        "messages": [
            {"role": "user",
             "content": PROMPT.format(n=n, words=WORDS_PER_DOC, delim=DELIM, query=query)}
        ],
        "reasoning": REASONING,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens_for(n),
    }


def build_requests(queries):
    """1トピック = 1リクエスト。custom_id はそのまま qid。"""
    return [{"custom_id": qid, "body": make_body(query, N_POOL)}
            for qid, query in queries]


def split_docs(content, expect_n):
    """区切り記号でN本に分割する。前後の空白・空行のゆらぎを許容する。"""
    parts = re.split(r"\n?\s*" + re.escape(DELIM) + r"\s*\n?", content)
    docs = [p.strip() for p in parts if p.strip()]
    if len(docs) != expect_n:
        print(f"  WARN 区切りで{len(docs)}本しか取れなかった（期待{expect_n}本）")
    return docs


# ============================================================
# API 呼び出し
# ============================================================
def submit_chunk(reqs):
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": MODEL,
        "requests": reqs,
    }
    r = requests.post(BASE, headers=HEADERS, data=json.dumps(payload))
    if r.status_code not in (200, 202):
        raise RuntimeError(f"submit failed {r.status_code}: {r.text[:800]}")
    return r.json()["id"]


def get_batch(bid, retries=5, backoff=3):
    """投入直後は反映が遅延して404が返ることがあるのでリトライする。"""
    for attempt in range(retries):
        r = requests.get(f"{BASE}/{bid}", headers=HEADERS)
        if r.status_code == 404 and attempt < retries - 1:
            print(f"  ({bid} がまだ見つからない。{backoff}秒待って再試行 "
                  f"{attempt + 1}/{retries})")
            time.sleep(backoff)
            continue
        r.raise_for_status()
        return r.json()


# ============================================================
# コマンド
# ============================================================
def cmd_smoke():
    """2トピック x SMOKE_N本だけ生成して、区切り分割と中身を確認する"""
    reqs = [
        {"custom_id": "smoke_bm25", "body": make_body("what is BM25 ranking", SMOKE_N)},
        {"custom_id": "smoke_dense", "body": make_body("what is dense retrieval", SMOKE_N)},
    ]
    bid = submit_chunk(reqs)
    print(f"batch {bid} を投入しました。完了まで待ちます...")

    while True:
        b = get_batch(bid)
        print(f"  status = {b['status']}")
        if b["status"] in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(20)

    if b["status"] != "completed":
        print(json.dumps(b.get("error"), indent=2, ensure_ascii=False))
        return

    print("\n--- 課金 ---")
    print(json.dumps(b.get("usage"), indent=2))
    for item in b.get("results") or []:
        if item.get("error"):
            print(f"\n[{item['custom_id']}] ERROR: {item['error']}")
            continue
        content = item["response"]["body"]["choices"][0]["message"]["content"]
        docs = split_docs(content, SMOKE_N)
        print(f"\n=== {item['custom_id']}: {len(docs)}本 ===")
        for i, d in enumerate(docs):
            print(f"\n--- doc {i} ({len(d.split())} words) ---\n{d}")


def cmd_submit():
    if os.path.exists(IDS_FILE):
        raise SystemExit(f"{IDS_FILE} が既にあります。退避してから再実行してください。")

    queries = load_queries()
    reqs = build_requests(queries)
    print(f"{len(queries)} topics x 1 request（各リクエストで{N_POOL}本まとめて生成）")

    ids = []
    for i in range(0, len(reqs), CHUNK):
        bid = submit_chunk(reqs[i:i + CHUNK])
        ids.append(bid)
        print(f"  submitted {bid}  ({i} - {min(i + CHUNK, len(reqs))})")

    with open(IDS_FILE, "w") as f:
        json.dump({"config": CONFIG, "batch_ids": ids}, f, indent=2)
    print(f"\n-> {IDS_FILE} に保存しました。poll で進捗を見てください。")


def cmd_poll():
    ids = json.load(open(IDS_FILE))["batch_ids"]
    for bid in ids:
        b = get_batch(bid)
        c = b.get("request_counts") or {}
        print(f"{bid}  {b['status']:12s}  "
              f"完了 {c.get('completed', 0)}/{c.get('total', 0)}  "
              f"失敗 {c.get('failed', 0)}")


def cmd_fetch():
    store = json.load(open(IDS_FILE))

    pools, cost, n_err = {}, 0.0, 0
    for bid in store["batch_ids"]:
        b = get_batch(bid)
        if b["status"] != "completed":
            print(f"  {bid}: {b['status']} — まだ完了していません")
            continue
        cost += (b.get("usage") or {}).get("cost", 0.0)
        for item in b["results"]:
            qid = item["custom_id"]
            if item.get("error") or not item.get("response"):
                print(f"  ERROR {qid}: {item.get('error')}")
                n_err += 1
                continue
            content = item["response"]["body"]["choices"][0]["message"]["content"]
            docs = split_docs(content, N_POOL)
            pools[qid] = docs

    queries = dict(load_queries())
    results = {}
    for qid, pool in pools.items():
        if len(pool) != N_POOL:
            print(f"  WARN {qid}: {len(pool)}/{N_POOL} 本しかありません")
        results[qid] = {
            "original_query": queries.get(qid),
            "pool": pool,
            "hyde_results": {f"hyde_{k}": pool[:k] for k in K_VALUES if k <= len(pool)},
            "word_counts": [len(t.split()) for t in pool],
        }

    json.dump({"config": store["config"], "results": results},
              open(OUT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    wc = sorted(w for r in results.values() for w in r["word_counts"])
    if wc:
        n = len(wc)
        print(f"\n生成語長: 中央値={wc[n // 2]}, "
              f"25%={wc[n // 4]}, 75%={wc[3 * n // 4]}  (目標={WORDS_PER_DOC})")
    print(f"トピック数={len(results)}  文書数={sum(len(r['pool']) for r in results.values())}  エラー={n_err}")
    print(f"課金額=${cost:.4f}")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    cmds = {"smoke": cmd_smoke, "submit": cmd_submit,
            "poll": cmd_poll, "fetch": cmd_fetch}
    if cmd not in cmds:
        raise SystemExit(f"使い方: python {sys.argv[0]} [smoke|submit|poll|fetch]")
    cmds[cmd]()
