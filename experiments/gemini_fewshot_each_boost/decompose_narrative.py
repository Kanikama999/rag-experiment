"""
TREC 2025 RAG のナラティブを、わかりやすく簡潔な質問文（Subquery）に分解する。

narrative_expansion.py が narrative 全体から直接 Query2doc 疑似文書を作っていたのに対し、
こちらは Query2doc の前段として「narrative が扱っている論点ごとに、単独で検索クエリとして
使える簡潔な質問文」に分解するだけを行う。分解数はトピックごとに可変（LLM に任せる）。

出力は decomposed_query2doc_expansion.py の入力になる。

使い方:
    python decompose_narrative.py smoke    # 2件だけ試す
    python decompose_narrative.py submit   # 本番投入
    python decompose_narrative.py poll     # 進捗確認
    python decompose_narrative.py fetch    # 結果回収
"""

import json
import os
import sys
import time
import hashlib

import requests

# ============================================================
# 設定
# ============================================================
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/beta/batches"

MODEL = "google/gemini-3.7-flash:batch"
REASONING = {"effort": "low"}    # このモデルはreasoning必須（"none"不可）。最小のlowを指定
TEMPERATURE = 0.0                # 分解は再現性重視。多様性は不要

QUERY_FIELD = "title"
INPUT_FILE = os.path.expanduser("~/data/trec_rag_2025_queries.jsonl")

CHUNK = 1000
IDS_FILE = "batch_ids_decompose.json"
OUT_FILE = "decomposed_queries.json"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# プロンプト
# ============================================================
PROMPT = """The following is a narrative-style search query describing an information need.
Decompose it into a small number of clear, self-contained, concise questions that
together cover the distinct facets it is asking about. Each question must be
understandable and answerable on its own, without seeing the narrative.

Output exactly one question per line, in English. Do not include numbering,
bullets, labels, or blank lines. Output nothing except the questions themselves.

Narrative: {narrative}"""

CONFIG = {
    "model": MODEL,
    "reasoning": REASONING,
    "temperature": TEMPERATURE,
    "query_field": QUERY_FIELD,
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


def make_body(narrative):
    return {
        "messages": [
            {"role": "user", "content": PROMPT.format(narrative=narrative)}
        ],
        "reasoning": REASONING,
        "temperature": TEMPERATURE,
    }


def build_requests(queries):
    """1トピック = 1リクエスト。custom_id はそのまま qid。"""
    return [{"custom_id": qid, "body": make_body(narrative)}
            for qid, narrative in queries]


def parse_questions(content):
    """1行1質問として分割する。空行は無視する。"""
    return [line.strip() for line in content.splitlines() if line.strip()]


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
    """2件だけ分解して、中身を確認する"""
    queries = load_queries()[:2]
    reqs = [{"custom_id": qid, "body": make_body(narrative)} for qid, narrative in queries]
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
        qs = parse_questions(content)
        print(f"\n=== {item['custom_id']}: {len(qs)}問 ===")
        for i, q in enumerate(qs):
            print(f"  {i}: {q}")


def cmd_submit():
    if os.path.exists(IDS_FILE):
        raise SystemExit(f"{IDS_FILE} が既にあります。退避してから再実行してください。")

    queries = load_queries()
    reqs = build_requests(queries)
    print(f"{len(queries)} topics x 1 request（各トピックを分解）")

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

    texts, cost, n_err = {}, 0.0, 0
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
            texts[qid] = content

    queries = dict(load_queries())
    results = {}
    for qid, content in texts.items():
        qs = parse_questions(content)
        if not qs:
            print(f"  WARN {qid}: 分解結果が0件")
        results[qid] = {
            "original_query": queries.get(qid),
            "decomposed_queries": qs,
        }

    json.dump({"config": store["config"], "results": results},
              open(OUT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    counts = sorted(len(r["decomposed_queries"]) for r in results.values())
    if counts:
        n = len(counts)
        print(f"\n分解数: 中央値={counts[n // 2]}, "
              f"最小={counts[0]}, 最大={counts[-1]}")
    print(f"トピック数={len(results)}  エラー={n_err}")
    print(f"課金額=${cost:.4f}")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    cmds = {"smoke": cmd_smoke, "submit": cmd_submit,
            "poll": cmd_poll, "fetch": cmd_fetch}
    if cmd not in cmds:
        raise SystemExit(f"使い方: python {sys.argv[0]} [smoke|submit|poll|fetch]")
    cmds[cmd]()
