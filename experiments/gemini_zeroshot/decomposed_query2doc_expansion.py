"""
decompose_narrative.py が作った Subquery（narrative を分解した簡潔な質問文）
1問ごとに、Query2doc 疑似文書を1本だけ生成する（zero-shot版）。

experiments/gemini_fewshot/ に同一構成のfew-shot版がある。比較実験用。

narrative_expansion.py は narrative 全体から複数本の Query2doc 疑似文書プールを作り、
後から k 本を切り出す方式だったが、こちらは「分解質問1問につきQuery2doc疑似文書1本」で固定。
分解数はトピックごとに可変なので、生成本数もトピックごとに変わる。

使い方:
    python decomposed_query2doc_expansion.py smoke    # 2問だけ試す
    python decomposed_query2doc_expansion.py submit   # 本番投入
    python decomposed_query2doc_expansion.py poll     # 進捗確認
    python decomposed_query2doc_expansion.py fetch    # 結果回収
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
TEMPERATURE = 1.0

WORDS_PER_DOC = 200

IN_FILE = "decomposed_queries.json"
CHUNK = 1000
IDS_FILE = "batch_ids_decomposed_query2doc.json"
OUT_FILE = f"multi_query2doc_decomposed_L{WORDS_PER_DOC}.json"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# プロンプト（zero-shot版）
# ============================================================
PROMPT = """Write one hypothetical document (a single paragraph) that would
answer the following search query. Target length: approximately {words} words.

Write in English. Start immediately with the body text as declarative sentences
of fact. Do not include a title, heading, or any label ending in a colon. Do not
use numbering, bullets, or prefixes such as "Document 1:". Write the document as
a single continuous paragraph with no line breaks.

Search query: {query}"""

CONFIG = {
    "model": MODEL,
    "reasoning": REASONING,
    "temperature": TEMPERATURE,
    "words_per_doc": WORDS_PER_DOC,
    "prompt_style": "zero-shot",
    "prompt_hash": hashlib.sha256(PROMPT.encode()).hexdigest()[:16],
}


# ============================================================
# 入力の読み込み
# ============================================================
def load_decomposed():
    """{qid: {"original_query":..., "decomposed_queries": [...]}}"""
    return json.load(open(IN_FILE, encoding="utf-8"))["results"]


def make_body(query):
    return {
        "messages": [
            {"role": "user",
             "content": PROMPT.format(words=WORDS_PER_DOC, query=query)}
        ],
        "reasoning": REASONING,
        "temperature": TEMPERATURE,
    }


def build_requests(decomposed):
    """1質問 = 1リクエスト。custom_id は '<qid>__q<番号>' 形式。"""
    reqs = []
    for qid, entry in decomposed.items():
        for j, dq in enumerate(entry["decomposed_queries"]):
            reqs.append({
                "custom_id": f"{qid}__q{j}",
                "body": make_body(dq),
            })
    return reqs


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
    """2問だけ生成して、中身を確認する"""
    reqs = [
        {"custom_id": "smoke__q0", "body": make_body("what is BM25 ranking")},
        {"custom_id": "smoke__q1", "body": make_body("what is dense retrieval")},
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
        t = item["response"]["body"]["choices"][0]["message"]["content"].strip()
        print(f"\n--- {item['custom_id']} ({len(t.split())} words) ---\n{t}")


def cmd_submit():
    if os.path.exists(IDS_FILE):
        raise SystemExit(f"{IDS_FILE} が既にあります。退避してから再実行してください。")

    decomposed = load_decomposed()
    reqs = build_requests(decomposed)
    n_topics = len(decomposed)
    print(f"{n_topics} topics x 可変質問数 = {len(reqs)} requests")

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
            if item.get("error") or not item.get("response"):
                print(f"  ERROR {item['custom_id']}: {item.get('error')}")
                n_err += 1
                continue
            body = item["response"]["body"]
            texts[item["custom_id"]] = \
                body["choices"][0]["message"]["content"].strip()

    decomposed = load_decomposed()
    results = {}
    for qid, entry in decomposed.items():
        dqs = entry["decomposed_queries"]
        docs = [texts.get(f"{qid}__q{j}") for j in range(len(dqs))]
        n_missing = sum(1 for d in docs if d is None)
        if n_missing:
            print(f"  WARN {qid}: {n_missing}/{len(dqs)} 問で Query2doc 生成が欠落")
        results[qid] = {
            "original_query": entry["original_query"],
            "decomposed_queries": dqs,
            # decomposed_queries[j] に対応する Query2doc 疑似文書（欠落時は None）
            "query2doc_docs": docs,
            "word_counts": [len(d.split()) if d else 0 for d in docs],
        }

    json.dump({"config": store["config"], "results": results},
              open(OUT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    wc = sorted(w for r in results.values() for w in r["word_counts"] if w)
    if wc:
        n = len(wc)
        print(f"\n生成語長: 中央値={wc[n // 2]}, "
              f"25%={wc[n // 4]}, 75%={wc[3 * n // 4]}  (目標={WORDS_PER_DOC})")
    print(f"トピック数={len(results)}  文書数={len(texts)}  エラー={n_err}")
    print(f"課金額=${cost:.4f}")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    cmds = {"smoke": cmd_smoke, "submit": cmd_submit,
            "poll": cmd_poll, "fetch": cmd_fetch}
    if cmd not in cmds:
        raise SystemExit(f"使い方: python {sys.argv[0]} [smoke|submit|poll|fetch]")
    cmds[cmd]()
