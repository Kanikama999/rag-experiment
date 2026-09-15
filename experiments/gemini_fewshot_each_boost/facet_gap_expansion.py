"""
decompose_narrative.py が作った Subquery群（narrativeを分解した簡潔な質問文、平均4〜5個/
トピック）が、narrativeの全facetを機械的に均等カバーしている保証はない。keystone_doc
（narrative全体を扱う文書）は既存Subqueryだけでも当たりやすいが、narrativeの狭い一facet
だけを扱う周辺文書は、そのfacet用のSubqueryが存在しなければ候補にすら上がらない
（miss_analysis調査で実際に確認された問題）。

本スクリプトは、narrative + 既存Subquery群をLLMに見せ、「まだカバーされていないfacet」を
自己点検させ、不足分だけ追加のSubqueryを生成させる。全facetが既にカバーされている場合は
空出力（0問）を許容する。

出力(facet_gap_subqueries.json)は decomposed_queries.json と同じ構造
({"results": {qid: {"original_query", "decomposed_queries"}}}) で、
decomposed_query2doc_webstyle_expansion.py にそのまま入力できる
（=既存の疑似文書生成パイプラインを再利用して、追加Subquery分だけ疑似文書を作れる）。

使い方:
    python facet_gap_expansion.py smoke    # 2件だけ試す
    python facet_gap_expansion.py submit   # 本番投入
    python facet_gap_expansion.py poll     # 進捗確認
    python facet_gap_expansion.py fetch    # 結果回収
"""

import json
import os
import sys
import time
import hashlib

import requests

# ============================================================
# 設定（decompose_narrative.pyと同じモデル・reasoning。分解の一種なので同じ設定に揃える）
# ============================================================
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/beta/batches"

MODEL = "google/gemini-3.7-flash:batch"
REASONING = {"effort": "low"}
TEMPERATURE = 0.0  # 分解と同じく再現性重視

IN_FILE = "decomposed_queries.json"
CHUNK = 1000
IDS_FILE = "batch_ids_facet_gap.json"
OUT_FILE = "facet_gap_subqueries.json"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# プロンプト
# ============================================================
PROMPT = """You are given a narrative-style search query describing an information need, \
and a set of questions that were previously derived from it.

Narrative: {narrative}

Existing questions:
{existing_questions}

The existing questions were generated at a fairly coarse, high-level granularity, and may \
have merged several genuinely distinct sub-topics into a single broad question. Your job \
is to find facets that deserve their OWN specific, narrower question, even if they are \
loosely mentioned inside one of the existing broad questions.

Look specifically for facets like these, if the narrative touches on them:
- Specific named entities mentioned or implied in the narrative (a particular place, \
organization, institution, law, treaty, person, product, event, or example) that could \
each be the subject of their own dedicated page, but are only addressed in general terms \
by an existing question.
- Distinct sub-processes, mechanisms, causes, or steps that an existing broad question \
bundles together, but that could each stand as their own separate question.
- Contrasting viewpoints, controversies, or opposing positions on the topic that are not \
separately addressed by any existing question.
- Specific sub-populations, categories, or types (a particular demographic, group, \
species, condition, or variant) that the narrative touches on but that an existing \
question only addresses in general, undifferentiated terms.
- Concrete factual sub-questions (a definition, statistic, date, specific cause, specific \
effect, or specific outcome) that are folded into a broader existing question but could \
reasonably be searched for on their own.

For each such facet you find, write one new clear, self-contained question (understandable \
and answerable on its own, without seeing the narrative or the other questions) that is \
NARROWER than, and targets specifically, that facet. Every new question must be strictly \
more specific than all existing questions; do not write a question that merely rephrases \
or broadens an existing one.

Output exactly one new question per line, in English. Do not include numbering, bullets, \
labels, or blank lines. Do not include any commentary or explanation. If you genuinely \
cannot find any facet that deserves its own narrower question, output nothing at all (an \
empty response)."""

CONFIG = {
    "model": MODEL,
    "reasoning": REASONING,
    "temperature": TEMPERATURE,
    "prompt_hash": hashlib.sha256(PROMPT.encode()).hexdigest()[:16],
}


# ============================================================
# 入力の読み込み
# ============================================================
def load_decomposed():
    """{qid: {"original_query":..., "decomposed_queries": [...]}}"""
    return json.load(open(IN_FILE, encoding="utf-8"))["results"]


def make_body(narrative, existing_qs):
    existing_block = "\n".join(f"- {q}" for q in existing_qs)
    return {
        "messages": [
            {"role": "user",
             "content": PROMPT.format(narrative=narrative, existing_questions=existing_block)}
        ],
        "reasoning": REASONING,
        "temperature": TEMPERATURE,
    }


def build_requests(decomposed):
    """1トピック = 1リクエスト。custom_id はそのまま qid。"""
    return [{"custom_id": qid, "body": make_body(entry["original_query"], entry["decomposed_queries"])}
            for qid, entry in decomposed.items() if entry.get("decomposed_queries")]


def parse_questions(content):
    return [line.strip() for line in content.splitlines() if line.strip()]


# ============================================================
# API 呼び出し
# ============================================================
def submit_chunk(reqs):
    payload = {"endpoint": "/v1/chat/completions", "model": MODEL, "requests": reqs}
    r = requests.post(BASE, headers=HEADERS, data=json.dumps(payload))
    if r.status_code not in (200, 202):
        raise RuntimeError(f"submit failed {r.status_code}: {r.text[:800]}")
    return r.json()["id"]


def get_batch(bid, retries=5, backoff=3):
    for attempt in range(retries):
        r = requests.get(f"{BASE}/{bid}", headers=HEADERS)
        if r.status_code == 404 and attempt < retries - 1:
            print(f"  ({bid} がまだ見つからない。{backoff}秒待って再試行 {attempt + 1}/{retries})")
            time.sleep(backoff)
            continue
        r.raise_for_status()
        return r.json()


# ============================================================
# コマンド
# ============================================================
def cmd_smoke():
    decomposed = load_decomposed()
    sample_qids = list(decomposed.keys())[:2]
    reqs = [{"custom_id": qid, "body": make_body(decomposed[qid]["original_query"],
                                                  decomposed[qid]["decomposed_queries"])}
            for qid in sample_qids]
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
        qid = item["custom_id"]
        print(f"\n=== qid={qid} ===")
        print("narrative:", decomposed[qid]["original_query"])
        print("既存Subquery:")
        for q in decomposed[qid]["decomposed_queries"]:
            print("  -", q)
        if item.get("error"):
            print(f"  ERROR: {item['error']}")
            continue
        content = item["response"]["body"]["choices"][0]["message"].get("content") or ""
        qs = parse_questions(content)
        print(f"追加Subquery ({len(qs)}問):")
        for q in qs:
            print("  +", q)


def cmd_submit():
    if os.path.exists(IDS_FILE):
        raise SystemExit(f"{IDS_FILE} が既にあります。退避してから再実行してください。")

    decomposed = load_decomposed()
    reqs = build_requests(decomposed)
    print(f"{len(reqs)} topics x 1 request（facet不足チェック）")

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
        print(f"{bid}  {b['status']:12s}  完了 {c.get('completed', 0)}/{c.get('total', 0)}  "
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
            content = item["response"]["body"]["choices"][0]["message"].get("content") or ""
            texts[qid] = content

    decomposed = load_decomposed()
    results = {}
    n_empty = 0
    for qid, content in texts.items():
        qs = parse_questions(content)
        if not qs:
            n_empty += 1
        results[qid] = {
            "original_query": decomposed[qid]["original_query"],
            "decomposed_queries": qs,  # facet_gap分のみ。既存分は含めない
        }

    json.dump({"config": store["config"], "results": results},
              open(OUT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    counts = sorted(len(r["decomposed_queries"]) for r in results.values())
    if counts:
        n = len(counts)
        print(f"\n追加Subquery数: 中央値={counts[n // 2]}, 最小={counts[0]}, 最大={counts[-1]}, "
              f"追加0件(=既にfull coverage)のトピック={n_empty}/{len(results)}")
    print(f"トピック数={len(results)}  エラー={n_err}")
    print(f"課金額=${cost:.4f}")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    cmds = {"smoke": cmd_smoke, "submit": cmd_submit, "poll": cmd_poll, "fetch": cmd_fetch}
    if cmd not in cmds:
        raise SystemExit(f"使い方: python {sys.argv[0]} [smoke|submit|poll|fetch]")
    cmds[cmd]()
