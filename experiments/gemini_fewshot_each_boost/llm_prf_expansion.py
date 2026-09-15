"""
evaluate_decomposed_webstyle_prf.py の統計的PRF（significant_terms、RM3ライク）を、
LLMによる関連度判断に置き換えた版（Generative Relevance Feedback的なアプローチ）。

Subquery単位の流れ:
1. webstyle_narrative_fielded_recallopt と同じ一次検索(bm25_fielded_recallopt)を行う。
2. 上位PRF_DEPTH件の文書のtitle/headingsを取得する。
3. Subquery + それら文書のtitle/headingsをLLMに見せ、「検索クエリに追加すべき、
   本当に関連していそうなキーワード」をPRF_TERMS個までLLMに判断させて抽出する
   （統計的PRFのsignificant_terms()と同じ役割だが、統計ではなくLLMの意味理解で選ぶ）。

出力は評価スクリプト側でbody_qに追加して二次検索するための拡張語リスト。
一次検索はLLM呼び出し不要なのでこのスクリプト内で先に行い、その結果をプロンプトに
埋め込んでバッチ投入する（他のexpansionスクリプトと違い、生成前にOpenSearch検索が要る）。

使い方:
    python llm_prf_expansion.py smoke    # 2問だけ試す
    python llm_prf_expansion.py submit   # 本番投入（一次検索→バッチ投入）
    python llm_prf_expansion.py poll     # 進捗確認
    python llm_prf_expansion.py fetch    # 結果回収
"""

import json
import os
import re
import sys
import time
import hashlib

import requests

# retriever.py に bm25_fielded_recallopt という専用関数は無い（評価スクリプト側にある）
# ので、ここでは retriever.bm25_fielded を直接同じブースト比(2,1,1)で呼ぶ
from retriever import bm25_fielded, client, INDEX

# ============================================================
# 設定
# ============================================================
API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/beta/batches"

MODEL = "openai/gpt-5.6-terra"
REASONING = {"effort": "none"}
TEMPERATURE = 1.0

QUERY_REPEAT = 5          # 一次検索時のnarrative繰り返し回数（評価スクリプトと合わせる）
PRF_DEPTH = 10            # 一次検索の上位何件をLLMに見せるか
PRF_TERMS = 10            # LLMに抽出させる拡張語の上限数
TITLE_BOOST, HEADINGS_BOOST, BODY_BOOST = 2, 1, 1   # recalloptと同じブースト比

DECOMPOSED_FILE = "decomposed_queries.json"
WEBSTYLE_FILE = "multi_query2doc_decomposed_webstyle_gpt56terra_L200.json"
QUERIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                             "data", "trec_rag_2025_queries.jsonl")

CHUNK = 1000
IDS_FILE = "batch_ids_llm_prf.json"
OUT_FILE = "llm_prf_expansion_terms.json"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# プロンプト
# ============================================================
PROMPT = """You are refining a search query using pseudo-relevance feedback. Below is a
specific question (the search intent), followed by the title and section headings of the
top {k} documents currently retrieved for it.

Question: {subquery}

Retrieved documents:
{docs}

Identify up to {n} additional keywords or short phrases that appear genuinely relevant
across these documents and would help refine the search toward the most on-topic
results. Prefer specific, discriminative terms over generic words. Do not just repeat
words already in the question. If none of the retrieved documents seem relevant or no
useful additional terms exist, output an empty array.

Output only a JSON array of strings (lowercase keywords/short phrases), nothing else.
Example output: ["term one", "term two", "term three"]"""

CONFIG = {
    "model": MODEL,
    "reasoning": REASONING,
    "temperature": TEMPERATURE,
    "query_repeat": QUERY_REPEAT,
    "prf_depth": PRF_DEPTH,
    "prf_terms": PRF_TERMS,
    "field_boosts": {"title": TITLE_BOOST, "headings": HEADINGS_BOOST, "body": BODY_BOOST},
    "prompt_hash": hashlib.sha256(PROMPT.encode()).hexdigest()[:16],
}


# ============================================================
# 入力読み込み・一次検索
# ============================================================
def load_queries():
    queries = {}
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                queries[str(d["id"])] = d["title"]
    return queries


def load_decomposed():
    return json.load(open(DECOMPOSED_FILE, encoding="utf-8"))["results"]


def load_webstyle():
    return json.load(open(WEBSTYLE_FILE, encoding="utf-8"))["results"]


def fetch_snippets(doc_ids):
    if not doc_ids:
        return {}
    res = client.mget(index=INDEX, body={"ids": doc_ids},
                       params={"_source_includes": "title,headings"})
    out = {}
    for d in res["docs"]:
        if not d.get("found"):
            continue
        src = d.get("_source", {})
        out[d["_id"]] = {
            "title": (src.get("title") or "").strip(),
            "headings": (src.get("headings") or "").replace("\n", " / ").strip(),
        }
    return out


def format_docs(snippets, doc_ids):
    lines = []
    for i, docid in enumerate(doc_ids, 1):
        s = snippets.get(docid)
        if not s:
            continue
        title = s["title"] or "(no title)"
        headings = s["headings"] or "(no headings)"
        lines.append(f"{i}. Title: {title}\n   Headings: {headings}")
    return "\n".join(lines)


def first_pass_and_prompts(queries, decomposed, webstyle):
    """全Subqueryについて一次検索を行い、(custom_id, prompt) のリストを返す。
    OpenSearch検索のみでLLM呼び出しは無い。時間がかかるので進捗を出す。"""
    items = []
    qids = sorted(decomposed.keys())
    t0 = time.time()
    for i, qid in enumerate(qids, 1):
        if qid not in queries or qid not in webstyle:
            continue
        q = queries[qid]
        repeated_q = " ".join([q] * QUERY_REPEAT)
        dqs = decomposed[qid]["decomposed_queries"]
        docs_structured = webstyle[qid].get("query2doc_docs_structured", [])
        for j, dq in enumerate(dqs):
            if j >= len(docs_structured) or not docs_structured[j]:
                continue
            doc = docs_structured[j]
            if not doc.get("body"):
                continue
            title_q = f"{repeated_q} {dq} {doc['title']}"
            headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
            body_q = f"{repeated_q} {dq} {doc['body']}"
            first_pass = bm25_fielded(title_q, headings_q, body_q, k=PRF_DEPTH,
                                       title_boost=TITLE_BOOST, headings_boost=HEADINGS_BOOST,
                                       body_boost=BODY_BOOST)
            doc_ids = [docid for docid, _ in first_pass]
            snippets = fetch_snippets(doc_ids)
            docs_text = format_docs(snippets, doc_ids)
            if not docs_text.strip():
                continue
            prompt = PROMPT.format(k=PRF_DEPTH, subquery=dq, docs=docs_text, n=PRF_TERMS)
            items.append((f"{qid}__q{j}", prompt))
        if i % 20 == 0 or i == len(qids):
            print(f"\r  一次検索: {i}/{len(qids)} トピック ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()
    return items


def make_body(prompt):
    return {
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": REASONING,
        "temperature": TEMPERATURE,
    }


# ============================================================
# JSON応答のパース
# ============================================================
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_terms(raw_text):
    text = _FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(t).strip() for t in obj if str(t).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return None


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
    queries = load_queries()
    decomposed = load_decomposed()
    webstyle = load_webstyle()
    # 最初の2 Subqueryだけに絞る
    qids = sorted(decomposed.keys())[:1]
    small_decomposed = {qid: {**decomposed[qid],
                               "decomposed_queries": decomposed[qid]["decomposed_queries"][:2]}
                         for qid in qids}
    items = first_pass_and_prompts(queries, small_decomposed, webstyle)
    reqs = [{"custom_id": cid, "body": make_body(p)} for cid, p in items]
    print(f"{len(reqs)} 件のリクエストを準備しました")
    for cid, p in items:
        print(f"\n=== {cid} のプロンプト ===")
        print(p[:1500])

    bid = submit_chunk(reqs)
    print(f"\nbatch {bid} を投入しました。完了まで待ちます...")
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
        raw = item["response"]["body"]["choices"][0]["message"]["content"].strip()
        terms = parse_terms(raw)
        print(f"\n--- {item['custom_id']} ---")
        print("terms:", terms if terms is not None else f"[PARSE FAILED] raw={raw}")


def cmd_submit():
    if os.path.exists(IDS_FILE):
        raise SystemExit(f"{IDS_FILE} が既にあります。退避してから再実行してください。")

    queries = load_queries()
    decomposed = load_decomposed()
    webstyle = load_webstyle()
    items = first_pass_and_prompts(queries, decomposed, webstyle)
    reqs = [{"custom_id": cid, "body": make_body(p)} for cid, p in items]
    print(f"{len(reqs)} 件のリクエストを準備しました")

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

    raw_texts, cost, n_err = {}, 0.0, 0
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
            raw_texts[item["custom_id"]] = body["choices"][0]["message"]["content"].strip()

    results = {}
    n_parse_fail = 0
    for cid, raw in raw_texts.items():
        qid, slot = cid.split("__q")
        terms = parse_terms(raw)
        if terms is None:
            n_parse_fail += 1
            terms = []
        results.setdefault(qid, {})[int(slot)] = terms

    json.dump({"config": store["config"], "results": results},
              open(OUT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    n_terms = [len(v) for r in results.values() for v in r.values()]
    print(f"トピック数={len(results)}  Subquery数={len(raw_texts)}  "
          f"APIエラー={n_err}  JSONパース失敗={n_parse_fail}")
    if n_terms:
        print(f"平均抽出語数={sum(n_terms) / len(n_terms):.1f}  "
              f"0件だったSubquery数={sum(1 for n in n_terms if n == 0)}")
    print(f"課金額=${cost:.4f}")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    cmds = {"smoke": cmd_smoke, "submit": cmd_submit, "poll": cmd_poll, "fetch": cmd_fetch}
    if cmd not in cmds:
        raise SystemExit(f"使い方: python {sys.argv[0]} [smoke|submit|poll|fetch]")
    cmds[cmd]()
