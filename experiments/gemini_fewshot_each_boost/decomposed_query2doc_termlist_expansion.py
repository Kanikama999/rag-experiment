"""
decomposed_query2doc_webstyle_expansion.py（title/headings/bodyを「文章」として生成する版）
の代替案。BM25のmatchクエリは本質的に「語の集合」としてクエリを扱うため、文章として
自然に読める必要はない、という発想に基づき、文章を書かせる代わりに最初から
「title欄に入りそうな語をN個」「headings欄に入りそうな語をK個」「body欄に入りそうな語を
M個」を直接列挙させる。

狙い:
- 文法・接続詞などの無駄なトークンを削り、同じ生成コストでより多くの有用な語を作れる
- 語数を明示的にコントロールできる（richterms実験でLuceneのmaxClauseCount(1024)に
  ぶつかった問題を、生成段階で語数を絞ることで根本的に回避できる）
- 実際のwebページのtitleも文章というよりキーワード寄りの短い句であることが多く、
  「文章」より「語の集合」の方が実態に近い可能性がある

出力の "query2doc_docs_terms" は {"title_terms": [...], "heading_terms": [...],
"body_terms": [...]} で、それぞれ" "で連結すればtitle_q/headings_q/body_qの
構築にそのまま使える（既存のevaluate_decomposed_webstyle_equalweight.pyのdq_pairs_structured
と同じ役割）。

使い方:
    python decomposed_query2doc_termlist_expansion.py smoke    # 2問だけ試す
    python decomposed_query2doc_termlist_expansion.py submit   # 本番投入
    python decomposed_query2doc_termlist_expansion.py poll     # 進捗確認
    python decomposed_query2doc_termlist_expansion.py fetch    # 結果回収
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

MODEL = "google/gemini-3.7-flash:batch"
REASONING = {"effort": "low"}    # このモデルはreasoning必須（"none"不可）。最小のlowを指定
TEMPERATURE = 1.0

N_TITLE_TERMS = 8
N_HEADING_TERMS = 20
N_BODY_TERMS = 40

IN_FILE = "decomposed_queries.json"
CHUNK = 1000
IDS_FILE = "batch_ids_termlist_expansion.json"
OUT_FILE = f"multi_query2doc_decomposed_termlist_T{N_TITLE_TERMS}_H{N_HEADING_TERMS}_B{N_BODY_TERMS}.json"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# プロンプト（few-shot、JSON出力でtitle/heading/body別に語リストを生成させる）
# ============================================================
FEWSHOT_EXAMPLES = [
    {
        "query": "What causes ocean tides, and how do the sun and moon each contribute?",
        "title_terms": [
            "ocean tides causes", "moon's gravitational pull", "sun's tidal effect",
            "why do tides happen", "tidal forces explained", "high tide low tide",
            "lunar tidal pull", "spring tides neap tides",
        ],
        "heading_terms": [
            "moon's gravitational pull", "sun's smaller contribution", "spring tides",
            "neap tides", "local geography and tidal range", "tidal bulge formation",
            "Earth's rotation and inertia", "gravitational pull strength",
            "tidal cycle timing", "coastal amplification effects",
            "Bay of Fundy tidal range", "new moon full moon alignment",
            "quarter moon tidal cancellation", "tidal range variation",
            "seafloor depth influence", "estuary width impact",
            "24 hour 50 minute cycle", "two high tides daily",
            "astronomical tidal influences", "tidal amplitude differences",
        ],
        "body_terms": [
            "gravitational pull", "seawater bulge", "Earth-Moon system", "inertial forces",
            "Earth's rotation", "high tide", "low tide", "tidal cycle",
            "24 hours 50 minutes", "solar tidal effect", "46 percent as strong",
            "new moon", "full moon", "spring tides", "quarter moon", "neap tides",
            "gravitational alignment", "coastline shape", "seafloor depth", "bay width",
            "estuary width", "tidal range", "Bay of Fundy", "fifteen meters",
            "less than a foot", "astronomical effects", "tidal amplification",
            "tidal dampening", "moon's proximity", "sun's distance",
            "ocean water movement", "gravitational force", "Earth's oceans",
            "tidal bulge", "opposite side bulge", "lunar cycle", "solar cycle",
            "coastal geography", "tidal magnitude", "tide timing",
        ],
    },
    {
        "query": "How do vaccines train the immune system to fight future infections?",
        "title_terms": [
            "how vaccines work", "immune system training", "vaccine immunity",
            "antibodies and vaccines", "immune memory explained", "vaccination process",
            "T cells B cells vaccines", "how immunity develops",
        ],
        "heading_terms": [
            "introducing a harmless pathogen component", "weakened virus",
            "inactivated bacterium", "activating T cells and B cells",
            "antigen-presenting cells", "adaptive immune response",
            "antibodies and immune memory", "memory cells persistence",
            "faster response on re-exposure", "secondary immune response",
            "vaccine types overview", "immune system recognition",
            "pathogen surface proteins", "helper T cells role",
            "killer T cells function", "long-lived memory cells",
            "measles polio influenza immunity", "milder illness after vaccination",
            "immunological memory basis", "years decades of protection",
        ],
        "body_terms": [
            "harmless pathogen component", "weakened virus", "inactivated bacterium",
            "viral protein fragment", "immune system recognition", "antigen-presenting cells",
            "T cells", "B cells", "adaptive immune response", "antibodies",
            "pathogen's surface proteins", "helper T cells", "immune response coordination",
            "killer T cells", "infected cells elimination", "memory cells",
            "long-lived immunity", "years or decades", "real pathogen exposure",
            "immediate recognition", "faster immune response", "stronger secondary response",
            "unvaccinated person comparison", "neutralizing infection", "symptoms prevention",
            "immunological memory", "milder illness", "no illness at all",
            "measles immunity", "polio immunity", "influenza immunity",
            "vaccination benefits", "immune system training", "disease prevention",
            "antibody production", "immune cell activation", "vaccine components",
            "protective immunity", "immune response speed", "infection resistance",
        ],
    },
]


def _format_fewshot_block():
    blocks = []
    for ex in FEWSHOT_EXAMPLES:
        out = {"title_terms": ex["title_terms"], "heading_terms": ex["heading_terms"],
               "body_terms": ex["body_terms"]}
        blocks.append(f"Search query: {ex['query']}\n{json.dumps(out, ensure_ascii=False)}")
    return "\n\n".join(blocks)


FEWSHOT_BLOCK = _format_fewshot_block()

PROMPT = """For the following search query, generate three lists of words and short
phrases (not full sentences) that would likely appear in a real, informative web page
answering this query, as if you were extracting keywords from the title tag, section
headings, and body text of that page.

Output only a single JSON object with exactly these keys:
- "title_terms": a JSON array of exactly {n_title} words/short phrases (each under 6
  words) that would plausibly appear in the page's <title> tag or headline.
- "heading_terms": a JSON array of exactly {n_heading} words/short phrases (each under
  6 words) that would plausibly appear across the page's section headings (<h2>/<h3>).
- "body_terms": a JSON array of exactly {n_body} words/short phrases (each under 5
  words) that would plausibly appear in the page's body text.

Across all three lists, use many different paraphrases, synonyms, and related technical
terms for the key concepts in the query instead of repeating the same wording — vary
terminology, phrasing, and word forms (e.g. synonyms, related technical terms,
singular/plural, noun/verb forms) so that a reader searching with different wordings
for the same idea would still find a matching term. Prefer concrete, topic-specific
terms over generic filler words. Do not repeat the exact same phrase across lists.

Write in English. Do not wrap the JSON in markdown code fences. Do not include any
text outside the JSON object.

Here are examples of the expected input/output style:

{examples}

Now generate term lists in the same style for this query:

Search query: {query}"""

CONFIG = {
    "model": MODEL,
    "reasoning": REASONING,
    "temperature": TEMPERATURE,
    "n_title_terms": N_TITLE_TERMS,
    "n_heading_terms": N_HEADING_TERMS,
    "n_body_terms": N_BODY_TERMS,
    "prompt_style": "few-shot-termlist",
    "n_fewshot_examples": len(FEWSHOT_EXAMPLES),
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
             "content": PROMPT.format(n_title=N_TITLE_TERMS, n_heading=N_HEADING_TERMS,
                                       n_body=N_BODY_TERMS, examples=FEWSHOT_BLOCK, query=query)}
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
# JSON応答のパース
# ============================================================
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_termlist(raw_text):
    """LLM出力から {"title_terms", "heading_terms", "body_terms"} を取り出す。
    壊れていたらNoneを返す。"""
    text = _FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        obj = json.loads(text)
        title_terms = [t.strip() for t in (obj.get("title_terms") or []) if isinstance(t, str) and t.strip()]
        heading_terms = [t.strip() for t in (obj.get("heading_terms") or []) if isinstance(t, str) and t.strip()]
        body_terms = [t.strip() for t in (obj.get("body_terms") or []) if isinstance(t, str) and t.strip()]
        if not body_terms:
            return None
        return {"title_terms": title_terms, "heading_terms": heading_terms, "body_terms": body_terms}
    except (json.JSONDecodeError, AttributeError):
        return None


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
    """2問だけ生成して、中身とJSONパースを確認する"""
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
        raw = item["response"]["body"]["choices"][0]["message"]["content"].strip()
        doc = parse_termlist(raw)
        print(f"\n--- {item['custom_id']} ---")
        if doc is None:
            print("  [PARSE FAILED] raw output:")
            print(raw)
            continue
        print(f"  title_terms ({len(doc['title_terms'])}): {doc['title_terms']}")
        print(f"  heading_terms ({len(doc['heading_terms'])}): {doc['heading_terms']}")
        print(f"  body_terms ({len(doc['body_terms'])}): {doc['body_terms']}")


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
            raw_texts[item["custom_id"]] = \
                body["choices"][0]["message"]["content"].strip()

    decomposed = load_decomposed()
    results = {}
    n_parse_fail = 0
    for qid, entry in decomposed.items():
        dqs = entry["decomposed_queries"]
        structured = []
        for j in range(len(dqs)):
            raw = raw_texts.get(f"{qid}__q{j}")
            doc = parse_termlist(raw) if raw is not None else None
            if raw is not None and doc is None:
                n_parse_fail += 1
                print(f"  WARN {qid}__q{j}: JSON parse failed")
            structured.append(doc)
        n_missing = sum(1 for d in structured if d is None)
        if n_missing:
            print(f"  WARN {qid}: {n_missing}/{len(dqs)} 問で語リストが欠落")
        results[qid] = {
            "original_query": entry["original_query"],
            "decomposed_queries": dqs,
            # decomposed_queries[j] に対応する語リスト（{"title_terms","heading_terms","body_terms"}）
            "query2doc_docs_terms": structured,
        }

    json.dump({"config": store["config"], "results": results},
              open(OUT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"トピック数={len(results)}  文書数={len(raw_texts)}  "
          f"APIエラー={n_err}  JSONパース失敗={n_parse_fail}")
    print(f"課金額=${cost:.4f}")
    print(f"-> {OUT_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "poll"
    cmds = {"smoke": cmd_smoke, "submit": cmd_submit,
            "poll": cmd_poll, "fetch": cmd_fetch}
    if cmd not in cmds:
        raise SystemExit(f"使い方: python {sys.argv[0]} [smoke|submit|poll|fetch]")
    cmds[cmd]()
