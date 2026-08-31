"""
decompose_narrative.py が作った Decomposed_Query（narrative を分解した簡潔な質問文）
1問ごとに、Query2doc 疑似文書を1本だけ生成する。

decomposed_query2doc_webstyle_expansion.py との違い: プロンプトに「元のnarrativeが
どうDecomposed_Queryへ分解されたか」の文脈を含める。decomposed_query2doc_webstyle_expansion.py
はDecomposed_Query単体しか見せていなかったため、LLMがnarrative固有の文脈（ユーザーが
本当に知りたい広い意図・スコープ）を無視して一般論的な疑似文書を書きやすい、という
問題があった（evaluate_decomposed_webstyle.py の結果で、narrativeを使わないdq_pseudodoc系が
baselineよりrecall@1000で一貫して劣ったことから示唆される）。narrativeをプロンプトに
見せることで、疑似文書の内容をnarrativeの文脈に沿ってより具体的に狙わせるのが狙い。

出力フォーマットは decomposed_query2doc_webstyle_expansion.py と同じ（title/headings/body
のJSON → flattenしたテキストを "query2doc_docs" に格納）。

使い方:
    python decomposed_query2doc_webstyle_ctx_expansion.py smoke    # 2問だけ試す
    python decomposed_query2doc_webstyle_ctx_expansion.py submit   # 本番投入
    python decomposed_query2doc_webstyle_ctx_expansion.py poll     # 進捗確認
    python decomposed_query2doc_webstyle_ctx_expansion.py fetch    # 結果回収
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

WORDS_PER_DOC = 200

IN_FILE = "decomposed_queries.json"
CHUNK = 1000
IDS_FILE = "batch_ids_decomposed_query2doc_webstyle_ctx.json"
OUT_FILE = f"multi_query2doc_decomposed_webstyle_ctx_L{WORDS_PER_DOC}.json"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# プロンプト（few-shot、narrative文脈つき、JSON出力でtitle/headings/bodyを分けて生成させる）
# ============================================================
FEWSHOT_EXAMPLES = [
    {
        "narrative": (
            "I'm trying to understand climate systems more broadly, including how "
            "ocean processes, atmospheric chemistry, and human activity interact to "
            "shape long-term weather patterns. As background, I first want to know "
            "about tides specifically: what causes them, and how strong they can get "
            "in different coastal locations."
        ),
        "query": "What causes ocean tides, and how do the sun and moon each contribute?",
        "title": "What Causes Ocean Tides? The Role of the Moon and Sun",
        "headings": [
            "The Moon's gravitational pull",
            "The Sun's smaller contribution",
            "Spring tides and neap tides",
            "Local geography and tidal range",
        ],
        "body": (
            "Ocean tides are caused primarily by the gravitational pull of the Moon and, "
            "to a lesser extent, the Sun, acting on Earth's oceans. The Moon's gravity "
            "pulls seawater toward it, creating a bulge on the side of Earth facing the "
            "Moon, while inertial forces from Earth's rotation around the Earth-Moon "
            "system create a second bulge on the opposite side, producing two high tides "
            "and two low tides in most locations roughly every 24 hours and 50 minutes. "
            "The Sun also exerts a gravitational pull on the oceans, but because it is "
            "much farther away than the Moon, its tidal effect is only about 46 percent "
            "as strong. When the Sun, Moon, and Earth align during a new or full moon, "
            "their combined gravitational pull produces unusually high spring tides, "
            "while during the first and third quarter moons their forces partially "
            "cancel out, producing smaller neap tides. Local geography, including the "
            "shape of coastlines, the depth of the seafloor, and the width of bays and "
            "estuaries, can amplify or dampen these astronomical effects, causing some "
            "coastal regions to see tidal ranges of less than a foot and others, such as "
            "the Bay of Fundy, to experience swings exceeding fifteen meters."
        ),
    },
    {
        "narrative": (
            "I'm researching public health infrastructure: how immunization programs "
            "are funded, evaluated, and communicated to the public, and how trust in "
            "them is built or lost. As background biology, I want to understand how "
            "vaccines actually train the immune system to fight future infections."
        ),
        "query": "How do vaccines train the immune system to fight future infections?",
        "title": "How Vaccines Train the Immune System",
        "headings": [
            "Introducing a harmless pathogen component",
            "Activating T cells and B cells",
            "Antibodies and immune memory",
            "Faster response on re-exposure",
        ],
        "body": (
            "Vaccines work by introducing a harmless component of a pathogen, such as a "
            "weakened virus, an inactivated bacterium, or a piece of viral protein, into "
            "the body so that the immune system can learn to recognize it without "
            "causing full-blown disease. Once introduced, specialized immune cells "
            "called antigen-presenting cells capture the vaccine material and display "
            "fragments of it to T cells and B cells, triggering an adaptive immune "
            "response. B cells respond by producing antibodies that bind specifically "
            "to the pathogen's surface proteins, marking it for destruction, while "
            "helper T cells coordinate the broader immune response and killer T cells "
            "eliminate infected cells directly. Critically, some of these activated B "
            "and T cells become long-lived memory cells that persist in the body for "
            "years or even decades after vaccination. If the real pathogen is later "
            "encountered, these memory cells recognize it immediately and mount a much "
            "faster and stronger secondary immune response than would occur in an "
            "unvaccinated person, often neutralizing the infection before symptoms "
            "develop. This immunological memory explains why vaccinated individuals "
            "typically experience milder illness, or no illness at all, when exposed "
            "to diseases such as measles, polio, or influenza."
        ),
    },
    {
        "narrative": (
            "For a world history course project on the transition from antiquity to "
            "the medieval period across Eurasia, I need background on several "
            "civilizational collapses. I'll start with what led to the fall of the "
            "Western Roman Empire and how that decline actually unfolded over time."
        ),
        "query": "What led to the fall of the Western Roman Empire, and how did the decline unfold?",
        "title": "The Fall of the Western Roman Empire: Causes and Timeline",
        "headings": [
            "Political instability and civil wars",
            "Economic troubles and overextended borders",
            "Division into Western and Eastern halves",
            "Germanic invasions and the end in 476 CE",
        ],
        "body": (
            "The fall of the Western Roman Empire in 476 CE was the result of a long "
            "process of political, military, and economic decline stretching over "
            "several centuries rather than a single catastrophic event. Chronic civil "
            "wars and rapid turnover of emperors weakened central authority, while the "
            "empire's vast borders became increasingly difficult and expensive to "
            "defend against migrating and invading peoples, including the Goths, "
            "Vandals, and Huns. Economic troubles, including heavy taxation, inflation "
            "caused by currency debasement, and disruptions to trade routes, eroded the "
            "empire's ability to fund its armies and maintain infrastructure. The "
            "division of the empire into Western and Eastern halves in the late third "
            "and fourth centuries left the West with fewer resources and a smaller tax "
            "base than the wealthier, more urbanized East, which would survive as the "
            "Byzantine Empire for another thousand years. Migrations of Germanic "
            "peoples across the Rhine and Danube frontiers, driven in part by pressure "
            "from the Huns further east, overwhelmed Roman defenses, and Rome itself "
            "was sacked by the Visigoths in 410 and by the Vandals in 455. The formal "
            "end came in 476 when the Germanic general Odoacer deposed the last Western "
            "emperor, Romulus Augustulus, marking the traditional close of ancient Rome "
            "and the start of early medieval Europe."
        ),
    },
]


def _format_fewshot_block():
    blocks = []
    for ex in FEWSHOT_EXAMPLES:
        out = {"title": ex["title"], "headings": ex["headings"], "body": ex["body"]}
        blocks.append(
            f"Narrative (context): {ex['narrative']}\n"
            f"Sub-question to answer: {ex['query']}\n"
            f"{json.dumps(out, ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


FEWSHOT_BLOCK = _format_fewshot_block()

PROMPT = """The sub-question below is one of several focused questions that a
longer, multi-part research narrative was broken down into. Read the
narrative to understand the broader context, intent, and scope the user
cares about, but write the hypothetical web page to specifically and
directly answer the sub-question itself — stay anchored to that sub-question's
particular facet, and use the narrative only to sharpen and disambiguate it,
not to answer the other facets of the narrative.

Write the metadata and body text of a single hypothetical web page that
would answer the sub-question, as if you were looking at the title tag,
section headings, and article body of a real informative page on this
topic.

Output only a single JSON object with exactly these keys:
- "title": a short page-title-like phrase (under 12 words), the way a <title>
  tag or headline would read.
- "headings": a JSON array of 2 to 4 short section-heading phrases (each
  under 8 words), the way <h2> headings on the page would read.
- "body": approximately {words} words of body text, written as declarative
  sentences of fact in a single continuous paragraph with no line breaks
  (do not repeat the headings verbatim as labels inside the body).

Across the title, headings, and body, use many different paraphrases and
synonymous expressions for the key concepts in the sub-question instead of
reusing the same wording every time — vary terminology, phrasing, and word
forms (e.g. synonyms, related technical terms, singular/plural, noun/verb
forms) so that a reader searching with different wordings for the same idea
would still find a matching term somewhere in the page.

Write in English. Do not wrap the JSON in markdown code fences. Do not
include any text outside the JSON object.

Here are examples of the expected input/output style:

{examples}

Now write a new page in the same style for this sub-question:

Narrative (context): {narrative}
Sub-question to answer: {query}"""

CONFIG = {
    "model": MODEL,
    "reasoning": REASONING,
    "temperature": TEMPERATURE,
    "words_per_doc": WORDS_PER_DOC,
    "prompt_style": "few-shot-webstyle-narrative-context",
    "n_fewshot_examples": len(FEWSHOT_EXAMPLES),
    "prompt_hash": hashlib.sha256(PROMPT.encode()).hexdigest()[:16],
}


# ============================================================
# 入力の読み込み
# ============================================================
def load_decomposed():
    """{qid: {"original_query":..., "decomposed_queries": [...]}}"""
    return json.load(open(IN_FILE, encoding="utf-8"))["results"]


def make_body(narrative, query):
    return {
        "messages": [
            {"role": "user",
             "content": PROMPT.format(words=WORDS_PER_DOC, examples=FEWSHOT_BLOCK,
                                       narrative=narrative, query=query)}
        ],
        "reasoning": REASONING,
        "temperature": TEMPERATURE,
    }


def build_requests(decomposed):
    """1質問 = 1リクエスト。custom_id は '<qid>__q<番号>' 形式。"""
    reqs = []
    for qid, entry in decomposed.items():
        narrative = entry["original_query"]
        for j, dq in enumerate(entry["decomposed_queries"]):
            reqs.append({
                "custom_id": f"{qid}__q{j}",
                "body": make_body(narrative, dq),
            })
    return reqs


# ============================================================
# JSON応答のパース
# ============================================================
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_webdoc(raw_text):
    """LLM出力から {"title", "headings", "body"} を取り出す。壊れていたらNoneを返す。"""
    text = _FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        obj = json.loads(text)
        title = (obj.get("title") or "").strip()
        headings = [h.strip() for h in (obj.get("headings") or []) if isinstance(h, str) and h.strip()]
        body = (obj.get("body") or "").strip()
        if not body:
            return None
        return {"title": title, "headings": headings, "body": body}
    except (json.JSONDecodeError, AttributeError):
        return None


def flatten_webdoc(doc):
    """title + headings + body を1本のテキストに連結する（bm25_body用）。"""
    parts = []
    if doc["title"]:
        parts.append(doc["title"])
    parts.extend(doc["headings"])
    parts.append(doc["body"])
    return "\n".join(parts)


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
    narrative = (
        "I'm studying how search engines find relevant documents, from classical "
        "statistical scoring methods to modern neural approaches, and how the two "
        "families of methods are often combined in practice."
    )
    reqs = [
        {"custom_id": "smoke__q0", "body": make_body(narrative, "what is BM25 ranking")},
        {"custom_id": "smoke__q1", "body": make_body(narrative, "what is dense retrieval")},
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
        doc = parse_webdoc(raw)
        print(f"\n--- {item['custom_id']} ---")
        if doc is None:
            print("  [PARSE FAILED] raw output:")
            print(raw)
            continue
        print(f"  title:    {doc['title']}")
        print(f"  headings: {doc['headings']}")
        print(f"  body ({len(doc['body'].split())} words): {doc['body'][:300]}...")


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
        structured, flat = [], []
        for j in range(len(dqs)):
            raw = raw_texts.get(f"{qid}__q{j}")
            doc = parse_webdoc(raw) if raw is not None else None
            if raw is not None and doc is None:
                n_parse_fail += 1
                print(f"  WARN {qid}__q{j}: JSON parse failed")
            structured.append(doc)
            flat.append(flatten_webdoc(doc) if doc else None)
        n_missing = sum(1 for d in flat if d is None)
        if n_missing:
            print(f"  WARN {qid}: {n_missing}/{len(dqs)} 問で疑似文書が欠落")
        results[qid] = {
            "original_query": entry["original_query"],
            "decomposed_queries": dqs,
            # decomposed_queries[j] に対応する疑似文書（title+headings+bodyを連結したフラットテキスト）
            "query2doc_docs": flat,
            # 同じ疑似文書のフィールド構造版（bm25_keyterms等で使う場合用）
            "query2doc_docs_structured": structured,
            "word_counts": [len(d.split()) if d else 0 for d in flat],
        }

    json.dump({"config": store["config"], "results": results},
              open(OUT_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    wc = sorted(w for r in results.values() for w in r["word_counts"] if w)
    if wc:
        n = len(wc)
        print(f"\n生成語長: 中央値={wc[n // 2]}, "
              f"25%={wc[n // 4]}, 75%={wc[3 * n // 4]}  (目標={WORDS_PER_DOC})")
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
