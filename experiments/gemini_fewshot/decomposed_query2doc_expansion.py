"""
decompose_narrative.py が作った Subquery（narrative を分解した簡潔な質問文）
1問ごとに、Query2doc 疑似文書を1本だけ生成する。

Query2doc論文の本来の設定に合わせ、few-shot（(query, document)の例を3件プロンプトに
含める）で生成する。zero-shot版との比較実験用。

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
# プロンプト（few-shot版。narrative_expansion.py は zero-shot のまま）
# ============================================================
# Query2doc論文の few-shot 設定にならい、(query, document) の例を3件与える。
# 実際の評価対象トピック（TREC 2025 RAG）とは無関係な話題を選んでいる。
FEWSHOT_EXAMPLES = [
    {
        "query": "What causes ocean tides, and how do the sun and moon each contribute?",
        "document": (
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
            "regions to experience tidal ranges of less than a foot and others, such as "
            "the Bay of Fundy, to see ranges exceeding fifteen meters."
        ),
    },
    {
        "query": "How do vaccines train the immune system to fight future infections?",
        "document": (
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
            "develop. This is the basis of immunological memory, and it explains why "
            "vaccinated individuals typically experience milder illness or no illness "
            "at all when exposed to diseases such as measles, polio, or influenza."
        ),
    },
    {
        "query": "What led to the fall of the Western Roman Empire, and how did the decline unfold?",
        "document": (
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
            "and the beginning of early medieval Europe."
        ),
    },
]


def _format_fewshot_block():
    blocks = [f"Search query: {ex['query']}\nDocument: {ex['document']}"
              for ex in FEWSHOT_EXAMPLES]
    return "\n\n".join(blocks)


FEWSHOT_BLOCK = _format_fewshot_block()

PROMPT = """Write one hypothetical document (a single paragraph) that would
answer the following search query. Target length: approximately {words} words.

Write in English. Start immediately with the body text as declarative sentences
of fact. Do not include a title, heading, or any label ending in a colon. Do not
use numbering, bullets, or prefixes such as "Document 1:". Write the document as
a single continuous paragraph with no line breaks.

Here are examples of the expected input/output style:

{examples}

Now write a new document in the same style for this query:

Search query: {query}"""

CONFIG = {
    "model": MODEL,
    "reasoning": REASONING,
    "temperature": TEMPERATURE,
    "words_per_doc": WORDS_PER_DOC,
    "prompt_style": "few-shot",
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
             "content": PROMPT.format(words=WORDS_PER_DOC, examples=FEWSHOT_BLOCK, query=query)}
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
