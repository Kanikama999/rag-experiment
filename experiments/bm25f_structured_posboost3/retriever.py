import math

from opensearchpy import OpenSearch
from collections import defaultdict

client = OpenSearch("http://localhost:9200", timeout=180, max_retries=2, retry_on_timeout=True)
INDEX = "msmarco-v21-doc"
QREL_POOL_INDEX = "qrel-segment-pool"

def bm25_body(query_text, k=100):
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "match": {
                "body": {
                    "query": query_text
                }
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]

def bm25_title(query_text, k=100):
    """titleフィールドだけをmatchする（フィールド別寄与の切り分け用）。"""
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "match": {
                "title": {
                    "query": query_text
                }
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]

def bm25_headings(query_text, k=100):
    """headingsフィールドだけをmatchする（フィールド別寄与の切り分け用）。"""
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "match": {
                "headings": {
                    "query": query_text
                }
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def bm25_url(query_text, k=100):
    """urlフィールドだけをmatchする（フィールド別寄与の切り分け用）。"""
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": ["url"],
        "query": {
            "wildcard": {
                "url": {
                    "value": f"*{query_text}*"
                }
            }
        }
    })
    return [(h["_id"], h["_score"], h["_source"]["url"]) for h in res["hits"]["hits"]]

if __name__ == "__main__":
    print("読み込み中...")
    print(client.indices.get_mapping(index=INDEX))
    query_text= "love"
    result = bm25_url(query_text, k=10)
    print(result)


def bm25_keyterms(query_text, k=100):
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["title^3", "headings^2", "body^1"]
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]

def bm25_fielded(title_text, headings_text, body_text, k=100,
                  title_boost=3, headings_boost=2, body_boost=1):
    """title/headings/bodyそれぞれ別のテキストを対応するフィールドにmatchし、
    bool/shouldで線形和（デフォルトはtitle^3 + headings^2 + body^1）にして検索する。
    bm25_keytermsが同じquery_textを3フィールドにぶつけてmax(best_fields)を取るのに対し、
    こちらはフィールドごとに別々のテキストを渡し、3フィールド分のスコアを全て合算する。
    ブースト値はtitle_boost/headings_boost/body_boostで変更できる（比率の見直し用）。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "bool": {
                "should": should
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def bm25_crossfields(query_text, k=100, title_boost=2, headings_boost=1, body_boost=1):
    """title/headings/bodyを1つの仮想フィールドとして扱い、TF/IDF統計を合算してから
    1回のBM25計算をする（multi_match type=cross_fields）。bm25_fieldedが各フィールド
    独立にBM25を計算してからスコアを線形和するのに対し、こちらは統計そのものを
    フィールド横断で合算する点が構造的に異なる（「titleにもbodyにも同じ語が出てくる」
    ときの評価のされ方が変わる）。bm25_fieldedと違い同じquery_textを3フィールドに
    投げる（best_fieldsのbm25_keytermsと同様の使い方）。"""
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "multi_match": {
                "query": query_text,
                "type": "cross_fields",
                "fields": [f"title^{title_boost}", f"headings^{headings_boost}", f"body^{body_boost}"]
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def bm25_fielded_msm(title_text, headings_text, body_text, k=100,
                      title_boost=2, headings_boost=1, body_boost=1,
                      minimum_should_match="30%"):
    """bm25_fieldedにminimum_should_matchを追加した版。各フィールドのmatch節で
    一定割合以上の語が一致しないとそのフィールドはスコアに寄与しない（デフォルトのOR
    一致だと1語当たっただけでもスコアが乗るため、弱いノイズ的一致を締める狙い）。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost,
                                             "minimum_should_match": minimum_should_match}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost,
                                                "minimum_should_match": minimum_should_match}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost,
                                            "minimum_should_match": minimum_should_match}}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "bool": {
                "should": should
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def analyze_terms(text, field="body", index=None):
    """textをそのフィールドのanalyzer（english、stopword除去+stemming）でトークン化し、
    重複を除いた語のリストを順序維持で返す（span_termクエリの構築用）。"""
    index = index or INDEX
    res = client.indices.analyze(index=index, body={"field": field, "text": text})
    seen, out = set(), []
    for t in res.get("tokens", []):
        tok = t["token"]
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def bm25_fielded_posboost(title_text, headings_text, body_text, span_terms, k=100,
                           title_boost=2, headings_boost=1, body_boost=1,
                           span_end=50, span_boost=2.0):
    """bm25_fieldedに、span_termsのいずれかが文書bodyの先頭span_end語以内に出現したら
    加点するspan_firstクエリを追加した版。BM25はTF/IDFと文書長しか見ず「文書内のどこに
    出現するか」を無視するため、その情報を補う狙い。span_termsは事前にanalyze_terms()で
    トークン化した語のリスト（重複除去済み、通常はSubquery本文など短いテキストから作る。
    長い繰り返しクエリをそのまま使うとspan_orの節数が増えて重くなるため）。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    if span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": span_end, "boost": span_boost}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "bool": {
                "should": should
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


DISCOURSE_SUMMARY_MARKERS = ["summari", "summar", "conclus", "conclud", "overal", "short"]
"""「in summary/to summarize/in conclusion/to conclude/overall/in short」等の
要約系discourse markerを、english analyzerでステミングした語形（重複含む語幹違い）。
"summary"→"summari", "summarize"→"summar" のように同じ語族でもステミング結果が
割れることがあるため、両方登録してある。"""

DISCOURSE_ABSTRACTION_MARKERS = DISCOURSE_SUMMARY_MARKERS + [
    "overview", "gener", "essenti", "fundament", "broadli", "typic",
]
"""DISCOURSE_SUMMARY_MARKERSに、「概要/一般に/本質的に/根本的に/大まかに/典型的に」等の
抽象化・一般化を示すdiscourse markerを追加した拡張版
（overview/in general・generally/essentially/fundamentally/broadly/typically）。
"basically"→"basic"、"at its core"→"core" は他の文脈でも高頻度に出現しすぎる
（曖昧な一致が増える）と判断して除外した。"""

DISCOURSE_HEADING_MARKERS = DISCOURSE_SUMMARY_MARKERS + [
    "overview", "highlight", "introduct", "faq", "takeawai",
]
"""DISCOURSE_SUMMARY_MARKERSに、webページの見出しラベルとしてよく使われる語
（Overview/Highlights/Introduction/FAQ/Key Takeawaysの"takeaways"部分）を追加した版。
候補にあった"About"/"Features"/"Benefits"/"How it Works"/"Getting Started"/
"Requirements"/"Important Information"/"Key"は、構成語の文書頻度が15〜63%と極めて
高く（コーパス全体の1割〜6割超の文書に出現）、見出しとしての識別力がほぼないため
除外した（overview=6.8%, highlight=5.0%, introduct=4.9%, faq=4.8%,
takeawai=0.7%と、既存markerと同程度以下の頻度に絞ってある）。"""


def bm25_fielded_discourseboost(title_text, headings_text, body_text, span_terms, k=100,
                                 title_boost=2, headings_boost=1, body_boost=1,
                                 markers=None, slop=20, marker_boost=3.0):
    """bm25_fieldedに、要約系discourse marker（"in summary"等）の直後（slop語以内、
    順序固定）にspan_termsのいずれかが出現したら加点するspan_nearクエリを追加した版。
    posboost（文書の先頭かどうか）とは別の切り口で、「文書内の構造的に重要そうな位置
    （要約・結論部）かどうか」を見る。markers未指定ならDISCOURSE_SUMMARY_MARKERSを使う。
    span_termsは事前にanalyze_terms()でトークン化した語のリスト（通常はSubquery本文
    など短いテキストから作る）。"""
    markers = markers if markers is not None else DISCOURSE_SUMMARY_MARKERS
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    if span_terms and markers:
        span_near_clauses = [
            {"span_near": {"clauses": [{"span_term": {"body": mk}}, {"span_term": {"body": t}}],
                            "slop": slop, "in_order": True}}
            for mk in markers for t in span_terms
        ]
        should.append({"bool": {"should": span_near_clauses, "boost": marker_boost}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "bool": {
                "should": should
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


# 命名について: 以下の bm25_equalweight_* 群は、title/headings/body を
# title_boost=headings_boost=body_boost=1 の「均等重みの線形和」で検索する系列である。
# 旧名は bm25_notitle_* だったが、titleフィールドを使わないという誤解を招くため
# 2026-09-09 に equalweight へ改名した（挙動は変更なし。titleは重み1で使っている）。
def bm25_equalweight_posboost_discourseboost(title_text, headings_text, body_text, span_terms, k=100,
                                          span_end=100, span_boost=15.0,
                                          markers=None, slop=20, marker_boost=3.0):
    """title/headings/bodyを均等（title_boost=headings_boost=body_boost=1、フィールド別の
    重み付けなし）にした上で、posboost（span_first、文書先頭span_end語以内）と
    discourseboost（span_near、要約系discourse markerの直後）の両方を足した版。
    「title重み付けというフィールド構造のチューニングをせずに、位置とdiscourse markerだけで
    どこまで伸ばせるか」を見る実験用。"""
    markers = markers if markers is not None else DISCOURSE_SUMMARY_MARKERS
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": 1}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": 1}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": 1}}})
    if span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": span_end, "boost": span_boost}})
        if markers:
            span_near_clauses = [
                {"span_near": {"clauses": [{"span_term": {"body": mk}}, {"span_term": {"body": t}}],
                                "slop": slop, "in_order": True}}
                for mk in markers for t in span_terms
            ]
            should.append({"bool": {"should": span_near_clauses, "boost": marker_boost}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "bool": {
                "should": should
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def bm25_fielded_posboost_discourseboost(title_text, headings_text, body_text, span_terms, k=100,
                                          title_boost=2, headings_boost=1, body_boost=1,
                                          span_end=100, span_boost=15.0,
                                          markers=None, slop=20, marker_boost=3.0):
    """bm25_equalweight_posboost_discourseboostの一般化版。title/headings/bodyのフィールド
    ブースト（デフォルトはrecallopt: title=2,headings=1,body=1）を保ったまま、posboost・
    discourseboostの両方を足す。「フィールドブーストとposboost/discourseboostは併用した
    方が良いのか、それともtitle均等化（bm25_equalweight_posboost_discourseboost）の方が
    良いのか」を比較する実験用。"""
    markers = markers if markers is not None else DISCOURSE_SUMMARY_MARKERS
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    if span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": span_end, "boost": span_boost}})
        if markers:
            span_near_clauses = [
                {"span_near": {"clauses": [{"span_term": {"body": mk}}, {"span_term": {"body": t}}],
                                "slop": slop, "in_order": True}}
                for mk in markers for t in span_terms
            ]
            should.append({"bool": {"should": span_near_clauses, "boost": marker_boost}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "bool": {
                "should": should
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def bm25_fielded_posboost_normalized(title_text, headings_text, body_text, span_terms, k=100,
                                      title_boost=1, headings_boost=1, body_boost=1,
                                      span_end=100, boost_ratio=0.02, ref_topn=10):
    """bm25_fielded_posboostの正規化版。span_boostを固定値ではなく、ベースfielded検索
    （posboostなし、title/headings/bodyのmatchのみ）の上位ref_topn件のスコア平均に対する
    比率(boost_ratio)で動的に計算する。bm25_body_posboost_normalizedのfielded版
    （Tier C: narrative×N+Subquery+疑似文書のような長いクエリ向け）。2段階検索になる。"""
    def _base_should():
        should = []
        if title_text and title_text.strip():
            should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
        if headings_text and headings_text.strip():
            should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
        if body_text and body_text.strip():
            should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
        return should

    base_res = client.search(index=INDEX, body={
        "size": max(k, ref_topn),
        "_source": False,
        "query": {"bool": {"should": _base_should()}}
    })
    base_hits = base_res["hits"]["hits"]
    if not base_hits:
        return []
    ref_scores = [h["_score"] for h in base_hits[:ref_topn]]
    reference_score = sum(ref_scores) / len(ref_scores)
    span_boost = boost_ratio * reference_score

    should = _base_should()
    if span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": span_end, "boost": span_boost}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def bm25_body_posboost_normalized(query_text, span_terms, k=100,
                                   span_end=100, boost_ratio=0.02, ref_topn=10):
    """posboostのspan_boostを固定値ではなく、ベースmatch検索（posboostなし）の
    上位ref_topn件のスコア平均に対する比率(boost_ratio)で動的に計算する正規化版。

    クエリ長対照実験で判明した問題: 固定span_boostは、BM25生スコアの絶対値スケールが
    クエリによって大きく異なる（短いクエリで上位スコアが15〜22、長いクエリ(narrative×5+
    Subquery+疑似文書)で590〜986）ため、長いクエリ用にチューニングした固定値（15.0）を
    短いクエリに使うと、posboostのボーナスだけでベーススコアを凌駕してランキングを
    破壊してしまう（実測: 短いクエリでは固定15がベーススコアの68〜100%、長いクエリ
    では1.5〜2.5%程度にしかならない）。boost_ratioで比率を固定すれば、クエリ長に
    関係なく同じ相対的な影響力になるはず、という仮説を検証する。

    2段階検索になる（まずposboostなしでベーススコアの規模を推定し、それを使って
    再検索する）ため、通常のposboost関数よりコストが2倍程度かかる。"""
    base_res = client.search(index=INDEX, body={
        "size": max(k, ref_topn),
        "_source": False,
        "query": {"match": {"body": {"query": query_text, "boost": 1}}}
    })
    base_hits = base_res["hits"]["hits"]
    if not base_hits:
        return []
    ref_scores = [h["_score"] for h in base_hits[:ref_topn]]
    reference_score = sum(ref_scores) / len(ref_scores)
    span_boost = boost_ratio * reference_score

    should = [{"match": {"body": {"query": query_text, "boost": 1}}}]
    if span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": span_end, "boost": span_boost}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def bm25_bodyonly_posboost_discourseboost(body_text, span_terms, k=100,
                                           span_end=100, span_boost=15.0,
                                           markers=None, slop=20, marker_boost=3.0):
    """title/headingsフィールドへのマッチを一切せず、bodyフィールドへのmatch＋posboost＋
    discourseboostだけにした版。「title/headingsに構造化して別々の実文書フィールドへ
    マッチさせること自体に意味があるのか、それともbodyへのposboost/discourseboostだけで
    説明がつくのか」を切り分ける実験用。body_textには疑似文書をflatten（title+headings+
    bodyを連結）したテキストを渡す想定。"""
    markers = markers if markers is not None else DISCOURSE_SUMMARY_MARKERS
    should = []
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": 1}}})
    if span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": span_end, "boost": span_boost}})
        if markers:
            span_near_clauses = [
                {"span_near": {"clauses": [{"span_term": {"body": mk}}, {"span_term": {"body": t}}],
                                "slop": slop, "in_order": True}}
                for mk in markers for t in span_terms
            ]
            should.append({"bool": {"should": span_near_clauses, "boost": marker_boost}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "bool": {
                "should": should
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def bm25_fielded_weighted(plain_text, title_words, headings_words, body_words, k=100,
                           title_boost=3, headings_boost=2, body_boost=1):
    """bm25_fieldedの拡張版。各フィールドについて、
    (1) narrative+Subquery等の"素の"クエリ文字列(plain_text)を通常通りmatch(boost=field_boost)、
    (2) 疑似文書側の生成語は単語ごとに個別のmatch(boost=field_boost*weight)にする。
    title_words/headings_words/body_words は [(word, weight), ...]（weightの合計は1、
    term_weights.extract_term_weights()の出力）。単語自体はステミングしていない表層形のまま
    渡し、アナライザ（english）による小文字化・ステミングはOpenSearch側に任せる。"""
    def field_clauses(field, boost, words):
        clauses = []
        if plain_text and plain_text.strip():
            clauses.append({"match": {field: {"query": plain_text, "boost": boost}}})
        for word, weight in words:
            clauses.append({"match": {field: {"query": word, "boost": boost * weight}}})
        return clauses

    should = (field_clauses("title", title_boost, title_words)
              + field_clauses("headings", headings_boost, headings_words)
              + field_clauses("body", body_boost, body_words))
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "bool": {
                "should": should
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def _span_or(span_terms):
    return {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}


def _cascade_clauses(span_terms, tiers):
    """累積多段窓。tiers = [(end, boost), ...]。窓は入れ子なので、先頭で当たった語ほど
    多くの段に同時に入り、加点が積み上がる（位置減衰関数の階段近似）。

    注意: 各段の寄与は boost の定数ではない。span_first は「窓内のspan頻度」を tf として
    BM25 で採点するため（2026-09-09 実測: 同一文書で end=50 → freq 15、end=100 → freq 28、
    end=400 → freq 145）、窓を広げるほど freq が増えてスコアも上がる方向に働く。
    したがって boost を単調減少させても、実効的な寄与が単調減少するとは限らない。
    段ごとの重みは理論値ではなく実測で決めること。

    また累積窓は入れ子ゆえに、窓が全文を覆ってしまう短い文書が全段を総取りする
    （2026-09-09 実測: body語数の中央値は938語で、end=400 で16%、end=800 で43%の文書が
    全文包含になる）。この文書長バイアスを消したい場合は _ring_clauses を使う。"""
    return [{"span_first": {"match": _span_or(span_terms), "end": end, "boost": boost}}
            for end, boost in tiers]


def _ring_clauses(span_terms, tiers):
    """リング窓。tiers = [(end, boost), ...] を昇順に受け取り、区間 [前のend, end) の
    disjoint なリングに分解して、それぞれに boost を与える。

    リングは span_not(include=span_first(end=b), exclude=span_first(end=a)) で作る
    （span_not は include のspanのうち exclude のspanと重なるものを落とすので、
    位置 a 未満のマッチが除かれ、[a, b) のマッチだけが残る）。

    累積窓との違いは、区間が排他になること。80語の短い文書は [200,400) のリングに
    原理的にマッチできないので、全段の加点を総取りできない。
    _cascade_clauses の全文包含バイアスに対する直接的な対処である。"""
    clauses, prev = [], 0
    for end, boost in tiers:
        inner = {"span_first": {"match": _span_or(span_terms), "end": end}}
        if prev == 0:
            clauses.append({"span_first": {"match": _span_or(span_terms),
                                            "end": end, "boost": boost}})
        else:
            clauses.append({"span_not": {
                "include": inner,
                "exclude": {"span_first": {"match": _span_or(span_terms), "end": prev}},
                "boost": boost}})
        prev = end
    return clauses


def bm25_equalweight_multiwindow(title_text, headings_text, body_text, span_terms, k=100,
                                  tiers=None, mode="cascade"):
    """bm25_equalweight_posboost_discourseboost(markers=[]) の位置ブースト部分を、
    単一窓 span_first から「多段窓」に置き換えた版。title/headings/body は均等重み(1:1:1)。

    tiers: [(end, boost), ...] を end の昇順で。既定は現行championと総ブースト量を
           揃えた4段（合計15）。
    mode : "cascade" = 入れ子の累積窓（_cascade_clauses）
           "ring"    = 排他のリング窓（_ring_clauses）
           "single"  = 従来どおり1段だけ（tiers の最後の1つを使う。対照用）
    """
    tiers = tiers if tiers is not None else [(50, 8.0), (100, 4.0), (200, 2.0), (400, 1.0)]
    should = []
    for field, text in (("title", title_text), ("headings", headings_text),
                        ("body", body_text)):
        if text and text.strip():
            should.append({"match": {field: {"query": text, "boost": 1}}})
    if span_terms and tiers:
        if mode == "cascade":
            should += _cascade_clauses(span_terms, tiers)
        elif mode == "ring":
            should += _ring_clauses(span_terms, tiers)
        elif mode == "single":
            end, boost = tiers[-1]
            should.append({"span_first": {"match": _span_or(span_terms),
                                           "end": end, "boost": boost}})
        else:
            raise ValueError(f"未知の mode: {mode}")
    res = client.search(index=INDEX, body={
        "size": k, "_source": False, "query": {"bool": {"should": should}}})
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def bm25_fielded_weighted_posboost(plain_text, title_words, headings_words, body_words,
                                   span_terms, k=100,
                                   title_boost=1, headings_boost=1, body_boost=1,
                                   span_end=100, span_boost=15.0):
    """bm25_fielded_weighted（疑似文書側の語をterm単位で重み付きmatchする）に、
    bm25_equalweight_posboost_discourseboost と同じ span_first の位置ブーストを足した版。

    デフォルト引数は均等重み(1:1:1)・span_end=100・span_boost=15で、champion
    （bm25_equalweight_posboost_discourseboost(markers=[])）と同じ検索条件の上に
    「term単位の重み付け」だけを乗せた形になる。span_boost<=0 または span_terms が空の
    ときは位置ブースト節を付けないので、同じ関数のまま位置ブーストのon/offを切り替えられる。

    title_words/headings_words/body_words は [(word, weight), ...]。全weightを1.0にすれば
    「term単位に分解はするが重み付けはしない」対照条件になる（bm25_fieldedはフィールド全体を
    1つのmatch節にまとめるためTFの効き方が違い、重み付けの効果だけを見る対照にはならない。
    そのため2x2の対照はこの関数のweight=1.0側で取ること）。"""
    def field_clauses(field, boost, words):
        clauses = []
        if plain_text and plain_text.strip():
            clauses.append({"match": {field: {"query": plain_text, "boost": boost}}})
        for word, weight in words:
            clauses.append({"match": {field: {"query": word, "boost": boost * weight}}})
        return clauses

    should = (field_clauses("title", title_boost, title_words)
              + field_clauses("headings", headings_boost, headings_words)
              + field_clauses("body", body_boost, body_words))
    if span_terms and span_boost > 0:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": span_end, "boost": span_boost}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {"bool": {"should": should}},
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]


def term_idf(word, field="body", index=None):
    """wordをそのフィールドのanalyzerで解析し、BM25型IDFを返す。
    インデックスに存在しない疑似ドキュメントを_termvectorsに投げることで、
    その語のdoc_freq/doc_countを取得する（実際に索引には残らない）。
    注意: _termvectorsは（doc指定時に明示的なroutingが無いため）実際にはある1つの
    シャードだけの統計を返す（8シャード構成なので母数はコーパス全体の目安、約1/8）。
    複数語間の相対的な大小比較（rare/commonの判定）としては十分実用的だが、
    絶対値としての厳密なコーパス全体IDFではない点に留意すること。
    wordが解析の結果ストップワード等で消える場合は0.0を返す。"""
    index = index or INDEX
    res = client.termvectors(index=index, body={
        "doc": {field: word},
        "fields": [field],
        "field_statistics": True,
        "term_statistics": True,
        "positions": False,
        "offsets": False,
        "payloads": False,
    })
    tv = res.get("term_vectors", {}).get(field)
    if not tv or not tv.get("terms"):
        return 0.0
    doc_count = tv["field_statistics"]["doc_count"]
    # 1語が複数termに分割される場合（複合語等）は最大doc_freq(=最も一般的なterm)を使う。
    # dfs指定時でもtermによってはdoc_freqが返らないことがあるので0扱いにする。
    doc_freqs = [t.get("doc_freq", 0) for t in tv["terms"].values()]
    if not doc_freqs or max(doc_freqs) == 0:
        return 0.0
    doc_freq = max(doc_freqs)
    return math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


def bm25_fielded_posboost_consistent(title_text, headings_text, body_text, span_terms, k=100,
                                      title_boost=1, headings_boost=1, body_boost=1,
                                      span_end=100, bonus_tf=5.0, rerank_n=100,
                                      bm25_k1=0.9, bm25_b=0.4):
    """posboostを外付けの加点クエリ（span_first + 固定/正規化boost）としてではなく、
    BM25の計算式そのものの内側（実効TF）に組み込んだ版。文書bodyの先頭span_end語以内に
    span_termsのいずれかの最初の出現があれば、そのtermの実効TFに bonus_tf を加算して
    BM25のtf飽和・IDF計算をやり直す。位置ボーナスがBM25自身のIDF・TF飽和ロジックを
    通るため、外付けのboost値をクエリのスコアスケールに合わせて正規化する必要が
    理論上ないはず、という設計（クエリ長対照実験でbm25_fielded_posboostの固定boostが
    クエリ長で破綻することが分かったことを受けた再定式化）。

    2段階処理: (1) 通常のfielded検索（posboostなし）で上位k件を取得、(2) 上位rerank_n件
    についてbodyフィールドのterm vectors（TF・出現位置・文書長）を取得し、位置ボーナスを
    反映したBM25スコアとの差分(delta)を元のスコアに加算して並べ替える。bm25_k1/bm25_bは
    このインデックスの実際の類似度設定に合わせた値（explain APIで確認済み、
    デフォルトのk1=1.2, b=0.75ではない）。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    hits = [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]
    if not span_terms or not hits:
        return hits

    rerank_ids = [docid for docid, _ in hits[:rerank_n]]
    orig_scores = {docid: score for docid, score in hits}

    # グローバル統計（N, avgdl, 各span_termのdf）を仮想文書1回の呼び出しでまとめて取得
    tv_global = client.termvectors(index=INDEX, body={
        "doc": {"body": " ".join(span_terms)},
        "fields": ["body"],
        "field_statistics": True,
        "term_statistics": True,
        "positions": False,
    })
    body_tv_global = tv_global.get("term_vectors", {}).get("body")
    if not body_tv_global:
        return hits
    fs = body_tv_global["field_statistics"]
    N = fs["doc_count"]
    avgdl = fs["sum_ttf"] / fs["doc_count"] if fs["doc_count"] else 1.0
    dfs = {t: info.get("doc_freq", 0) for t, info in body_tv_global["terms"].items()}

    def idf(term):
        n = dfs.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    def tf_norm(freq, dl):
        if freq <= 0:
            return 0.0
        return freq / (freq + bm25_k1 * (1 - bm25_b + bm25_b * dl / avgdl))

    # 候補文書ごとのterm vectors（TF・位置。統計情報は上で取得済みなので付けない＝高速）
    tv_docs = client.mtermvectors(index=INDEX, body={
        "ids": rerank_ids,
        "parameters": {
            "fields": ["body"],
            "term_statistics": False,
            "field_statistics": False,
            "positions": True,
            "offsets": False,
            "payloads": False,
        }
    })

    adjusted = dict(orig_scores)
    for doc in tv_docs.get("docs", []):
        docid = doc.get("_id")
        body_tv = doc.get("term_vectors", {}).get("body")
        if not body_tv or docid not in orig_scores:
            continue
        terms_info = body_tv["terms"]
        dl = sum(t["term_freq"] for t in terms_info.values()) or 1
        delta = 0.0
        for t in span_terms:
            info = terms_info.get(t)
            tf = info["term_freq"] if info else 0
            if info and info.get("tokens"):
                first_pos = min(tok["position"] for tok in info["tokens"])
            else:
                first_pos = None
            bonus = bonus_tf if (first_pos is not None and first_pos < span_end) else 0.0
            if tf == 0 and bonus == 0.0:
                continue
            score_orig_t = idf(t) * tf_norm(tf, dl)
            score_adj_t = idf(t) * tf_norm(tf + bonus, dl)
            delta += (score_adj_t - score_orig_t) * body_boost
        adjusted[docid] = orig_scores[docid] + delta

    final = sorted(((docid, adjusted.get(docid, score)) for docid, score in hits), key=lambda x: -x[1])
    return final


def bm25_fielded_posboost_idfnorm(title_text, headings_text, body_text, span_terms, k=100,
                                   title_boost=1, headings_boost=1, body_boost=1,
                                   span_end=100, boost_ratio=0.05, ref_topn=10, rerank_n=100):
    """bm25_fielded_posboost_consistentの反省を踏まえた改訂版。

    posboost_consistentでは、位置ボーナスをBM25の実効TF（tf+bonus_tf）としてtf_norm
    関数に通していたが、これは「1語あたりの最大加点 = idf(t) * (1 - tf_norm(元のtf))」
    という固定上限（TF飽和のceiling）に本質的に縛られることが数値実験で判明した
    （典型的な語でidf=1〜4、既にtf>0の語ではさらに小さい上限になり、bonus_tfをいくら
    大きくしてもこの上限を超えられない。実測でもnDCG@10の改善は最大+4%程度に留まり、
    正規化boost（+30〜94%）とは桁違いだった）。

    この版では、位置ボーナスをtf_norm関数の内側に押し込むのをやめ、「どのtermに
    どれだけ配分するか」だけBM25のidf哲学（情報量の少ない一般語より、情報量の多い
    レア語の位置的一致をより重視する）に従わせ、「全体としてどれだけ強くするか」は
    bm25_fielded_posboost_normalizedと同じ仕組み（ベース検索上位ref_topn件の平均
    スコアに対する比率boost_ratio）で決める。すなわち、あるterm tがdoc dの先頭
    span_end語以内に出現していれば、そのtermのidf(t)に比例した配分で
    boost_ratio * reference_scoreを山分けする：

        bonus(d) = boost_ratio * reference_score
                   * sum(idf(t) for t in span_terms if 先頭span_end語以内に出現)
                   / sum(idf(t) for t in span_terms)

    これはTF飽和のceilingに縛られない（tf_normを経由しないため）ので、正規化
    boostと同等の効果量を狙いつつ、「どのtermが重要か」の判断だけはBM25自身の
    idf計算に委ねる、という妥協案。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    res = client.search(index=INDEX, body={
        "size": max(k, ref_topn),
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    all_hits = [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]
    if not all_hits:
        return []
    hits = all_hits[:k]
    if not span_terms:
        return hits

    ref_scores = [score for _, score in all_hits[:ref_topn]]
    reference_score = sum(ref_scores) / len(ref_scores) if ref_scores else 0.0

    tv_global = client.termvectors(index=INDEX, body={
        "doc": {"body": " ".join(span_terms)},
        "fields": ["body"],
        "field_statistics": True,
        "term_statistics": True,
        "positions": False,
    })
    body_tv_global = tv_global.get("term_vectors", {}).get("body")
    if not body_tv_global:
        return hits
    fs = body_tv_global["field_statistics"]
    N = fs["doc_count"]
    dfs = {t: info.get("doc_freq", 0) for t, info in body_tv_global["terms"].items()}

    def idf(term):
        n = dfs.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    term_idfs = {t: idf(t) for t in span_terms}
    idf_sum = sum(term_idfs.values())
    if idf_sum <= 0 or reference_score <= 0:
        return hits

    rerank_ids = [docid for docid, _ in hits[:rerank_n]]
    orig_scores = {docid: score for docid, score in hits}

    tv_docs = client.mtermvectors(index=INDEX, body={
        "ids": rerank_ids,
        "parameters": {
            "fields": ["body"],
            "term_statistics": False,
            "field_statistics": False,
            "positions": True,
            "offsets": False,
            "payloads": False,
        }
    })

    adjusted = dict(orig_scores)
    for doc in tv_docs.get("docs", []):
        docid = doc.get("_id")
        body_tv = doc.get("term_vectors", {}).get("body")
        if not body_tv or docid not in orig_scores:
            continue
        terms_info = body_tv["terms"]
        triggered_idf = 0.0
        for t in span_terms:
            info = terms_info.get(t)
            if not info or not info.get("tokens"):
                continue
            first_pos = min(tok["position"] for tok in info["tokens"])
            if first_pos < span_end:
                triggered_idf += term_idfs.get(t, 0.0)
        if triggered_idf <= 0:
            continue
        bonus = boost_ratio * reference_score * (triggered_idf / idf_sum) * body_boost
        adjusted[docid] = orig_scores[docid] + bonus

    final = sorted(((docid, adjusted.get(docid, score)) for docid, score in hits), key=lambda x: -x[1])
    return final


def bm25_fielded_posboost_relative(title_text, headings_text, body_text, span_terms, k=100,
                                    title_boost=1, headings_boost=1, body_boost=1,
                                    span_frac=0.1, span_min=30, span_max=300,
                                    use_relative=True, fixed_span_end=100,
                                    boost_ratio=0.05, ref_topn=10, rerank_n=100):
    """posboostの「文書先頭何語以内か」の判定窓を、文書長dlに応じて相対化できるか
    どうかを比較するための実験用関数（use_relativeで切り替え、他は完全に同一条件）。

    固定span_end=100は文書長を一切考慮しない。ランダムサンプル(n=300)の実測では
    body dlの分布はmin=48, median=610, mean=1072, p90=1993, p99=10102, max=19739と
    裾の長い分布で、dl<100（span_end=100が文書全体をカバーし判定が無意味化する
    退化ケース）は2.3%とまれだが、dl>1000（先頭100語が本文の10%未満しかカバー
    しない）は32%と無視できない割合。長い文書ほど「実質的には早い段落」に書いて
    あっても不当にposboostの恩恵を受けにくい可能性がある、という仮説を検証する。

    use_relative=Trueなら判定窓を clamp(span_frac * dl, span_min, span_max) とし、
    Falseならfixed_span_endを使う。ボーナスの掛け方はbm25_fielded_posboost_normalized
    と同じ（基準スコアに対する比率、triggerしたら定額付与）にして、判定窓の相対化
    "だけ"が効果に与える影響を切り分ける。bm25_fielded_posboost_idfnorm等と同じく
    事後リランク方式（上位rerank_n件のみ並べ替え）のため、recall@1000は原理的に
    不変（top-1000集合自体は動かせない）——ここではnDCG@10等の順位変化のみで
    「相対化に見込みがあるか」を判定する一次スクリーニング。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    res = client.search(index=INDEX, body={
        "size": max(k, ref_topn),
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    all_hits = [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]
    if not all_hits:
        return []
    hits = all_hits[:k]
    if not span_terms:
        return hits

    ref_scores = [score for _, score in all_hits[:ref_topn]]
    reference_score = sum(ref_scores) / len(ref_scores) if ref_scores else 0.0
    if reference_score <= 0:
        return hits

    rerank_ids = [docid for docid, _ in hits[:rerank_n]]
    orig_scores = {docid: score for docid, score in hits}

    tv_docs = client.mtermvectors(index=INDEX, body={
        "ids": rerank_ids,
        "parameters": {
            "fields": ["body"],
            "term_statistics": False,
            "field_statistics": False,
            "positions": True,
            "offsets": False,
            "payloads": False,
        }
    })

    bonus = boost_ratio * reference_score * body_boost
    adjusted = dict(orig_scores)
    for doc in tv_docs.get("docs", []):
        docid = doc.get("_id")
        body_tv = doc.get("term_vectors", {}).get("body")
        if not body_tv or docid not in orig_scores:
            continue
        terms_info = body_tv["terms"]
        dl = sum(t["term_freq"] for t in terms_info.values()) or 1
        span_end = min(span_max, max(span_min, span_frac * dl)) if use_relative else fixed_span_end
        triggered = False
        for t in span_terms:
            info = terms_info.get(t)
            if not info or not info.get("tokens"):
                continue
            first_pos = min(tok["position"] for tok in info["tokens"])
            if first_pos < span_end:
                triggered = True
                break
        if triggered:
            adjusted[docid] = orig_scores[docid] + bonus

    final = sorted(((docid, adjusted.get(docid, score)) for docid, score in hits), key=lambda x: -x[1])
    return final


# qrelsを一切使わずにコーパス全体から測定した「文書内の相対位置ごとの平均IDF」
# （measure_position_idf_density_relative.py、position_idf_density_relative_result.json、
# msmarco-v21-docから798文書・64万トークンをランダムサンプリング）を、
# 「トークン加重平均IDF（5.054994）を1.0とする比率」に正規化したテーブル。
#   - 冒頭ごく一部(1.052)と末尾(1.015)が山、相対位置7〜15%が谷(0.977)というU字カーブ
#   - 谷が1.0を下回るので「平均より情報が薄い位置」を素直に減点できる
#   - 振れ幅は 0.977〜1.052 と実測どおりの控えめな値
# 実測値をそのまま使うと順位がほとんど動かないため、amplifyパラメータで
# 「形はそのまま・振れ幅だけ拡大」できるようにしてある。
# 注意: このグローバルなカーブを使う位置ボーナスは、対照実験
# （evaluate_posdensity_occ.py）でボーナス無しの対照条件を下回り、boost_ratioを上げるほど
# 単調に悪化した＝関連性と逆相関することが確定している。文書ごとに情報の濃い領域の位置が
# 異なるため、コーパス平均を取ると構造が消えてしまうのが原因と考えられる
# （文書ごとに密度を求める代替案は local_density_profile() を参照）。
RELATIVE_POSITION_MEAN_RATIO = [
    (0.0000, 0.0050, 1.051609),
    (0.0050, 0.0100, 1.019759),
    (0.0100, 0.0200, 1.013897),
    (0.0200, 0.0300, 0.995265),
    (0.0300, 0.0500, 0.996784),
    (0.0500, 0.0700, 0.998339),
    (0.0700, 0.1000, 0.977792),
    (0.1000, 0.1500, 0.977406),
    (0.1500, 0.2000, 0.991377),
    (0.2000, 0.3000, 0.990578),
    (0.3000, 0.4000, 0.998635),
    (0.4000, 0.5000, 0.992230),
    (0.5000, 0.6000, 1.003002),
    (0.6000, 0.7000, 1.005396),
    (0.7000, 0.8000, 1.009011),
    (0.8000, 0.9000, 1.003207),
    (0.9000, 1.0001, 1.015416),
]


def relative_position_meanratio(frac, amplify=1.0, table=RELATIVE_POSITION_MEAN_RATIO):
    """相対位置fracにおける重み = 1 + amplify * (実測比率 - 1)。

    amplify=1.0なら実測そのまま（0.977〜1.052）。amplifyを上げると、カーブの形
    （どこが山でどこが谷か。qrels非依存の実測で決まっている）はそのままに、
    振れ幅だけが1.0を中心に拡大する。amplifyは「位置情報を語の存在自体に対して
    どれだけ重視するか」という単一のスカラーで、カーブの形状には一切影響しない。
    負にならないよう下限0.0でクリップする。"""
    r = table[-1][2]
    if frac < table[0][0]:
        r = table[0][2]
    else:
        for lo, hi, rr in table:
            if lo <= frac < hi:
                r = rr
                break
    return max(0.0, 1.0 + amplify * (r - 1.0))


def bm25_equalweight_posmeanratio(title_text, headings_text, body_text, span_terms, k=100,
                               title_boost=1, headings_boost=1, body_boost=1,
                               boost_ratio=0.05, ref_topn=10, rerank_n=200, amplify=1.0):
    """bm25_equalweight_posudensityと同じ「qrels非依存の実測位置カーブで加点する」方式だが、
    重みの正規化をmin-max（0〜1に引き伸ばす）ではなくコーパス平均=1.0の比率
    （RELATIVE_POSITION_MEAN_RATIO）にした別実装。

    bm25_equalweight_posudensityとの違いは重みの作り方だけ:
      - posudensity: weight_base + minmax(avg_idf)。谷が下限（0.0または1.0）に張り付き、
        ピークとの差が実測の7.6%から100%へ大きく増幅される。谷を「減点」にはできない。
      - posmeanratio: 1 + amplify*(avg_idf/コーパス平均 - 1)。谷は1.0未満（0.977）となり
        平均より情報の薄い位置は素直に減点される。amplify=1.0なら増幅なしの実測値そのまま、
        amplifyを上げると形を保ったまま振れ幅だけ拡大する。

    それ以外（候補プール取得→mtermvectorsで位置取得→文書長で正規化→
    triggered_idf/idf_sum比でボーナスを配分）はbm25_equalweight_posudensity・
    bm25_fielded_posboost_idfnormと同一。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    res = client.search(index=INDEX, body={
        "size": max(k, ref_topn),
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    all_hits = [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]
    if not all_hits:
        return []
    hits = all_hits[:k]
    if not span_terms:
        return hits

    ref_scores = [score for _, score in all_hits[:ref_topn]]
    reference_score = sum(ref_scores) / len(ref_scores) if ref_scores else 0.0

    tv_global = client.termvectors(index=INDEX, body={
        "doc": {"body": " ".join(span_terms)},
        "fields": ["body"],
        "field_statistics": True,
        "term_statistics": True,
        "positions": False,
    })
    body_tv_global = tv_global.get("term_vectors", {}).get("body")
    if not body_tv_global:
        return hits
    fs = body_tv_global["field_statistics"]
    N = fs["doc_count"]
    dfs = {t: info.get("doc_freq", 0) for t, info in body_tv_global["terms"].items()}

    def idf(term):
        n = dfs.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    term_idfs = {t: idf(t) for t in span_terms}
    idf_sum = sum(term_idfs.values())
    if idf_sum <= 0 or reference_score <= 0:
        return hits

    rerank_ids = [docid for docid, _ in hits[:rerank_n]]
    orig_scores = {docid: score for docid, score in hits}

    tv_docs = client.mtermvectors(index=INDEX, body={
        "ids": rerank_ids,
        "parameters": {
            "fields": ["body"],
            "term_statistics": False,
            "field_statistics": False,
            "positions": True,
            "offsets": False,
            "payloads": False,
        }
    })

    adjusted = dict(orig_scores)
    for doc in tv_docs.get("docs", []):
        docid = doc.get("_id")
        body_tv = doc.get("term_vectors", {}).get("body")
        if not body_tv or docid not in orig_scores:
            continue
        terms_info = body_tv["terms"]
        dl = sum(t["term_freq"] for t in terms_info.values())
        if dl <= 0:
            continue
        triggered_idf = 0.0
        for t in span_terms:
            info = terms_info.get(t)
            if not info or not info.get("tokens"):
                continue
            first_pos = min(tok["position"] for tok in info["tokens"])
            triggered_idf += term_idfs.get(t, 0.0) * relative_position_meanratio(first_pos / dl,
                                                                                 amplify=amplify)
        if triggered_idf <= 0:
            continue
        bonus = boost_ratio * reference_score * (triggered_idf / idf_sum) * body_boost
        adjusted[docid] = orig_scores[docid] + bonus

    final = sorted(((docid, adjusted.get(docid, score)) for docid, score in hits), key=lambda x: -x[1])
    return final


def bm25_equalweight_posdensity_occ(title_text, headings_text, body_text, span_terms, k=100,
                                 title_boost=2, headings_boost=1, body_boost=1,
                                 boost_ratio=0.05, ref_topn=10, rerank_n=200,
                                 amplify=1.0, occurrences="all"):
    """位置ボーナスの診断用実装。bm25_equalweight_posudensity / bm25_equalweight_posmeanratio が
    抱えていた2つの問題を切り分けられるようにしてある。

    問題1: 測定した対象と使う対象のズレ。
      RELATIVE_POSITION_IMPORTANCE / RELATIVE_POSITION_MEAN_RATIO は「全トークンの出現」に
      ついて相対位置ごとの平均IDFを測ったものだが、既存実装は各span_termの「初出位置」しか
      見ていなかった。実測（診断スクリプトで候補文書11,244サンプル）では初出位置は冒頭に
      強く偏り、88.4%が相対位置0.5未満、末尾の山(0.9〜1.0)に該当するのはわずか1.49%。
      つまり測定したカーブの右半分は初出位置ベースでは事実上発火しない。
      occurrences は3通り:
        "all"/"mean": 全出現位置をカーブで重み付けして平均。測定時の意味（トークン単位の
          平均IDF）とは一致するが、頻出語ほど平均が文書全体の平均(≈1.0)に回帰して判別力が
          消えるという欠点がある。
        "first": 初出位置のみ（従来実装）。
        "max": 全出現のうち最大値＝「一度でも情報密度の高い位置に出現するか」という存在判定。
          championのspan_first（先頭100語以内に出現するか）と同じ構造で、判定に使うカーブ
          だけを実測値に差し替えたことになる。"all"と"first"がどちらも対照条件を下回った
          （evaluate_posdensity_occ.py）のは平均・初出という集約の仕方がchampionと
          逆だったせいである可能性があり、それを切り分けるための条件。

    問題2: ボーナス幅。boost_ratio=0.05では位置ボーナスが順位をほとんど動かせていない
      （zoneアブレーションで前半のみ/後半のみ/全体という正反対の重み付けがnDCG@10で
      ±0.002以内に収まった）。boost_ratio=0.0を渡すとボーナス計算とmtermvectors取得を
      まるごとスキップし、「位置ボーナス無し」の対照条件（ベース検索そのまま）になる。

    重みは RELATIVE_POSITION_MEAN_RATIO（コーパス平均IDF=1.0の比率、谷が0.977で減点になる）
    を relative_position_meanratio(amplify=...) 経由で使う。amplifyはカーブの形を保ったまま
    振れ幅だけを拡大する。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    res = client.search(index=INDEX, body={
        "size": max(k, ref_topn),
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    all_hits = [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]
    if not all_hits:
        return []
    hits = all_hits[:k]
    # boost_ratio=0.0 は「位置ボーナス無し」の対照条件。再ランキングを一切行わない
    # （mtermvectorsも呼ばないので高速）。
    if not span_terms or boost_ratio <= 0:
        return hits

    ref_scores = [score for _, score in all_hits[:ref_topn]]
    reference_score = sum(ref_scores) / len(ref_scores) if ref_scores else 0.0

    tv_global = client.termvectors(index=INDEX, body={
        "doc": {"body": " ".join(span_terms)},
        "fields": ["body"],
        "field_statistics": True,
        "term_statistics": True,
        "positions": False,
    })
    body_tv_global = tv_global.get("term_vectors", {}).get("body")
    if not body_tv_global:
        return hits
    fs = body_tv_global["field_statistics"]
    N = fs["doc_count"]
    dfs = {t: info.get("doc_freq", 0) for t, info in body_tv_global["terms"].items()}

    def idf(term):
        n = dfs.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    term_idfs = {t: idf(t) for t in span_terms}
    idf_sum = sum(term_idfs.values())
    if idf_sum <= 0 or reference_score <= 0:
        return hits

    rerank_ids = [docid for docid, _ in hits[:rerank_n]]
    orig_scores = {docid: score for docid, score in hits}

    tv_docs = client.mtermvectors(index=INDEX, body={
        "ids": rerank_ids,
        "parameters": {
            "fields": ["body"],
            "term_statistics": False,
            "field_statistics": False,
            "positions": True,
            "offsets": False,
            "payloads": False,
        }
    })

    adjusted = dict(orig_scores)
    for doc in tv_docs.get("docs", []):
        docid = doc.get("_id")
        body_tv = doc.get("term_vectors", {}).get("body")
        if not body_tv or docid not in orig_scores:
            continue
        terms_info = body_tv["terms"]
        dl = sum(t["term_freq"] for t in terms_info.values())
        if dl <= 0:
            continue
        triggered_idf = 0.0
        for t in span_terms:
            info = terms_info.get(t)
            if not info or not info.get("tokens"):
                continue
            positions = [tok["position"] for tok in info["tokens"]]
            if occurrences == "first":
                w = relative_position_meanratio(min(positions) / dl, amplify=amplify)
            elif occurrences == "max":
                # 「この語が一度でも情報密度の高い位置に出現するか」という存在判定。
                # championのspan_first（先頭100語以内に出現するか）と同じ構造で、
                # 判定に使うカーブだけを実測値に差し替えたことになる。
                w = max(relative_position_meanratio(p / dl, amplify=amplify) for p in positions)
            else:
                # 全出現をカーブで重み付けして平均する（"all"/"mean"）。
                # 注意: 頻出語ほど平均が文書全体の平均(≈1.0)に回帰して判別力が消える。
                # championが存在判定なのに対しこちらは平均であり、構造が異なる。
                w = sum(relative_position_meanratio(p / dl, amplify=amplify)
                        for p in positions) / len(positions)
            triggered_idf += term_idfs.get(t, 0.0) * w
        if triggered_idf <= 0:
            continue
        bonus = boost_ratio * reference_score * (triggered_idf / idf_sum) * body_boost
        adjusted[docid] = orig_scores[docid] + bonus

    final = sorted(((docid, adjusted.get(docid, score)) for docid, score in hits), key=lambda x: -x[1])
    return final


def local_density_profile(terms_info, N, window=50):
    """1文書のbody term vectors（positions + term_statistics付き）から、
    「その文書自身の情報密度プロファイル」を作って返す。

    戻り値は (density, dl) で、densityは長さdlの配列。density[p]は位置pを中心とする
    幅windowの窓に含まれるトークンの平均IDFを、その文書全体の平均IDFで割った値
    （1.0 = その文書として標準的な情報密度、>1.0 = レア語が密集した領域、
    <1.0 = 定型文・ナビゲーション等の情報の薄い領域）。文書ごとに正規化してあるので、
    語彙の豊かさが違う文書間でも比較できる。

    RELATIVE_POSITION_IMPORTANCE（コーパス全体を平均した位置カーブ）が事実上効かなかった
    のは、文書ごとに情報の濃い領域の場所がバラバラで、800文書ぶん平均すると振れ幅が
    10%程度まで潰れてしまうためだった。こちらは平均を取らず、各文書の中でどこが濃いかを
    その文書自身から求める。"""
    if not terms_info:
        return [], 0
    idf_at, present = {}, []
    max_pos = -1
    for term, info in terms_info.items():
        df = info.get("doc_freq", 0)
        if df <= 0:
            continue
        w = math.log(1 + (N - df + 0.5) / (df + 0.5))
        for tok in info.get("tokens", []):
            p = tok["position"]
            idf_at[p] = w
            if p > max_pos:
                max_pos = p
    if max_pos < 0:
        return [], 0
    dl = max_pos + 1
    # analyzerがstopwordを落とすため位置に欠番がある。欠番は平均から除く。
    present = sorted(idf_at)
    if not present:
        return [], 0
    doc_mean = sum(idf_at.values()) / len(idf_at)
    if doc_mean <= 0:
        return [], dl

    # 累積和で窓平均をO(dl)で計算する（欠番を除いた個数でも割れるようにcountも累積）
    cum_idf = [0.0] * (dl + 1)
    cum_cnt = [0] * (dl + 1)
    for p in range(dl):
        v = idf_at.get(p, 0.0)
        cum_idf[p + 1] = cum_idf[p] + v
        cum_cnt[p + 1] = cum_cnt[p] + (1 if p in idf_at else 0)

    half = max(1, window // 2)
    density = [1.0] * dl
    for p in range(dl):
        lo, hi = max(0, p - half), min(dl, p + half + 1)
        c = cum_cnt[hi] - cum_cnt[lo]
        if c <= 0:
            density[p] = 1.0
        else:
            density[p] = ((cum_idf[hi] - cum_idf[lo]) / c) / doc_mean
    return density, dl


def bm25_equalweight_localdensity(title_text, headings_text, body_text, span_terms, k=100,
                               title_boost=2, headings_boost=1, body_boost=1,
                               boost_ratio=0.5, ref_topn=10, rerank_n=200,
                               window=50, amplify=1.0, aggregate="mean"):
    """クエリ語が「その文書自身の情報が濃い領域」に乗っているほど加点する位置ボーナス。

    これまでのposudensity / posmeanratio / posdensity_occ は、コーパス全体を平均した
    グローバルな位置カーブ（相対位置0.02は一般に重要、等）を使っていたが、対照実験
    （evaluate_posdensity_occ.py）でボーナス無しの対照条件を下回り、しかもboost_ratioを
    上げるほど単調に悪化した＝関連性と逆相関することが確定した。原因は、文書ごとに情報の
    濃い領域の位置がバラバラなため、コーパス平均を取ると構造が消えてしまうこと。

    本関数はグローバルなカーブを一切使わず、local_density_profile()で文書ごとに
    「その文書の中でどこにレア語（＝高IDF語）が密集しているか」を求め、クエリ語がその
    密集領域に出現しているかどうかで加点する。初出位置も使わない（全出現を見る）。

    aggregate="mean" なら各クエリ語の全出現位置での密度の平均、"max" なら最大値を使う
    （"max"は「一箇所でも情報の濃い場所に出ていれば良し」とする解釈）。
    amplifyは密度の振れ幅を1.0中心に拡大する。boost_ratio=0.0で位置ボーナス無しの
    対照条件になる（mtermvectors取得もスキップするので高速）。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_boost}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_boost}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_boost}}})
    res = client.search(index=INDEX, body={
        "size": max(k, ref_topn),
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    all_hits = [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]
    if not all_hits:
        return []
    hits = all_hits[:k]
    if not span_terms or boost_ratio <= 0:
        return hits

    ref_scores = [score for _, score in all_hits[:ref_topn]]
    reference_score = sum(ref_scores) / len(ref_scores) if ref_scores else 0.0
    if reference_score <= 0:
        return hits

    N = _bm25f_total_docs()
    rerank_ids = [docid for docid, _ in hits[:rerank_n]]
    orig_scores = {docid: score for docid, score in hits}

    # term_statistics=Trueにすることで、文書中の全語のdoc_freqが一緒に返る
    # （密度プロファイルを作るには、クエリ語だけでなく全語のIDFが必要）。
    tv_docs = client.mtermvectors(index=INDEX, body={
        "ids": rerank_ids,
        "parameters": {
            "fields": ["body"],
            "term_statistics": True,
            "field_statistics": False,
            "positions": True,
            "offsets": False,
            "payloads": False,
        }
    })

    adjusted = dict(orig_scores)
    for doc in tv_docs.get("docs", []):
        docid = doc.get("_id")
        body_tv = doc.get("term_vectors", {}).get("body")
        if not body_tv or docid not in orig_scores:
            continue
        terms_info = body_tv["terms"]
        density, dl = local_density_profile(terms_info, N, window=window)
        if dl <= 0:
            continue

        term_idfs, weighted = {}, 0.0
        for t in span_terms:
            info = terms_info.get(t)
            if not info or not info.get("tokens"):
                continue
            df = info.get("doc_freq", 0)
            if df <= 0:
                continue
            term_idfs[t] = math.log(1 + (N - df + 0.5) / (df + 0.5))
        idf_sum = sum(term_idfs.values())
        if idf_sum <= 0:
            continue

        for t, t_idf in term_idfs.items():
            vals = [density[tok["position"]] for tok in terms_info[t]["tokens"]
                    if tok["position"] < dl]
            if not vals:
                continue
            d = max(vals) if aggregate == "max" else sum(vals) / len(vals)
            weighted += t_idf * max(0.0, 1.0 + amplify * (d - 1.0))

        bonus = boost_ratio * reference_score * (weighted / idf_sum) * body_boost
        adjusted[docid] = orig_scores[docid] + bonus

    final = sorted(((docid, adjusted.get(docid, score)) for docid, score in hits), key=lambda x: -x[1])
    return final


def bm25_equalweight_posboost_localdensity(title_text, headings_text, body_text, span_terms, k=100,
                                        span_end=100, span_boost=15.0,
                                        boost_ratio=0.5, ref_topn=10, rerank_n=200,
                                        window=50, amplify=-1.0, aggregate="max"):
    """doc版の現状最良設定（webstyle_narrative_equalweight_posboost_only＝
    bm25_equalweight_posboost_discourseboost(markers=[])）の結果に、局所情報密度による
    再ランキングを上乗せする版。

    bm25_equalweight_localdensity()が「championの代わりに」局所密度を使うのに対し、こちらは
    championのネイティブクエリ（title/headings/body均等 + span_first(end=span_end,
    boost=span_boost)）をそのまま一次検索に使い、その上位rerank_n件だけを局所密度で
    並べ替える。位置ボーナス（span_first）と局所密度は別のシグナルなので、両方使えるなら
    上乗せできるはず、という仮説の検証用。

    amplifyのデフォルトが負(-1.0)なのは、事前診断で「クエリ語が異常に密度の高い位置に
    出現する文書ほど正解ではない」（max集約でCohen's d = -0.308、正解1.3099 /
    不正解1.3562）と分かったため。密度の突出した箇所はレア語が不自然に密集した領域
    （キーワードの羅列、タグクラウド、用語集、ナビゲーション等）である可能性が高く、
    そこにクエリ語が埋まっている文書は「そのトピックについて書かれた記事」ではない、
    という解釈。weight = 1 + amplify*(density - 1) なので amplify<0 かつ density>1 で
    weight<1（減点）になる。

    boost_ratio=0.0を渡すと再ランキングを行わず、champion素のままの結果を返す
    （＝対照条件）。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": 1}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": 1}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": 1}}})
    if span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": span_end, "boost": span_boost}})
    res = client.search(index=INDEX, body={
        "size": max(k, ref_topn),
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    all_hits = [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]
    if not all_hits:
        return []
    hits = all_hits[:k]
    if not span_terms or boost_ratio <= 0:
        return hits

    ref_scores = [score for _, score in all_hits[:ref_topn]]
    reference_score = sum(ref_scores) / len(ref_scores) if ref_scores else 0.0
    if reference_score <= 0:
        return hits

    N = _bm25f_total_docs()
    rerank_ids = [docid for docid, _ in hits[:rerank_n]]
    orig_scores = {docid: score for docid, score in hits}

    tv_docs = client.mtermvectors(index=INDEX, body={
        "ids": rerank_ids,
        "parameters": {
            "fields": ["body"],
            "term_statistics": True,
            "field_statistics": False,
            "positions": True,
            "offsets": False,
            "payloads": False,
        }
    })

    adjusted = dict(orig_scores)
    for doc in tv_docs.get("docs", []):
        docid = doc.get("_id")
        body_tv = doc.get("term_vectors", {}).get("body")
        if not body_tv or docid not in orig_scores:
            continue
        terms_info = body_tv["terms"]
        density, dl = local_density_profile(terms_info, N, window=window)
        if dl <= 0:
            continue

        term_idfs = {}
        for t in span_terms:
            info = terms_info.get(t)
            if not info or not info.get("tokens"):
                continue
            df = info.get("doc_freq", 0)
            if df > 0:
                term_idfs[t] = math.log(1 + (N - df + 0.5) / (df + 0.5))
        idf_sum = sum(term_idfs.values())
        if idf_sum <= 0:
            continue

        weighted = 0.0
        for t, t_idf in term_idfs.items():
            vals = [density[tok["position"]] for tok in terms_info[t]["tokens"]
                    if tok["position"] < dl]
            if not vals:
                continue
            d = max(vals) if aggregate == "max" else sum(vals) / len(vals)
            weighted += t_idf * max(0.0, 1.0 + amplify * (d - 1.0))

        bonus = boost_ratio * reference_score * (weighted / idf_sum)
        adjusted[docid] = orig_scores[docid] + bonus

    final = sorted(((docid, adjusted.get(docid, score)) for docid, score in hits), key=lambda x: -x[1])
    return final


_BM25F_N_CACHE = None
_BM25F_AVGDL_CACHE = {}
_BM25F_DF_CACHE = {}


def _bm25f_total_docs(index=None):
    """インデックス全体の文書数N（_countなので正確、シャード集約済み）。1回計算してキャッシュ。"""
    global _BM25F_N_CACHE
    if _BM25F_N_CACHE is None:
        _BM25F_N_CACHE = client.count(index=index or INDEX)["count"]
    return _BM25F_N_CACHE


def _bm25f_avgdl(field, index=None):
    """フィールドの平均長（トークン数）。term_idf()と同じ仮想文書termvectorsの手法で
    取得する（field_statisticsは実際には1シャードだけの集計値だが、デフォルトの
    ハッシュルーティングで文書はシャード間にほぼ均等ランダムに分布するため、平均長の
    推定値としては十分実用的。term_idf()のdocstring参照）。プロセス内で1回だけ計算する。"""
    if field not in _BM25F_AVGDL_CACHE:
        index = index or INDEX
        # プローブ語は必ず analyzer を通過する（＝ストップワードでない）語であること。
        # 2026-09-10 まで "the" を使っていたが english_search はこれを除去するため
        # term_vectors が空で返り、field_statistics が取れず 1.0 にフォールバックしていた。
        # その結果 B = (1-b) + b*(dl/avgdl) の avgdl が 1.0 になり、body の寄与が
        # 約220分の1に潰れていた（正しい avgdl は title 7.2 / headings 50.9 / body 1096.9）。
        # BM25F系の結果は全てこのバグの影響下にあったので、修正後は再測定が必要。
        avgdl = None
        for probe in ("water", "information", "system"):
            res = client.termvectors(index=index, body={
                "doc": {field: probe},
                "fields": [field],
                "field_statistics": True,
                "term_statistics": False,
                "positions": False, "offsets": False, "payloads": False,
            })
            fs = res.get("term_vectors", {}).get(field, {}).get("field_statistics")
            if fs and fs.get("doc_count"):
                avgdl = fs["sum_ttf"] / fs["doc_count"]
                break
        if avgdl is None:
            raise RuntimeError(
                f"_bm25f_avgdl({field!r}): field_statistics を取得できませんでした。"
                "プローブ語が全てストップワード扱いされている可能性があります。")
        _BM25F_AVGDL_CACHE[field] = avgdl
    return _BM25F_AVGDL_CACHE[field]


def _bm25f_combined_df(term, index=None):
    """termがtitle/headings/bodyのいずれかに出現する文書数（フィールド横断のdocument
    frequency）を_countで正確に取得する（3フィールド分のdoc_freqを単純合算すると
    同じ文書が複数フィールドに一致して二重計上されるため、bool/shouldでOR結合したヒット数
    を数える必要がある）。termは既にanalyze_terms()でトークン化済みの語である前提
    （termクエリなので再analyzeされない）。プロセス内でterm単位にキャッシュする。"""
    if term not in _BM25F_DF_CACHE:
        index = index or INDEX
        res = client.count(index=index, body={
            "query": {
                "bool": {
                    "should": [
                        {"term": {"title": term}},
                        {"term": {"headings": term}},
                        {"term": {"body": term}},
                    ]
                }
            }
        })
        _BM25F_DF_CACHE[term] = res.get("count", 0)
    return _BM25F_DF_CACHE[term]


def bm25f_prepare(title_text, headings_text, body_text,
                   title_weight=3.0, headings_weight=2.0, body_weight=1.0,
                   candidate_k=100, rerank_n=None):
    """bm25f()のうちネットワークI/Oが必要な部分（候補プール取得・IDF計算・
    _mtermvectorsでのフィールド別tf取得）だけを行い、後段のスコア計算に必要な生データを
    まとめて返す。b_title/b_headings/b_body/bm25_k1はここでは使わない（純Pythonの
    bm25f_score()側のパラメータ）ので、同じcandidate_k/weightsで複数のb_*組み合わせを
    グリッドサーチする際、候補プール取得・_mtermvectors取得を1回に使い回せる
    （search_bm25f_field_b.py参照）。b_*同様weightsも固定なら候補プール自体は
    変わらないため、weightsを変えたい場合のみ再度prepareし直すこと。"""
    rerank_n = rerank_n or candidate_k

    # 候補プール取得もscoring側と同じフィールド重みでboostする（重みを揃えないと
    # bm25_fielded系との比較で候補プールの母集団自体が変わってしまい、re-rank後の
    # 差がscoring方式の違いなのか候補プールの違いなのか切り分けられなくなるため）。
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": title_weight}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": headings_weight}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": body_weight}}})
    if not should:
        return None
    res = client.search(index=INDEX, body={
        "size": candidate_k,
        "_source": False,
        "query": {"bool": {"should": should}}
    })
    candidates = [h["_id"] for h in res["hits"]["hits"]]
    if not candidates:
        return None
    rerank_ids = candidates[:rerank_n]

    query_terms = []
    seen = set()
    for field, text in (("title", title_text), ("headings", headings_text), ("body", body_text)):
        if not text or not text.strip():
            continue
        for t in analyze_terms(text, field=field):
            if t not in seen:
                seen.add(t)
                query_terms.append(t)

    avgdl = {f: _bm25f_avgdl(f) for f in ("title", "headings", "body")}
    N = _bm25f_total_docs()
    idf = {t: (lambda df: math.log(1 + (N - df + 0.5) / (df + 0.5)) if df > 0 else 0.0)
           (_bm25f_combined_df(t))
           for t in query_terms}

    docs = {}
    if query_terms:
        tv_docs = client.mtermvectors(index=INDEX, body={
            "ids": rerank_ids,
            "parameters": {
                "fields": ["title", "headings", "body"],
                "term_statistics": False,
                "field_statistics": False,
                "positions": False, "offsets": False, "payloads": False,
            }
        })
        for doc in tv_docs.get("docs", []):
            docid = doc.get("_id")
            tvs = doc.get("term_vectors", {})
            field_terms, field_len = {}, {}
            for field in ("title", "headings", "body"):
                terms_info = tvs.get(field, {}).get("terms", {})
                field_terms[field] = terms_info
                field_len[field] = sum(info["term_freq"] for info in terms_info.values())
            docs[docid] = {"field_terms": field_terms, "field_len": field_len}

    return {
        "candidates": candidates,
        "query_terms": query_terms,
        "idf": idf,
        "avgdl": avgdl,
        "docs": docs,
    }


def bm25f_score(raw, k=100, title_weight=3.0, headings_weight=2.0, body_weight=1.0,
                b_title=0.4, b_headings=0.4, b_body=0.4, bm25_k1=0.9):
    """bm25f_prepare()が返した生データからBM25Fスコアを計算してtop-k返す（純Python、
    ネットワークI/Oなし）。b_*・weightを変えたグリッドサーチで使い回す用
    （search_bm25f_field_b.py参照。weightsを変える場合は候補プール自体が
    bm25f_prepare側のweightsと食い違う点に注意——プール取得後の重み変更はスコア計算にしか
    反映されない）。"""
    if raw is None:
        return []
    if not raw["query_terms"]:
        return [(docid, 0.0) for docid in raw["candidates"][:k]]

    idf, avgdl, docs = raw["idf"], raw["avgdl"], raw["docs"]
    weights = {"title": title_weight, "headings": headings_weight, "body": body_weight}
    bs = {"title": b_title, "headings": b_headings, "body": b_body}

    scores = {}
    for docid, doc in docs.items():
        field_terms, field_len = doc["field_terms"], doc["field_len"]
        score = 0.0
        for t in raw["query_terms"]:
            t_idf = idf.get(t, 0.0)
            if t_idf <= 0:
                continue
            pseudo_tf = 0.0
            for field in ("title", "headings", "body"):
                info = field_terms[field].get(t)
                if not info:
                    continue
                tf = info["term_freq"]
                dl = field_len[field]
                B = (1 - bs[field]) + bs[field] * (dl / avgdl[field]) if avgdl[field] else 1.0
                pseudo_tf += weights[field] * tf / B
            if pseudo_tf > 0:
                score += t_idf * pseudo_tf / (bm25_k1 + pseudo_tf)
        scores[docid] = score

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:k]


def bm25f(title_text, headings_text, body_text, k=100,
          title_weight=3.0, headings_weight=2.0, body_weight=1.0,
          b_title=0.4, b_headings=0.4, b_body=0.4, bm25_k1=0.9,
          candidate_k=None, rerank_n=None):
    """Robertson/Zaragoza/Taylor (2004) "Simple BM25 Extension to Multiple Weighted
    Fields" 定義通りの、本来の意味でのBM25Fを実装する。

    bm25_fieldedはtitle/headings/bodyそれぞれ独立に通常のBM25（フィールドごとの
    IDF・文書長正規化・tf飽和を別々に計算）を実行してから3つのスコアを線形和するだけで、
    これは「フィールドをまたいだ真のBM25F」ではない（REPORT.md 4.3節で報告した
    「BM25生スコアはクエリ間でスケールが揃わない」問題は、実はこの独立計算構造――
    フィールドごとに別々のIDF・別々のtf飽和を経由してから単純加算する――に起因する）。
    真のBM25Fは逆に、
      (1) フィールドごとの生tfをフィールド長で正規化してから重み付き合算し
          （擬似tf(t,d) = Σ_field weight_field * tf(t,field,d) / B_field(d)、
           B_field(d) = (1 - b_field) + b_field * dl_field(d) / avgdl_field）、
      (2) 合算後の擬似tfに対して単一のIDF・単一のtf飽和関数を1回だけ適用する
          （score(d) = Σ_t idf(t) * pseudo_tf(t,d) / (k1 + pseudo_tf(t,d))）。
    「titleにもbodyにも出てくる語」を二重計上せず1つの強いシグナルとして正しく評価できる
    点、および全termで単一のIDFスケールに揃う点が、線形和方式(bm25_fielded)や
    OpenSearchのmulti_match type=cross_fields（bm25_crossfields、Lucene独自の近似で
    厳密なBM25Fではない）との構造的な違い。

    OpenSearchはBM25Fをネイティブ実行するクエリ型を持たないため、
    (1) bm25_fieldedと同じbool/should検索でcandidate_k件の候補プールを取得し、
    (2) _mtermvectorsで候補文書のフィールド別tf・フィールド長を取得し、
    (3) Python側でBM25Fの式を厳密計算してスコアを再計算・re-rankする、
    という2段構成にした（bm25_fielded_posboost_consistentと同じ構成パターン）。
    実体はbm25f_prepare()+bm25f_score()の薄いラッパー（1回だけ使うならこちらで十分。
    同じweightsで複数のb_*を試すグリッドサーチではbm25f_prepare()を使い回すこと）。
    候補プールに入らなかった文書を新たに発見することはできない（re-rankのみ）ため、
    recall@1000等はcandidate_k（デフォルトはkと同じ）に事実上律速される点に注意
    （精度指標（nDCG@10・precision@100）の改善を主目的としたrerankとして使うのが
    現実的。全文書に対して厳密なBM25Fを1クエリで計算するにはインデックス自体を
    フィールド横断の実効長情報を持つ形で作り直す必要があり、本関数の対象外）。

    b_*はフィールドごとの文書長正規化強度、bm25_k1は飽和パラメータ（このインデックスの
    実際のsimilarity設定 k1=0.9, b=0.4 をデフォルトに採用。term_idf()・
    bm25_fielded_posboost_consistent()と同じ値）。avgdlは_bm25f_avgdl()
    （term_idf()と同じ仮想文書termvectorsの手法、シャードローカルな概算値）。
    IDFは_bm25f_combined_df()でtitle/headings/body横断の正確なdocument frequency
    （語がいずれかのフィールドに出現する文書数、二重計上なし）を語ごとに取得して計算する
    単一のIDF（フィールドごとに別々のIDFを使う線形和方式との核心的な違い）。"""
    candidate_k = candidate_k or k
    raw = bm25f_prepare(title_text, headings_text, body_text,
                         title_weight=title_weight, headings_weight=headings_weight,
                         body_weight=body_weight, candidate_k=candidate_k, rerank_n=rerank_n)
    if raw is None:
        return []
    return bm25f_score(raw, k=k, title_weight=title_weight, headings_weight=headings_weight,
                        body_weight=body_weight, b_title=b_title, b_headings=b_headings,
                        b_body=b_body, bm25_k1=bm25_k1)


def significant_terms(doc_ids, field="body", size=10):
    """一次検索の上位doc_idsを疑似適合文書集合とし、significant_textアグリゲーションで
    背景（インデックス全体）に対して統計的に有意な語をRM3の拡張語候補として返す。
    シャード分散のdoc frequencyを自前集計するより頑健なので、mtermvectorsではなく
    こちらを使う。返る語はインデックスのanalyzer（english、stopword除去+stemming済み）
    でトークナイズされた語そのもの。"""
    if not doc_ids:
        return []
    res = client.search(index=INDEX, body={
        "size": 0,
        "query": {"ids": {"values": doc_ids}},
        "aggs": {
            "sig_terms": {
                "significant_text": {
                    "field": field,
                    "size": size,
                }
            }
        }
    })
    buckets = res.get("aggregations", {}).get("sig_terms", {}).get("buckets", [])
    return [b["key"] for b in buckets]


def bm25_qrelpool(query_text, k=100):
    """qrelで正解判定されているセグメントだけのインデックス(10,281件)を対象にBM25検索する。"""
    res = client.search(index=QREL_POOL_INDEX, body={
        "size": k,
        "_source": False,
        "query": {
            "match": {
                "body": {
                    "query": query_text
                }
            }
        }
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]

def rrf_fuse(rank_lists, k=60, top_n=100, weights=None):
    """weightsを指定すると重み付きRRF（各リストの1/(k+rank)にweightを掛けて合算）になる。
    未指定（デフォルト）なら全リスト等倍で、従来のRRFと同じ。"""
    if weights is None:
        weights = [1.0] * len(rank_lists)
    fused = defaultdict(float)
    for w, lst in zip(weights, rank_lists):
        for rank, (docid, _) in enumerate(lst, start=1):
            fused[docid] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:top_n]

def show_title_headings(fused, n=10):
    ids = [d for d, _ in fused[:n]]
    res = client.mget(index=INDEX,
                      body={
                          "ids": ids,
                      },
                      params={
                          "_source_includes": "title,headings,url"
                      },
    )
    meta = {d["_id"]: d.get("_source", {}) for d in res["docs"]}
    for docid, score in fused[:n]:
        m = meta.get(docid, {})
        title = m.get("title") or "(no title)"
        hs = [h.strip() for h in (m.get("headings") or "").split("\n") if h.strip()]
        print(f"{score:.5f}  title:{title[:70]}\n")
        print(f"          {' | '.join(hs)[:150]}\n")
        print(f"          {docid}")
        print("-" *60)
        
        


# ============================================================================
# 見出し役割（heading role）ブースト
#
# DISCOURSE_*_MARKERS は body 中の談話標識（"in conclusion" 等）の直後を見る仕組み
# だったが、実測すると要約系マーカーは relevant/nonrelevant の分離にほとんど寄与
# しなかった（measure_heading_role_conditional.py: summaryのlift 1.19〜2.12 に対し
# definition 3.04〜3.34、cause 2.49〜3.15）。
#
# こちらは着目点を body から headings フィールドへ移し、「見出しは役割を持つ」
# （意味→定義、語源→起源、まとめ→要約、注意点→重要 …）という仮説を検索式にする。
# 1本の見出しの中で「役割語」と「クエリ語」が近接している文書を加点する。
# headings フィールドは見出しを改行で連結した1本のテキストなので、slop を小さく
# 取ることで「同一見出し内」を近似する（改行に position gap は入らないため、
# slop を大きくすると隣の見出しへ漏れる点に注意）。
# ============================================================================

HEADING_ROLE_PATTERNS = {
    "definition": r"\bwhat (is|are|was|were)\b|\bdefinition\b|\bdefined\b|\bmeaning\b|\bwhat does .* mean\b",
    "detail":     r"\bhow (it|they|does|do) work|\bexplained\b|\bin detail\b|\bdetails\b|\boverview\b|\babout\b",
    "origin":     r"\bhistory\b|\borigin(s)?\b|\betymolog|\bwhere .* (come|comes) from\b|\bbackground\b",
    "qa":         r"\bfaq(s)?\b|frequently asked|\bq ?& ?a\b|\bquestions\b|\bpeople also ask\b",
    "summary":    r"\bsummary\b|\bconclusion(s)?\b|key takeaway|\bin short\b|bottom line|\btl;?dr\b|\brecap\b",
    "caution":    r"\bwarning(s)?\b|\bcaution\b|\brisk(s)?\b|side effect|\bprecaution|\bimportant\b|\bnote\b|\bbeware\b|\bdanger",
    "comparison": r"\bvs\.?\b|\bversus\b|difference(s)? between|\bcompared? (to|with)\b|\bcomparison\b",
    "procedure":  r"\bhow to\b|\bsteps?\b|\bguide\b|\binstructions\b|\btutorial\b",
    "cause":      r"\bwhy\b|\bcause(s|d)?\b|\breason(s)?\b|\bdue to\b",
}
"""見出しテキスト／Subqueryテキストの両方に当てて役割ラベル・意図ラベルを付ける正規表現。
文書側（見出し）とクエリ側（Subquery）で同じ語彙を使うのが要点で、両者が一致した
（aligned）ときに relevant との相関が最も強くなる
（measure_heading_role_alignment.py: plain 1.84 < crossed 2.20 < aligned 3.06）。"""

HEADING_ROLE_MARKERS = {
    "definition": ["definit", "defin", "mean"],
    "detail":     ["detail", "overview", "explain"],
    "origin":     ["histori", "origin", "background", "etymolog"],
    "qa":         ["faq", "frequent", "question", "ask"],
    "summary":    ["conclus", "summari", "summar", "conclud", "overal", "takeawai", "recap"],
    "caution":    ["warn", "risk", "precaut", "caution", "danger", "side"],
    "comparison": ["vs", "versu", "differ", "compar", "comparison"],
    "procedure":  ["step", "guid", "instruct", "tutori"],
    "cause":      ["why", "caus", "reason"],
}
"""HEADING_ROLE_PATTERNSに対応する、english analyzerでのステム形マーカー。
既存のDISCOURSE_*_MARKERSと同じ基準で、headingsフィールドの文書頻度が高すぎて
識別力を持たない語は落としてある（heading_role_lexicon_df.json 参照）:
what=19.9%, how=22.0%, about=12.0%, work=5.1%, between=2.2% を除外。
採用した語はいずれも headings 文書頻度 6.4% 以下。"""

_heading_role_re = None


def heading_roles(text):
    """テキスト（見出し1本、またはSubquery）に含まれる役割ラベルの集合を返す。
    文書側に使えば「その見出しの役割」、クエリ側に使えば「そのSubqueryの意図」になる。"""
    global _heading_role_re
    if _heading_role_re is None:
        import re
        _heading_role_re = {r: re.compile(p, re.I) for r, p in HEADING_ROLE_PATTERNS.items()}
    return {r for r, rx in _heading_role_re.items() if rx.search(text)}


def role_markers_for(roles, max_markers=None):
    """役割ラベル集合 -> マーカーstemのリスト（重複除去、元の順序を保つ）。"""
    out = []
    for r in roles:
        for mk in HEADING_ROLE_MARKERS.get(r, []):
            if mk not in out:
                out.append(mk)
    return out[:max_markers] if max_markers else out


def bm25_equalweight_posboost_headingroleboost(title_text, headings_text, body_text, span_terms,
                                               role_markers, k=100,
                                               span_end=100, span_boost=15.0,
                                               heading_slop=6, heading_role_boost=5.0):
    """現行チャンピオン（bm25_equalweight_posboost_discourseboost の markers=[] 相当、
    title/headings/body均等 + body先頭span_end語へのposboost）に、headingsフィールドでの
    「役割語 × クエリ語」近接ブーストを足した版。

    role_markers は呼び出し側で決める役割語stemのリスト。Subqueryの意図に一致する役割
    （heading_roles(subquery) -> role_markers_for(...)）を渡すのが提案手法（aligned）、
    全役割を渡すのが役割非依存の対照条件（all）にあたる。空リストならチャンピオンと同一。
    span_terms は analyze_terms() で得たクエリ語stem（bodyもheadingsも同じ
    english analyzer なので使い回せる）。"""
    should = []
    if title_text and title_text.strip():
        should.append({"match": {"title": {"query": title_text, "boost": 1}}})
    if headings_text and headings_text.strip():
        should.append({"match": {"headings": {"query": headings_text, "boost": 1}}})
    if body_text and body_text.strip():
        should.append({"match": {"body": {"query": body_text, "boost": 1}}})
    if span_terms:
        span_or = {"span_or": {"clauses": [{"span_term": {"body": t}} for t in span_terms]}}
        should.append({"span_first": {"match": span_or, "end": span_end, "boost": span_boost}})
        if role_markers:
            # in_order=False: "What is X" / "X definition" のどちらの語順も拾う
            role_near = [
                {"span_near": {"clauses": [{"span_term": {"headings": mk}},
                                           {"span_term": {"headings": t}}],
                               "slop": heading_slop, "in_order": False}}
                for mk in role_markers for t in span_terms
            ]
            should.append({"bool": {"should": role_near, "boost": heading_role_boost}})
    res = client.search(index=INDEX, body={
        "size": k,
        "_source": False,
        "query": {"bool": {"should": should}},
    })
    return [(h["_id"], h["_score"]) for h in res["hits"]["hits"]]
