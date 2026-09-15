"""
decomposed_query2doc_webstyle_expansion.py が保存した query2doc_token_logprobs
（生成JSON文字列に対する、BPEトークンごとの{"token","logprob","bytes"}）から、
title/headings/body それぞれについて「単語(term)ごとの重要度重み」を作る。

手順:
1. 生JSON文字列(raw)中で、各フィールドの値がどのバイト範囲にあるかを
   json.dumps(値, ensure_ascii=False) の文字列を探すことで特定する。
2. トークンのbytes長を先頭から累積し、各トークンのバイト範囲を求める。
3. フィールドの値(プレーンテキスト)を \\w+ で単語分割し、各単語のバイト範囲に
   重なるトークンのlogprobを合算する（対数確率の連鎖律: 単語の同時生成確率の対数
   = 構成トークンのlogprobの和）。
4. フィールド内で softmax(logit_t / temperature) して、単語ごとの重みにする。

単語の大文字小文字・語形はそのまま返す（ステミング等はOpenSearch側のanalyzerに任せる
設計なので、ここでは表層形のまま扱う）。
"""

import json
import math
import re

from retriever import term_idf

_WORD_RE = re.compile(r"\w+")
_idf_cache = {}


def _cached_idf(word, field):
    key = (field, word.lower())
    if key not in _idf_cache:
        _idf_cache[key] = term_idf(word, field=field)
    return _idf_cache[key]


def combine_with_idf(words_weighted, field):
    """words_weighted: [(word, weight)]（合計が単語数=平均1になるよう正規化済み、
    例えばextract_term_weights(sign=-1)のsurprisal出力）。実コーパスのIDF
    （ストップワード等は0）で追加に重み付けし、combined = weight * (idf / 平均idf)
    を再度平均1になるよう正規化して返す。"surprisalは高いが実は corpus 全体では
    ありふれた語"（例:"Water"）を、IDFの低さで割り引く狙い。
    全語のIDFが0（=全てストップワード等）の場合は元の重みをそのまま返す。"""
    if not words_weighted:
        return []
    idfs = [_cached_idf(w, field) for w, _ in words_weighted]
    mean_idf = sum(idfs) / len(idfs)
    if mean_idf <= 0:
        return words_weighted
    combined = [weight * (idf / mean_idf) for (_, weight), idf in zip(words_weighted, idfs)]
    mean_combined = sum(combined) / len(combined)
    if mean_combined <= 0:
        return words_weighted
    return [(word, c / mean_combined) for (word, _), c in zip(words_weighted, combined)]


def _token_byte_spans(token_logprobs):
    """[(start_byte, end_byte, logprob), ...] をトークン出現順に返す。"""
    spans = []
    pos = 0
    for t in token_logprobs:
        n = len(t["bytes"]) if t.get("bytes") is not None else len(t["token"].encode("utf-8"))
        spans.append((pos, pos + n, t["logprob"]))
        pos += n
    return spans


def _find_field_byte_span(raw, value, search_from_char=0):
    """rawの中で、valueがJSON文字列としてエスケープされた形の位置を探し、
    (start_byte, end_byte, 次の検索開始位置=文字index) を返す。見つからなければNone。"""
    escaped = json.dumps(value, ensure_ascii=False)[1:-1]
    if not escaped:
        return None
    char_pos = raw.find(escaped, search_from_char)
    if char_pos == -1:
        return None
    start_byte = len(raw[:char_pos].encode("utf-8"))
    end_byte = start_byte + len(escaped.encode("utf-8"))
    return start_byte, end_byte, char_pos + len(escaped)


def _words_with_raw_logits(field_start_byte, field_text, token_spans):
    """1つのテキスト片の中の単語ごとの生logit [(word, logit), ...] を返す（softmax前）。"""
    words = []
    for m in _WORD_RE.finditer(field_text):
        word = m.group()
        w_start = field_start_byte + len(field_text[:m.start()].encode("utf-8"))
        w_end = field_start_byte + len(field_text[:m.end()].encode("utf-8"))
        logit = sum(lp for (ts, te, lp) in token_spans if ts < w_end and te > w_start)
        words.append((word, logit))
    return words


def _softmax_words(words, temperature, sign=1):
    """[(word, logit), ...] -> [(word, weight), ...]（フィールド全体でsoftmax、平均1になるよう
    N倍して返す）。平均1にするのは、重みなし版(bm25_fielded)が全単語に暗黙的に重み1を
    均等にかけて合算しているのと同じ「総量」を保ったまま、単語間の相対的な強弱だけを
    反映させるため（合計1のままだと単語数で割った分だけ全体が薄まり、重み付けの符号に
    関わらず性能が下がるだけになってしまう）。
    sign=1: logitが高い(モデルが確信していた予測しやすい語)ほど重みが高い。
    sign=-1: surprisal(-logit)が高い(予測しにくい=情報量が多い語)ほど重みが高い。"""
    if not words:
        return []
    n = len(words)
    logits = [sign * l for _, l in words]
    m = max(logits)
    exps = [math.exp((l - m) / temperature) for l in logits]
    z = sum(exps)
    return [(w, n * e / z) for (w, _), e in zip(words, exps)]


def extract_term_weights(raw, structured_doc, token_logprobs, temperature=1.0, sign=1):
    """raw: LLMが返した生JSON文字列。structured_doc: parse_webdoc()の出力
    ({"title","headings","body"})。token_logprobs: query2doc_token_logprobs[j]。
    sign=1: confidence方式（logitが高い語ほど重い）。sign=-1: surprisal方式（logitが低い
    ＝予測しにくい語ほど重い）。

    戻り値: {"title": [(word, weight), ...], "headings": [...], "body": [...]}
    見つからなかった/logprobが無いフィールドは空リスト。
    """
    token_spans = _token_byte_spans(token_logprobs)
    out = {"title": [], "headings": [], "body": []}
    if not token_spans:
        return out

    search_from = 0

    title = structured_doc.get("title") or ""
    if title:
        found = _find_field_byte_span(raw, title, search_from)
        if found:
            start_byte, end_byte, search_from = found
            out["title"] = _softmax_words(
                _words_with_raw_logits(start_byte, title, token_spans), temperature, sign)

    heading_words = []
    for h in structured_doc.get("headings") or []:
        found = _find_field_byte_span(raw, h, search_from)
        if not found:
            continue
        start_byte, end_byte, search_from = found
        heading_words.extend(_words_with_raw_logits(start_byte, h, token_spans))
    if heading_words:
        out["headings"] = _softmax_words(heading_words, temperature, sign)

    body = structured_doc.get("body") or ""
    if body:
        found = _find_field_byte_span(raw, body, search_from)
        if found:
            start_byte, end_byte, search_from = found
            out["body"] = _softmax_words(
                _words_with_raw_logits(start_byte, body, token_spans), temperature, sign)

    return out


def uniform_term_weights(structured_doc):
    """全単語の重みを1.0にした [(word, 1.0), ...] を title/headings/body ごとに返す。
    bm25_fielded_weighted_posboost に渡すと「term単位に分解はするが重み付けはしない」
    対照条件になる（重み付けの効果だけを切り出すための対照）。"""
    out = {}
    for field in ("title", "headings", "body"):
        value = structured_doc.get(field) or ("" if field != "headings" else [])
        text = " ".join(value) if isinstance(value, list) else value
        out[field] = [(m.group(), 1.0) for m in _WORD_RE.finditer(text)]
    return out


def idf_term_weights(structured_doc, alpha=1.0):
    r"""生成時のtoken logprobを使わず、実コーパスのIDFだけから単語重みを作る。

    alpha: IDF強調の強さを制御する指数。weight ∝ (idf/平均idf)**alpha を平均1に再正規化する。
      BM25はスコア計算時に既にidfを掛けているため、この重みをboostに使うと実効的な重みは
      idf**(1+alpha) になる。したがって
        alpha=0   → idf**1（素のBM25と同じ。全語が重み1の均等）
        alpha=1   → idf**2（IDFを二重に掛ける。2026-09-09以前の既定）
      alpha を1に固定する理論的根拠は無い（BM25のidfはRobertson-Spärck Jones重みの導出を
      持つが、それを2乗する導出は存在しない）。クエリ側の重みスロット
      （通常はクエリ内項頻度の飽和項が入る場所）にIDFを入れる操作なので、
      強さは経験的に探索すべき量である。alpha ごとの実測は
      idf_alpha_sweep_result.json を参照。

    gemini-3.7-flash のようにlogprobが取得できないモデルの疑似文書でも使える項重み付け。
    フィールドごとに \w+ で単語分割し、weight = idf / mean(idf) として平均1に正規化する
    （extract_term_weights のsoftmax出力と同じく「総量は変えず相対的な強弱だけを与える」
    正規化。bm25_fielded_weighted系はweightをboostの倍率に使うため、平均1でないと
    重み付けの符号と無関係に全体のスコアスケールが動いてしまう）。

    IDFが全語0（全てストップワード等）のフィールドは、全語1.0の均等重みにフォールバックする。
    """
    out = {}
    for field in ("title", "headings", "body"):
        value = structured_doc.get(field) or ("" if field != "headings" else [])
        text = " ".join(value) if isinstance(value, list) else value
        words = [m.group() for m in _WORD_RE.finditer(text)]
        if not words:
            out[field] = []
            continue
        idfs = [_cached_idf(w, field) for w in words]
        mean_idf = sum(idfs) / len(idfs)
        if mean_idf <= 0:
            out[field] = [(w, 1.0) for w in words]
        else:
            # idf=0（english analyzerのストップワード等）の語は boost=0 になり検索に
            # 寄与しないので、節数を減らすために落とす（結果は同じで速いだけ）。
            kept = [(w, i) for w, i in zip(words, idfs) if i > 0]
            if not kept:
                out[field] = []
                continue
            raw = [(i / mean_idf) ** alpha for _, i in kept]
            # 指数をかけると平均が1からずれるので再正規化する。これにより alpha は
            # 「重みの総量」ではなく「語間の強弱の付け方」だけを変える。
            mean_raw = sum(raw) / len(raw)
            if mean_raw <= 0:
                out[field] = [(w, 1.0) for w, _ in kept]
            else:
                out[field] = [(w, r / mean_raw) for (w, _), r in zip(kept, raw)]
    return out
