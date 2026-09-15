"""
evaluate_decomposed_webstyle_scorerrf.py（検索結果スコアベースの重み付きRRF、効果はほぼ
誤差レベルだった）の発展版。重み付きRRFは各リストのスコアを定数倍するだけなので、
候補プール（どの文書がそもそも最終順位の土俵に上がるか）自体は変わらず、僅かな順位の
入れ替えにしかならない。これに対して本スクリプトの webstyle_narrative_fielded_recallopt_
quotaunion は「重要なSubqueryほど最終1000件の中に多くの枠（クォータ）を確保する」という
**クォータ制の和集合**にする。

具体的には、Subqueryごとに一次検索結果上位SCORE_WEIGHT_TOPN件のBM25スコア合計を
richnessとし、richnessの比率でTOPK=1000件を按分してクォータ（枠数）を決める。richnessの
大きいSubqueryから順に、そのSubquery自身のランキングの上位から**そのSubquery独自に**
クォータ分だけ文書を確定的に採用する（他のSubqueryでのスコアの高低に関係なく採用される
ため、スコア比較では埋もれてしまうが「そのSubqueryの中では上位」という文書を拾える）。
重複はスキップして次のSubqueryへ進み、全Subquery分のクォータを使い切っても1000件に
満たない場合は、通常のRRF融合スコアが高い順に残りを埋める。順位付け（nDCG@10等に影響）は
採用集合に対して通常のRRF融合スコアを使う。

比較対象は baseline / webstyle_narrative_fielded_recallopt（均等重みRRF、既存の最良recall
条件） / webstyle_narrative_fielded_recallopt_quotaunion（クォータ制和集合）の3条件。
RETRIEVE_K=1000。

以下は元スクリプト（evaluate_decomposed_webstyle.py）のdocstringからの引用（条件の定義）。

条件:
- baseline: narrativeそのままで bm25_body（参考値）
- webstyle: Subquery + (title+headings+bodyを連結した)疑似文書 で bm25_body
- webstyle_keyterms: webstyleと同じ疑似文書だが、bm25_keyterms
  （title^3, headings^2, body^1のmulti_match。best_fields=フィールドごとのスコアの
  「最大値」を採用、線形和ではない）で検索
- webstyle_narrative_fielded: narrative（QUERY_REPEAT回繰り返し）+ Subquery を
  疑似文書のtitle/headings/bodyそれぞれに付けた上で、生成されたtitleはtitleフィールド、
  headingsはheadingsフィールド、bodyはbodyフィールドへ別々にmatchし、bm25_fielded
  （title^3 + headings^2 + body^1の線形和。bool/shouldでスコアを足し合わせる）で検索
- fielded_pool_rerank: RRFをやめて「候補プール（和集合）+ スコア引き継ぎ」方式にしたもの。
  Subquery毎にwebstyle_narrative_fieldedと同じ検索(title/headings/bodyの
  個別match、重み3,2,1、bm25_fielded)を行い、それぞれ上位POOL_K件のdocidとBM25スコアを
  取得する。この時点で得られたスコアをそのまま使い、同じdocidが複数のSubquery
  の上位POOL_K件に入っていればスコアを合算（線形和）し、Subqueryを横断した
  1本のランキングにする（再クエリはしない）。RRFのような順位ベースの融合ではなく、
  スコアそのものを引き継いで足し合わせる点が webstyle_narrative_fielded との違い。
- fielded_title_only / fielded_headings_only / fielded_body_only: webstyle_narrative_fielded
  で使っているtitle_q/headings_q/body_q（narrative×N + Subquery + 生成した
  title/headings/bodyそれぞれ）は同じだが、対応するフィールド1つだけをmatchする
  （他の2フィールドは無視）。title/headings/bodyのどれが効いているかを切り分けるための
  アブレーション。単体性能はheadings > title > bodyの順だった（別途実施済み）。
- webstyle_narrative_fielded_hboost: webstyle_narrative_fieldedと全く同じ組み立てだが、
  フィールドブーストをtitle^3+headings^2+body^1からtitle^1+headings^3+body^1に変更した
  もの。単体性能で最も強かったheadingsの重みを上げるとどうなるかを見る（結果: 全指標で
  元の3,2,1より悪化した）。
- webstyle_narrative_fielded_regboost: フィールドブーストをlearn_field_boosts.pyの線形回帰
  で推定した比率（title=3.4379, headings=2.4865, body=1.0、元の3,2,1に近い値）に変更した
  もの（結果: 元の3,2,1よりわずかに悪化）。
- webstyle_narrative_fielded_recallopt / webstyle_narrative_fielded_ndcgopt:
  search_field_boosts.py（実際のパイプラインでrecall@1000/nDCG@10を直接最大化する座標降下法
  探索、35トピックのサブセットで実施）で見つかった比率をそれぞれ使う版。
  recallopt: title=2, headings=1, body=1。ndcgopt: title=3, headings=3, body=1。

baseline/webstyle/webstyle_keyterms/webstyle_narrative_fielded/webstyle_narrative_fielded_hboost/
fielded_title_only/fielded_headings_only/fielded_body_onlyは Subquery
（narrativeは使わない）+ 疑似文書 を連結して検索し、トピック内の全Subquery分を
RRF融合する。fielded_pool_rerankだけRRFを使わず上記のプール+リランキング方式になる。

新規のLLM生成は不要。multi_query2doc_decomposed_webstyle_L200.json が
既に生成済みであることが前提。

使い方:
    python evaluate_decomposed_webstyle.py [QUERY_REPEAT]   # 省略時は5
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import bm25_body, bm25_keyterms, bm25_fielded, bm25_title, bm25_headings, rrf_fuse


def bm25_fielded_hboost(title_text, headings_text, body_text, k=100):
    """bm25_fieldedのブースト比をtitle^1+headings^3+body^1に変えた版。"""
    return bm25_fielded(title_text, headings_text, body_text, k=k,
                         title_boost=1, headings_boost=3, body_boost=1)


def bm25_fielded_regboost(title_text, headings_text, body_text, k=100):
    """learn_field_boosts.pyの線形回帰で推定したブースト比
    （title=3.4379, headings=2.4865, body=1.0、field_boost_regression.json参照）を使う版。"""
    return bm25_fielded(title_text, headings_text, body_text, k=k,
                         title_boost=3.4379, headings_boost=2.4865, body_boost=1.0)


def bm25_fielded_recallopt(title_text, headings_text, body_text, k=100):
    """search_field_boosts.pyの直接探索でrecall@1000を最大化したブースト比
    （title=2, headings=1, body=1、field_boost_search_result.json参照）を使う版。"""
    return bm25_fielded(title_text, headings_text, body_text, k=k,
                         title_boost=2, headings_boost=1, body_boost=1)


def bm25_fielded_ndcgopt(title_text, headings_text, body_text, k=100):
    """search_field_boosts.pyの直接探索でnDCG@10を最大化したブースト比
    （title=3, headings=3, body=1、field_boost_search_result.json参照）を使う版。"""
    return bm25_fielded(title_text, headings_text, body_text, k=k,
                         title_boost=3, headings_boost=3, body_boost=1)

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 1000  # 検索深度を3000→1000に縮小（TOPKと同じ）した実験版
POOL_K = 1000  # fielded_pool_rerank: Subquery毎に候補プールへ入れる件数
SCORE_WEIGHT_TOPN = 100  # scorerrf: Subqueryの重み計算に使う上位件数
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5  # narrativeをN回繰り返してから連結する

METHODS = ["baseline", "webstyle_narrative_fielded_recallopt",
           "webstyle_narrative_fielded_recallopt_quotaunion"]

METRIC_SPECS = ["recall.100", "recall.1000", "ndcg_cut.10", "P.100"]
METRICS = set(METRIC_SPECS)
METRIC_KEYS = [s.replace(".", "_") for s in METRIC_SPECS]


def load_qrels(path):
    qrels, skipped = {}, 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                skipped += 1
                if skipped <= 3:
                    print(f"  [warn] {os.path.basename(path)}:{lineno} "
                          f"列数 {len(parts)} をスキップ", file=sys.stderr)
                continue
            qid, _, docid, rel = parts
            qrels.setdefault(qid, {})[docid] = int(rel)
    if skipped:
        print(f"  [warn] {os.path.basename(path)}: 計 {skipped} 行スキップ",
              file=sys.stderr)
    return qrels


def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                queries[obj["id"]] = obj["title"]
    return queries


class CachedSearch:

    def __init__(self, fn, topk):
        self.fn = fn
        self.topk = topk
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, text):
        key = text.strip()
        if not key:
            return []
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self._cache[key] = self.fn(key, k=self.topk)
        return self._cache[key]

    def stats(self, label):
        return f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回"


class CachedFieldedSearch:
    """bm25_fielded(title, headings, body, k) 用。3テキストのタプルでキャッシュする。"""

    def __init__(self, fn, topk):
        self.fn = fn
        self.topk = topk
        self._cache = {}
        self.calls = self.hits = 0

    def __call__(self, title_text, headings_text, body_text):
        key = (title_text.strip(), headings_text.strip(), body_text.strip())
        if not any(key):
            return []
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self._cache[key] = self.fn(*key, k=self.topk)
        return self._cache[key]

    def stats(self, label):
        return f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回"


def fuse(lists, topk):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


def quota_fuse(lists, weights, topk):
    """重要度（weights）の比率でtopk件を按分し、richness降順にSubquery自身のランキング
    上位からクォータ分を確定採用する（他リストのスコアと無関係に採用されるので、
    スコア比較では埋もれる文書も拾える）。クォータ消化後にtopk未満なら通常のRRF融合
    スコア上位で埋める。順位付け（run[docid]のスコア）はその通常RRF融合スコアを使う。"""
    pairs = [(lst, w) for lst, w in zip(lists, weights) if lst]
    if not pairs:
        return {}
    lists2 = [p[0] for p in pairs]
    weights2 = [p[1] for p in pairs]
    total_w = sum(weights2) or 1.0
    quotas = [max(1, round(topk * w / total_w)) for w in weights2]

    order = sorted(range(len(lists2)), key=lambda i: -weights2[i])
    included = {}
    for i in order:
        count = 0
        for docid, _ in lists2[i]:
            if docid in included:
                continue
            included[docid] = True
            count += 1
            if count >= quotas[i]:
                break

    fused_all = dict(rrf_fuse(lists2, top_n=100000))
    if len(included) < topk:
        for docid, _ in sorted(fused_all.items(), key=lambda x: -x[1]):
            if docid in included:
                continue
            included[docid] = True
            if len(included) >= topk:
                break

    run = {docid: float(fused_all.get(docid, 0.0)) for docid in included}
    if len(run) > topk:
        run = dict(sorted(run.items(), key=lambda x: -x[1])[:topk])
    return run


def weighted_fuse(lists, weights, topk):
    """listsとweightsは同じ長さ（空リストは重みごと除外）。重み付きRRFで統合する。"""
    pairs = [(lst, w) for lst, w in zip(lists, weights) if lst]
    if not pairs:
        return {}
    lists2 = [p[0] for p in pairs]
    weights2 = [p[1] for p in pairs]
    return {docid: float(score)
            for docid, score in rrf_fuse(lists2, top_n=topk, weights=weights2)}


def dq_pairs(entry):
    """(Subquery, 疑似文書フラットテキスト) のペアを、欠落文書を除いて返す。"""
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"], entry["query2doc_docs"])
            if doc and doc.strip()]


def dq_pairs_structured(entry):
    """(Subquery, {"title","headings","body"}) のペアを、欠落文書を除いて返す。"""
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"],
                                          entry["query2doc_docs_structured"])
            if doc and doc.get("body")]


def build_run(method, qids, queries, webstyle,
               search_body, search_keyterms, search_title, search_headings,
               fielded_searchers):
    """fielded_searchers: {method名: search_fielded系キャッシュ} のdict。
    webstyle_narrative_fielded* という名前の全バリアント（ブースト比違い）はここに
    まとめてあり、下の分岐は1つで済ませている。"""
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]

        if method == "baseline":
            run[qid] = fuse([search_body(q)], TOPK)
        elif method == "webstyle":
            pairs = dq_pairs(webstyle[qid])
            lists = [search_body(f"{dq} {doc}") for dq, doc in pairs]
            run[qid] = fuse(lists, TOPK)
        elif method == "webstyle_keyterms":
            pairs = dq_pairs(webstyle[qid])
            lists = [search_keyterms(f"{dq} {doc}") for dq, doc in pairs]
            run[qid] = fuse(lists, TOPK)
        elif method in fielded_searchers:
            search_fn = fielded_searchers[method]
            pairs = dq_pairs_structured(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = []
            for dq, doc in pairs:
                title_q = f"{repeated_q} {dq} {doc['title']}"
                headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
                body_q = f"{repeated_q} {dq} {doc['body']}"
                lists.append(search_fn(title_q, headings_q, body_q))
            run[qid] = fuse(lists, TOPK)
        elif method == "fielded_pool_rerank":
            pairs = dq_pairs_structured(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            total_score = {}
            for dq, doc in pairs:
                title_q = f"{repeated_q} {dq} {doc['title']}"
                headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
                body_q = f"{repeated_q} {dq} {doc['body']}"
                # webstyle_narrative_fieldedと同じ検索・同じキャッシュを使い回し、
                # 上位POOL_K件のスコアをそのまま足し合わせる（再クエリしない）
                for docid, score in fielded_searchers["webstyle_narrative_fielded"](title_q, headings_q, body_q)[:POOL_K]:
                    total_score[docid] = total_score.get(docid, 0.0) + float(score)
            run[qid] = dict(sorted(total_score.items(), key=lambda x: -x[1])[:TOPK])
        elif method == "fielded_title_only":
            pairs = dq_pairs_structured(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = [search_title(f"{repeated_q} {dq} {doc['title']}") for dq, doc in pairs]
            run[qid] = fuse(lists, TOPK)
        elif method == "fielded_headings_only":
            pairs = dq_pairs_structured(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = [search_headings(f"{repeated_q} {dq} {' '.join(doc['headings'])}") for dq, doc in pairs]
            run[qid] = fuse(lists, TOPK)
        elif method == "fielded_body_only":
            pairs = dq_pairs_structured(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists = [search_body(f"{repeated_q} {dq} {doc['body']}") for dq, doc in pairs]
            run[qid] = fuse(lists, TOPK)
        elif method == "webstyle_narrative_fielded_recallopt_quotaunion":
            # Subqueryごとの一次検索スコア合計（上位SCORE_WEIGHT_TOPN件）をrichnessとし、
            # richness比率でTOPK件を按分。richnessの大きいSubqueryから順に、そのSubquery
            # 自身のランキング上位からクォータ分を確定採用する（クォータ制和集合）。
            search_fn = fielded_searchers["webstyle_narrative_fielded_recallopt"]
            pairs = dq_pairs_structured(webstyle[qid])
            repeated_q = " ".join([q] * QUERY_REPEAT)
            lists, weights = [], []
            for dq, doc in pairs:
                title_q = f"{repeated_q} {dq} {doc['title']}"
                headings_q = f"{repeated_q} {dq} {' '.join(doc['headings'])}"
                body_q = f"{repeated_q} {dq} {doc['body']}"
                result = search_fn(title_q, headings_q, body_q)
                lists.append(result)
                weights.append(sum(score for _, score in result[:SCORE_WEIGHT_TOPN]))
            run[qid] = quota_fuse(lists, weights, TOPK)
        else:
            raise ValueError(f"未知の method: {method}")

        if i % 20 == 0 or i == len(qids):
            print(f"\r  {method}: {i}/{len(qids)} ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    print()
    return run


def evaluate(run, qrels, eval_qids, label):
    target = [q for q in eval_qids if q in qrels]
    if not target:
        print(f"[{label}] 採点対象なし")
        return None, 0

    evaluator = pytrec_eval.RelevanceEvaluator({q: qrels[q] for q in target}, METRICS)
    results = evaluator.evaluate({q: run[q] for q in target})
    n = len(results)
    agg = {m: sum(r[m] for r in results.values()) / n for m in METRIC_KEYS}

    print(f"[{label}] n={n}  " + "  ".join(f"{m}={agg[m]:.4f}" for m in METRIC_KEYS))
    return agg, n


def check_docid_overlap(run, qrels, eval_qids, label):
    run_ids, qrel_ids = set(), set()
    for qid in eval_qids:
        if qid in qrels:
            run_ids |= set(run.get(qid, {}))
            qrel_ids |= set(qrels[qid])
    if not qrel_ids:
        return
    overlap = len(run_ids & qrel_ids)
    print(f"  [check/{label}] qrels docid {len(qrel_ids)} 件中 {overlap} 件が run に出現")
    if overlap == 0:
        print(f"  [ERROR/{label}] docid が 1 件も一致しない。粒度を確認すること。",
              file=sys.stderr)


def main():
    print("読み込み中...")
    if not os.path.exists(WEBSTYLE_FILE):
        raise SystemExit(
            f"{WEBSTYLE_FILE} がありません。先に "
            f"`python decomposed_query2doc_webstyle_expansion.py submit` → "
            f"`fetch` で生成してください。")
    with open(WEBSTYLE_FILE) as f:
        webstyle = json.load(f)["results"]
    queries = load_queries(QUERIES_FILE)
    qrels = {name: load_qrels(os.path.join(DATA_DIR, f"keystone_qrels_{name}.txt"))
             for name in QREL_SETS}

    valid_qids = sorted(
        qid for qid in webstyle
        if qid in queries
        and webstyle[qid].get("decomposed_queries")
        and any(d and d.strip() for d in webstyle[qid].get("query2doc_docs", []))
    )
    print(f"\n疑似文書がある {len(valid_qids)} クエリ")
    for name in QREL_SETS:
        print(f"  qrels[{name}]: {len(qrels[name])} qids / "
              f"うち対象内 {len(set(valid_qids) & set(qrels[name]))} qids")
    if not valid_qids:
        print("採点対象が空。疑似文書ファイルの生成状況を確認すること。",
              file=sys.stderr)
        return

    print(f"条件: {', '.join(METHODS)}   TOPK={TOPK}   RETRIEVE_K={RETRIEVE_K}   "
          f"QUERY_REPEAT={QUERY_REPEAT}")
    print("=" * 72)

    search_body = CachedSearch(bm25_body, RETRIEVE_K)
    search_keyterms = CachedSearch(bm25_keyterms, RETRIEVE_K)
    search_title = CachedSearch(bm25_title, RETRIEVE_K)
    search_headings = CachedSearch(bm25_headings, RETRIEVE_K)
    fielded_searchers = {
        "webstyle_narrative_fielded": CachedFieldedSearch(bm25_fielded, RETRIEVE_K),
        "webstyle_narrative_fielded_hboost": CachedFieldedSearch(bm25_fielded_hboost, RETRIEVE_K),
        "webstyle_narrative_fielded_regboost": CachedFieldedSearch(bm25_fielded_regboost, RETRIEVE_K),
        "webstyle_narrative_fielded_recallopt": CachedFieldedSearch(bm25_fielded_recallopt, RETRIEVE_K),
        "webstyle_narrative_fielded_ndcgopt": CachedFieldedSearch(bm25_fielded_ndcgopt, RETRIEVE_K),
    }
    runs = {m: build_run(m, valid_qids, queries, webstyle,
                          search_body, search_keyterms, search_title, search_headings,
                          fielded_searchers)
            for m in METHODS}
    print(f"\n{search_body.stats('bm25_body')}")
    print(search_keyterms.stats('bm25_keyterms'))
    print(search_title.stats('bm25_title'))
    print(search_headings.stats('bm25_headings'))
    for name, searcher in fielded_searchers.items():
        print(searcher.stats(name))

    eval_qids = [q for q in valid_qids if all(runs[m].get(q) for m in METHODS)]
    if len(eval_qids) < len(valid_qids):
        print(f"[warn] 検索結果が空になった {len(valid_qids) - len(eval_qids)} クエリを"
              f"全条件から除外", file=sys.stderr)
    print(f"最終採点対象: {len(eval_qids)} クエリ（全条件共通）")

    print("\n--- docid 粒度チェック ---")
    for name in QREL_SETS:
        check_docid_overlap(runs["baseline"], qrels[name], eval_qids, name)

    summary, n_seen = {}, {}
    for method in METHODS:
        print(f"\n### {method} ###")
        for name in QREL_SETS:
            agg, n = evaluate(runs[method], qrels[name], eval_qids, f"{method} / {name}")
            summary.setdefault(name, {})[method] = agg
            n_seen.setdefault(name, set()).add(n)

    for name in QREL_SETS:
        if len(n_seen[name]) > 1:
            print(f"[ERROR] qrels[{name}] の n が条件間で不一致: {sorted(n_seen[name])}",
                  file=sys.stderr)

    print("\n" + "=" * 72)
    print(f"SUMMARY   対象: {len(eval_qids)} クエリ")
    for name in QREL_SETS:
        print(f"\n[{name}]")
        labels = {"recall_100": "recall@100", "recall_1000": "recall@1000",
                  "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
        header = "method".ljust(28) + "".join(
            labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(28)
            if agg is None:
                print(row + "-")
                continue
            for key in METRIC_KEYS:
                cell = f"{agg[key]:.4f}"
                if method != "baseline" and base:
                    cell += f" ({agg[key] - base[key]:+.4f})"
                row += cell.ljust(16)
            print(row)
    print("=" * 72)

    out_path = os.path.join(RAG_DIR, f"decomposed_query2doc_webstyle_eval_summary_quotaunion_rep{QUERY_REPEAT}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
