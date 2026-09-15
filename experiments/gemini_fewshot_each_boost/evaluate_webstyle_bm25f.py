"""
真のBM25F（Robertson/Zaragoza/Taylor 2004、retriever.py の bm25f()）を、既存の最良
fielded設定（webstyle_narrative_fielded_recallopt、title=2/headings=1/body=1の線形和）
と同じフィールド重みで比較する。

bm25_fielded系（recallopt含む）はtitle/headings/bodyそれぞれ独立にBM25（フィールドごとの
IDF・文書長正規化・tf飽和）を計算してからスコアを線形和するだけで、フィールドをまたいだ
真のBM25Fではない（REPORT.md 4.3節で報告した「BM25生スコアはクエリ間でスケールが揃わない」
問題は、この独立計算構造に起因する可能性がある）。bm25f()はフィールドごとの生tfを
フィールド長で正規化してから重み付き合算し、単一のIDF・単一のtf飽和関数を1回だけ適用する
（詳細はretriever.py bm25f()のdocstring参照）。

条件（2つのみ、baselineは参考値）:
- baseline: narrativeそのままで bm25_body
- webstyle_narrative_fielded_recallopt: 既存の最良fielded設定（title=2,headings=1,body=1
  の線形和bm25_fielded）。参照用。
- webstyle_narrative_fielded_bm25f: 同じフィールド重み（title=2,headings=1,body=1）を
  bm25f()に渡した版。フィールド重みを揃えることで「線形和 vs 真のBM25F」という
  結合方式そのものの違いだけを切り分ける。

bm25f()はOpenSearchにBM25Fクエリ型が無いため2段構成（bool/should検索でcandidate_k件の
候補プール→_mtermvectorsでフィールド別tf取得→Python側で厳密計算・re-rank）になっており、
候補プールを超えて新規文書を発見できない。RETRIEVE_K=3000（他条件と同じ）をcandidate_kに
使うため、bm25f呼び出し1回が約7〜8秒かかる（プロファイル済み）。105トピック×平均4.5
Subquery ≈ 470回で、bm25f単体だけで小一時間かかる見込み。

使い方:
    python evaluate_webstyle_bm25f.py [QUERY_REPEAT] [SUBSET_STRIDE]
    # QUERY_REPEAT省略時は5。SUBSET_STRIDE省略時は1（全105トピック）。
    # search_field_boosts.pyの探索フェーズと同じ間引き方（valid_qids[::STRIDE]）で
    # トピック数を間引ける（bm25fは1回の呼び出しに約19秒かかるため、全105トピックだと
    # 2時間半規模になる。STRIDE=3で約35トピックに間引くと約50分に収まる）。
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytrec_eval

from retriever import bm25_body, bm25_fielded, bm25f, rrf_fuse


def bm25_fielded_recallopt(title_text, headings_text, body_text, k=100):
    """search_field_boosts.pyで見つかった既存最良比率（title=2,headings=1,body=1）。"""
    return bm25_fielded(title_text, headings_text, body_text, k=k,
                         title_boost=2, headings_boost=1, body_boost=1)


def bm25f_recallopt(title_text, headings_text, body_text, k=100):
    """recalloptと同じフィールド重み（title=2,headings=1,body=1）を真のBM25Fに渡す版。
    b_title/b_headings/b_bodyはこのインデックスのbody用similarity設定をそのまま流用した
    デフォルト値（0.4/0.4/0.4）。search_bm25f_field_b.py（35トピックサブセット、
    candidate_k=200での探索）でこれが最適でないことが分かったため、bm25f_tuned()も
    参照のこと。candidate_k/rerank_nはkと同じ（＝呼び出し元のRETRIEVE_K）にし、他条件と
    同じ候補プールサイズで比較する。"""
    return bm25f(title_text, headings_text, body_text, k=k,
                 title_weight=2.0, headings_weight=1.0, body_weight=1.0,
                 candidate_k=k, rerank_n=k)


def bm25f_tuned(title_text, headings_text, body_text, k=100):
    """bm25f_recalloptと同じフィールド重みだが、b_title/b_headings/b_bodyを
    search_bm25f_field_b.py（35トピックサブセット、candidate_k=200での探索、
    bm25f_field_b_search_result.json）で見つかった最良値に変更した版
    （b_title=0.6, b_headings=0.2, b_body=0.2。デフォルト一律0.4に対しnDCG@10が
    35トピックサブセットで0.4118→0.4467、+8.5%相対改善）。titleは通常のbodyより
    強い長さ正規化（b=0.6）、headingsは弱い正規化（b=0.2）が最良という、
    「短いフィールドほど正規化を弱める」という当初の仮説とは逆の結果だった点に注意
    （bm25f_field_b_search_result.json参照）。"""
    return bm25f(title_text, headings_text, body_text, k=k,
                 title_weight=2.0, headings_weight=1.0, body_weight=1.0,
                 b_title=0.6, b_headings=0.2, b_body=0.2,
                 candidate_k=k, rerank_n=k)


RAG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(RAG_DIR, "..", "..", "..", "data")
WEBSTYLE_FILE = os.path.join(RAG_DIR, "multi_query2doc_decomposed_webstyle_L200.json")
QUERIES_FILE = os.path.join(DATA_DIR, "trec_rag_2025_queries.jsonl")

QREL_SETS = ["coverage", "consensus"]
TOPK = 1000
RETRIEVE_K = 3000
QUERY_REPEAT = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SUBSET_STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1

METHODS = ["baseline", "webstyle_narrative_fielded_recallopt", "webstyle_narrative_fielded_bm25f_tuned"]

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
    """search_fn(title, headings, body, k) 用。3テキストのタプルでキャッシュする。"""

    def __init__(self, fn, topk):
        self.fn = fn
        self.topk = topk
        self._cache = {}
        self.calls = self.hits = 0
        self.total_time = 0.0

    def __call__(self, title_text, headings_text, body_text):
        key = (title_text.strip(), headings_text.strip(), body_text.strip())
        if not any(key):
            return []
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        t0 = time.time()
        self._cache[key] = self.fn(*key, k=self.topk)
        self.total_time += time.time() - t0
        return self._cache[key]

    def stats(self, label):
        avg = self.total_time / self.calls if self.calls else 0.0
        return (f"{label}: 呼び出し {self.calls} 回 / キャッシュヒット {self.hits} 回 / "
                f"合計 {self.total_time:.0f}s (平均 {avg:.2f}s/回)")


def fuse(lists, topk):
    lists = [lst for lst in lists if lst]
    if not lists:
        return {}
    return {docid: float(score) for docid, score in rrf_fuse(lists, top_n=topk)}


def dq_pairs_structured(entry):
    return [(dq, doc) for dq, doc in zip(entry["decomposed_queries"],
                                          entry["query2doc_docs_structured"])
            if doc and doc.get("body")]


def build_run(method, qids, queries, webstyle, search_body, fielded_searchers):
    run, t0 = {}, time.time()
    for i, qid in enumerate(qids, 1):
        q = queries[qid]

        if method == "baseline":
            run[qid] = fuse([search_body(q)], TOPK)
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
        else:
            raise ValueError(f"未知の method: {method}")

        if i % 5 == 0 or i == len(qids):
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
        raise SystemExit(f"{WEBSTYLE_FILE} がありません。")
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
    if SUBSET_STRIDE > 1:
        valid_qids = valid_qids[::SUBSET_STRIDE]
        print(f"  SUBSET_STRIDE={SUBSET_STRIDE} で間引き: {len(valid_qids)} クエリを使用"
              f"（search_field_boosts.pyの探索フェーズと同じ間引き方）")
    for name in QREL_SETS:
        print(f"  qrels[{name}]: {len(qrels[name])} qids / "
              f"うち対象内 {len(set(valid_qids) & set(qrels[name]))} qids")
    if not valid_qids:
        print("採点対象が空。疑似文書ファイルの生成状況を確認すること。", file=sys.stderr)
        return

    print(f"条件: {', '.join(METHODS)}   TOPK={TOPK}   RETRIEVE_K={RETRIEVE_K}   "
          f"QUERY_REPEAT={QUERY_REPEAT}   SUBSET_STRIDE={SUBSET_STRIDE}")
    print("=" * 72)

    search_body = CachedSearch(bm25_body, RETRIEVE_K)
    fielded_searchers = {
        "webstyle_narrative_fielded_recallopt": CachedFieldedSearch(bm25_fielded_recallopt, RETRIEVE_K),
        "webstyle_narrative_fielded_bm25f_tuned": CachedFieldedSearch(bm25f_tuned, RETRIEVE_K),
    }
    runs = {m: build_run(m, valid_qids, queries, webstyle, search_body, fielded_searchers)
            for m in METHODS}
    print(f"\n{search_body.stats('bm25_body')}")
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
        header = "method".ljust(38) + "".join(labels[k].ljust(16) for k in METRIC_KEYS)
        print(header)
        print("-" * len(header))
        base = summary[name].get("baseline")
        for method in METHODS:
            agg = summary[name].get(method)
            row = method.ljust(38)
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

    suffix = f"_stride{SUBSET_STRIDE}" if SUBSET_STRIDE > 1 else ""
    out_path = os.path.join(RAG_DIR, f"webstyle_bm25f_tuned_eval_summary_rep{QUERY_REPEAT}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump({"n_queries": len(eval_qids), "topk": TOPK,
                   "qids": eval_qids, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"集計値を保存: {out_path}")


if __name__ == "__main__":
    main()
