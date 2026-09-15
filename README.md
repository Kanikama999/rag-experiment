# rag-experiment

TREC 2025 RAG Track（narrative形式のクエリ）に対する、LLM生成疑似文書によるクエリ拡張＋BM25/RRFの検索実験。

## 用語について（重要）

このリポジトリの手法は「HyDE」ではない。本来のHyDE（Gao et al. 2022, *Precise
Zero-Shot Dense Retrieval without Relevance Labels*）は、生成した疑似文書を
**dense retrieverでembeddingし、ベクトル類似度検索**にかける手法。

ここで実装しているのは、疑似文書のテキストをそのまま**BM25（疎検索）のクエリ文字列**
として（多くの場合、元クエリと連結して）使う方式であり、embeddingは一切使っていない。
これは **Query2doc**（Wang et al. 2023）や **MuGI** が扱う「LLM生成疑似文書による
疎検索向けクエリ拡張」に近い。元クエリを複数回繰り返してから連結する
（`QUERY_REPEAT`）ことで性能が大きく改善する現象も、Query2doc/MuGIの知見と一致する。

コード中では、生成した疑似文書そのものを指す変数・キー名に `query2doc` を使っている
（例: `query2doc_docs`, `query2doc_results`, `query2doc_k`）。

## 全体の流れ

```
narrative(クエリ) --[LLMでQuery2doc疑似文書生成]--> multi_query2doc_*.json
                                                     |
                                                     v
              (元クエリ×N回) + Query2doc疑似文書 で msmarco-v21-doc を検索(RRF融合)
                                                     |
                                                     v
                              qrelsで採点(recall/nDCG/precision)
```

もう一段階として、narrativeを複数の Subquery（簡潔な質問文）に分解し、
質問ごとにQuery2doc疑似文書を1本ずつ生成する実験系列もある。

オラクル実験は、LLM生成の疑似文書の代わりに**TRECの人間が正解判定した本物のセグメント
本文**をクエリ拡張に使い、「疑似文書が理想的に書けたらどこまで伸びるか」という上限値を
測るもの。

## 構成（`experiments/`）

**1フォルダ = 1実験条件（モデル × プロンプト方式）。コードと結果を同じフォルダに
まとめてあり、各フォルダをそのまま`cd`して実行すれば完結する**（`retriever.py`も
フォルダごとに複製済みなので、追加のimportパス設定は不要）。

```
experiments/
  gpt56terra_zeroshot/   旧モデル(gpt-5.6-terra)・zero-shot（中断済み）
  gemini_zeroshot/       現行モデル(gemini-3.7-flash)・zero-shot
  gemini_fewshot/        現行モデル・few-shot（現在の本命設定）
  oracle/                オラクル実験（正解セグメント使用）
```

どのフォルダに何が入っているか、各スクリプトの役割は
**[`experiments/README.md`](experiments/README.md)** 参照。

以前は結果を`~/data/`にもコピーする運用だったが、二重管理で分かりにくかったため廃止
した。`~/data/`には元々の外部入力ファイル（クエリ・qrels）のみ残す。

## 使用モデル・プロンプト方式の変遷

1. `openai/gpt-5.6-terra`（OpenRouter Batch API, zero-shot）で開始 → `experiments/gpt56terra_zeroshot/`
2. 2026-08-28頃、OpenRouter側でOpenAI系モデルがBatch API対応から外れる仕様変更が発生し、
   `google/gemini-3.7-flash:batch`（zero-shot）に切り替え → `experiments/gemini_zeroshot/`
3. Query2doc論文本来の設定（few-shot）に合わせ、疑似文書生成プロンプトを
   zero-shot→few-shot（(query, document)の例を3件添付）に変更 → `experiments/gemini_fewshot/`（現行）

**結論（2026-08-30時点）**: narrativeを分解しない方式（`query2doc_k`）の方が、
分解する方式（Subquery系）より全指標で一貫して優位。few-shot化・
QUERY_REPEAT=5はどちらの方式にも効くが、優劣自体は逆転しない。詳細は
[`query2doc_bm25_experiment_summary.md`](query2doc_bm25_experiment_summary.md)参照。

## 検索対象（OpenSearch, `opensearch-msmarco`コンテナ, `localhost:9200`）

| インデックス | 件数 | 内容 |
|---|---|---|
| `msmarco-v21-doc` | 1096万件 | MS MARCO v2.1 文書レベル（フルドキュメント）。メインの検索対象 |
| `qrel-segment-pool` | 10,281件 | qrelで正解判定されているセグメントのみ。現状どの評価パイプラインからも検索対象としては未使用（オラクル実験の「クエリ側の材料」の元データを別途JSONで持っているので、こちらは参照用） |

セグメント単位（MS MARCO v2.1 doc-segmented、TREC Task Rの検索対象）で
本格的に検索したい場合は、pyserini配布のビルド済みLuceneインデックス
（`msmarco-v2.1-doc-segmented`、84GB）を使うのが現実的。自前でOpenSearchに
全量インデックス化すると150〜250GB級になり非現実的（ストレージ確保を別途相談中）。

## トップレベルのファイル

- `retriever.py` — BM25検索とRRF融合の共通関数のマスターコピー（`bm25_body`/`bm25_keyterms`/`bm25_qrelpool`/`rrf_fuse`）。各`experiments/*/`フォルダに複製して使っているので、バグ修正時は複製し直すこと
- `search.py` — 動作確認用の最小サンプル（`requests`で直接叩くだけ、特定の実験と紐付かない）
- `narrative_expansion_batch.py` / `unused_narrative_expansion_batch/` — （非推奨・現在未使用）1トピック=1リクエストでN本まとめて生成しようとしたが、要求30本に対し実際は平均10本程度しか返らず不完全なため使っていない
- `query2doc_bm25_experiment_summary.md` — 実験結果の詳細まとめ（研究室内共有用）

`~/data/`にある関連ファイル（外部入力のみ。生成物は置かない）:

- `trec_rag_2025_queries.jsonl` — 105トピックのnarrativeクエリ
- `keystone_qrels_coverage.txt` / `keystone_qrels_consensus.txt` — 文書レベルのqrels（105トピック中の一部をカバー）
- `2025-rag-qrels.txt` — TREC公式の生qrels（セグメント単位、22トピックのみ）
