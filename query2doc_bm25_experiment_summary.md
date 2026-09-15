# Query2doc（zero-shot / few-shot）× BM25 実験まとめ（2026-08-30）

> 用語について: 当初「HyDE」と呼んでいたが、embeddingを一切使わずBM25（疎検索）の
> クエリ文字列として疑似文書を連結しているだけなので、正確には**Query2doc**
> （Wang et al. 2023）に近い手法。このドキュメントでは以後この呼称に統一する。

## 結論だけ先に

- **今回のパイプラインは本来のHyDE（dense embedding版）ではない。** 疑似文書をembeddingせず、テキストのままBM25クエリに連結しているだけ。実態は **Query2doc** / MuGI に近い「LLM生成疑似文書によるBM25向けクエリ拡張」。
- narrative全体から複数本のQuery2doc疑似文書を生成し、**元クエリ（narrative）を5回リピートしてから連結**する方式（`query2doc_k`, QUERY_REPEAT=5）が、今回試した中で最も良い結果だった。
- narrativeを**Subquery（分解した簡潔な質問文）に分割してからQuery2doc**する方式は、分割しない方式（narrative全体から複数疑似文書）に一貫して劣った。
- プロンプトを**zero-shot→few-shot（(query, document)の例を3件添付）に変えると、どちらの方式も全指標で改善**した。ただし優劣（非分解版が優位）は逆転しなかった。
- 途中でLLMモデルを `openai/gpt-5.6-terra` → `google/gemini-3.7-flash` に切り替えている（理由は後述）。**過去の結果と比較する際はこの点に注意。**

---

## 1. 背景

TREC 2025 RAG Track の narrative形式クエリ（1クエリが数文の長い自然文）に対して、Query2doc（LLM生成疑似文書）でクエリ拡張を行い、BM25検索の精度がどう変わるかを検証した。

## 2. データ

| 項目 | 内容 |
|---|---|
| クエリ | `~/data/trec_rag_2025_queries.jsonl`（105トピック、narrative形式） |
| 検索対象 | OpenSearch `msmarco-v21-doc`（MS MARCO v2.1 文書レベル、約1096万件） |
| qrels（正解） | `keystone_qrels_coverage.txt`（22トピック） / `keystone_qrels_consensus.txt`（105トピック、より広くカバー） |
| 検索方式 | BM25（`match` query、密ベクトル検索は不使用） |

## 3. 手法の詳細 — 本来のHyDEとの違い（重要）

本来のHyDE（Gao et al. 2022, *Precise Zero-Shot Dense Retrieval without Relevance Labels*）は、

1. LLMで疑似文書（Pseudo Reference）を生成
2. その疑似文書を **Dense Retrieverでembedding** し
3. **ベクトル類似度検索（ANN）** で文書を検索する

という、embedding空間での操作が本質。

**今回実装したパイプラインは、この2〜3を一切やっていない。** `retriever.py` の検索関数（`bm25_body` など）はOpenSearchの `match` query（BM25スコアリング）のみを使っており、`knn` や密ベクトル検索は使っていない。疑似文書は**生成したテキストをそのまま**BM25の検索クエリ文字列として（元クエリと連結して）使っているだけ。

つまり実態は **Query2doc**（Wang et al. 2023）や **MuGI** が扱っている手法設定そのもの。特にQuery2docは「元クエリを複数回繰り返してから疑似文書と連結し、疎検索にかける」という手法で、これは元クエリの語がBM25の項頻度（TF）で疑似文書の語に埋もれてしまうのを防ぐためのテクニック。`QUERY_REPEAT`（元クエリの繰り返し回数）を1→5に増やしただけで全指標が明確に改善したのは、この知見と一致する。

コード上も、生成した疑似文書やその関連キー名は `query2doc_*`（`query2doc_docs`, `query2doc_results`, `query2doc_k` など）に統一済み。

## 4. 使用モデル・プロンプト方式（2軸で変遷あり）

| 軸 | 変遷 |
|---|---|
| モデル | `openai/gpt-5.6-terra` → `google/gemini-3.7-flash:batch` |
| プロンプト方式 | zero-shot → few-shot（(query, document)の例を3件添付） |

**モデルを切り替えた理由**: OpenRouterのBatch APIが、OpenAI系モデル（`gpt-5.6-terra`, `gpt-4o`, `gpt-4o-mini`等）に対して `Model does not have a :batch endpoint` エラーを返すようになった。調査の結果、OpenRouterの`:batch`対応モデル一覧からOpenAI系モデルが軒並み外れており（Anthropic/Google/Mistral等は残っている）、プラットフォーム側の仕様変更と判断。代替として`google/gemini-3.7-flash:batch`に切り替えた。

**プロンプトをfew-shot化した理由**: Query2doc論文の本来の設定はfew-shotであり、zero-shotのみでの評価は片手落ちという指摘を受けたため、両方で比較した。

実験条件と結果の対応は下表の通り（詳細は[`experiments/README.md`](experiments/README.md)）:

| 条件 | フォルダ | 状態 |
|---|---|---|
| gpt-5.6-terra, zero-shot | `experiments/gpt56terra_zeroshot/` | 中断（モデル切替のため） |
| gemini-3.7-flash, zero-shot | `experiments/gemini_zeroshot/` | 完了 |
| gemini-3.7-flash, few-shot | `experiments/gemini_fewshot/` | **現行・完了**（rep1は未実行） |

## 5. パイプライン構成

| スクリプト | 役割 |
|---|---|
| `narrative_expansion.py` | narrative全体から独立にQuery2doc疑似文書をN_POOL=30本生成（1文書=1リクエスト、約200語/本、few-shot） |
| `decompose_narrative.py` | narrativeを Subquery（簡潔な質問文、可変個数、平均4〜5個/トピック）に分解 |
| `decomposed_query2doc_expansion.py` | 各Subqueryにつき1本のQuery2doc疑似文書を生成（few-shot） |
| `retriever.py` | `bm25_body()`（BM25検索）、`rrf_fuse()`（Reciprocal Rank Fusion, k=60） |
| `evaluate_rep1.py` / `evaluate_rep5.py` | narrative全体プール版の評価。`query2doc_k`: `(narrativeをQUERY_REPEAT回繰り返し) + 疑似文書` をk本それぞれ検索しRRF融合 |
| `evaluate_decomposed_rep1.py` | Subquery版の評価（`narrative + 疑似文書`を各Subquery分RRF融合、QUERY_REPEATなし） |
| `evaluate_decomposed_variants.py` | Subquery版で、クエリ組み立て方を3通り比較 |

**評価指標**: recall@100, recall@1000, nDCG@10, precision@100（`pytrec_eval`使用）、TOPK=1000。

## 6. 結果

### A. narrative全体からのQuery2doc疑似文書プール（query2doc_10, QUERY_REPEAT=5, consensus n≈103-105）

| 条件 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline（narrativeのみ） | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| zero-shot | 0.0354 (+0.0026) | 0.1476 (-0.0019) | 0.3142 (+0.0493) | 0.4047 (+0.0310) |
| **few-shot** | **0.0361 (+0.0036)** | **0.1490 (+0.0019)** | **0.3266 (+0.0632)** | **0.4137 (+0.0421)** |

few-shot化でrecall@1000もマイナスからプラスに転換。k本数は2〜3本程度でほぼ性能が飽和し、30本まで増やしても顕著な改善はなかった。

### B. Subquery版（query2doc_dq, QUERY_REPEAT=5, consensus）

| 条件 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| zero-shot | 0.0338 (+0.0011) | 0.1471 (-0.0024) | 0.2865 (+0.0216) | 0.3870 (+0.0132) |
| **few-shot** | **0.0342 (+0.0022)** | **0.1489 (+0.0012)** | **0.2964 (+0.0336)** | **0.3970 (+0.0266)** |

こちらもfew-shotで全指標が改善。dq_pseudodoc（narrativeを使わない版）もnDCG@10が-0.0060→+0.0100とプラスに転換した。

### C. 最終比較（両方few-shot・QUERY_REPEAT=5・consensus）

| | query2doc_dq（分解版） | query2doc_10（非分解版） |
|---|---|---|
| recall@100 | 0.0342 | **0.0361** |
| recall@1000 | **0.1489** | 0.1490（ほぼ同着） |
| nDCG@10 | 0.2964 | **0.3266** |
| precision@100 | 0.3970 | **0.4137** |

**結論**: few-shot化はどちらの方式にも効いたが、公平な条件（両方few-shot）で比較しても**非分解版が依然として優位**。recall@1000はほぼ同着まで詰まったが、nDCG@10・recall@100・precision@100は非分解版が一貫して上回る。narrativeを分解する工夫そのものは、zero-shotでもfew-shotでも効果が確認できなかった。

## 7. 今後の課題

- **本物のHyDE（dense embedding）との比較実験が必要。** 今回はBM25限定なので、密検索（Contrieverやbi-encoder等）でのHyDEも別途試すべき。特にMuGIはdense/sparse両対応の拡張手法を提案しているので、そちらも参考になる。
- Subquery版のk本数（現状1問=1本固定）を増やして、非分解版と同条件で比較する余地もある。
- `experiments/gemini_fewshot/`のrep1（QUERY_REPEAT=1）は未実行（rep5が本命だったため優先度を下げた）。

## 8. コスト・所要時間（gemini-3.7-flash、few-shot分含む）

- narrative全体プール生成（105トピック×30本）: 約$1.5〜1.6
- Subquery分解（105トピック）: 約$0.03
- Subquery毎の疑似文書生成（451本）: 約$0.2
- 評価（BM25検索のみ、LLM API不使用）: 追加コストなし

## 9. コード・データの場所

- コード: `/home/takeuchi/rag-experiment/`（gitリポジトリ、リモート: `git@github.com:Kanikama999/rag-experiment.git`）
  - ※ このセッションで追加・変更したファイルは**まだコミットしていません**。共有時はpushが必要。
- 生成データ・評価結果: `experiments/<実験条件>/`配下にコードと一緒に条件ごとに整理済み。詳細は[`experiments/README.md`](experiments/README.md)
- `~/data/`には外部入力ファイル（クエリ・qrels）のみ。生成物のコピーは廃止した
