# experiments/ 索引

1フォルダ = 1つの実験条件（モデル × プロンプト方式）。**コードと結果を同じフォルダに
まとめてあり、各フォルダ単体で完結する**（`retriever.py`も複製済み）。

| フォルダ | モデル | プロンプト方式 | 状態 |
|---|---|---|---|
| `gpt56terra_zeroshot/` | `openai/gpt-5.6-terra` | zero-shot | 最初期の実験。OpenRouterがOpenAI系モデルのBatch APIを打ち切ったため中断 |
| `gemini_zeroshot/` | `google/gemini-3.7-flash` | zero-shot | Geminiに切り替え後の最初の実験一式。narrative全体プール版・Decomposed_Query版とも完備（rep1/rep5両方） |
| `gemini_fewshot/` | `google/gemini-3.7-flash` | few-shot（3例） | **現行**。narrative全体プール版・Decomposed_Query版とも完備だが、rep1（QUERY_REPEAT=1）は未実行（rep5の方が良い結果だったため） |
| `oracle/` | — | — | LLM生成の代わりに、TRECの人間正解セグメントをクエリ拡張に使うオラクル上限実験（22トピックのみ） |

各フォルダをそのまま`cd`して`python narrative_expansion.py submit`のように実行できる
（`retriever.py`はフォルダ内にコピーがあるので追加のimportパス設定は不要）。

## 各フォルダの中身（narrative全体プール版・Decomposed_Query版に共通するもの）

- `narrative_expansion.py` — narrative全体からQuery2doc疑似文書プール（30本）を生成するスクリプト
- `decompose_narrative.py` / `decomposed_queries.json` — narrativeをDecomposed_Queryに分解するスクリプトとその出力（`gpt56terra_zeroshot/`・`oracle/`にはない。Decomposed_Query系はGemini移行後に作った手法のため）
- `decomposed_query2doc_expansion.py` — Decomposed_Query 1問につき1本のQuery2doc疑似文書を生成するスクリプト
- `multi_query2doc_L200.json` — 上記`narrative_expansion.py`の出力
- `multi_query2doc_decomposed_L200.json` — 上記`decomposed_query2doc_expansion.py`の出力
- `evaluate_rep{1,5}.py` / `query2doc_eval_summary_rep{1,5}.json` — narrative全体プール版の評価スクリプトと結果（`{1,5}` = QUERY_REPEAT）
- `evaluate_decomposed_rep1.py` / `decomposed_query2doc_eval_summary.json` — Decomposed_Query版の単純評価
- `evaluate_decomposed_variants.py` / `decomposed_query2doc_variants_eval_summary_rep{1,5}.json` — Decomposed_Query版の3手法比較
- `batch_ids_*.json` — OpenRouter Batch APIのジョブID記録（生成時の進捗追跡用、再利用不要）
- `retriever.py` — BM25検索・RRF融合の共通関数（全フォルダ共通、複製）

`oracle/`のみ構成が異なり、`qrel_segment_pool.py`（正解セグメントpool生成）、
`build_qrel_segment_index.py`、`evaluate_qrelpool{,_rep5}.py`とその結果一式。

`gpt56terra_zeroshot/`のみDecomposed_Query系がない（narrative全体プール版のみ）。

## 結論（詳細は `../query2doc_bm25_experiment_summary.md` 参照）

- narrativeを分解しない方式（`query2doc_k`）が、分解する方式（Decomposed_Query系）より常に優位
- few-shot化・QUERY_REPEAT=5は、どちらの方式にも共通して効いた
