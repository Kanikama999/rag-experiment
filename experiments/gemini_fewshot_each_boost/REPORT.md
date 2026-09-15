# Web風構造化疑似文書 × フィールド別BM25検索 実験まとめ（2026-09-01）

> このフォルダ（`experiments/gemini_fewshot_2/`）は、論文執筆用に必要なコード・生成データ・
> 評価結果だけを `experiments/gemini_fewshot/` から抜き出したものである。作業の全履歴は
> `experiments/gemini_fewshot/` 側に残っている。

## 結論だけ先に

- Subquery（narrativeを分解した簡潔な質問文）ごとに、LLMへ**「そのトピックについての
  webページ」のスタイル（title / headings / body の3フィールドを持つJSON）で疑似参照文を
  生成させ**、生成されたtitleはOpenSearch文書側のtitleフィールド、headingsはheadingsフィールド、
  bodyはbodyフィールドへ**個別にBM25でmatchし、3フィールド分のスコアを線形和で合算する**
  （`bm25_fielded`、bool/should）検索方式が、今回試した中で最も良い結果だった。
- 検索クエリには narrative（元の長いクエリ）を**5回繰り返して**Subqueryと疑似文書に
  連結している（`QUERY_REPEAT=5`）。narrativeを検索クエリから外す、あるいは繰り返し回数を
  減らすと性能が大きく落ちる。
- フィールドブースト比（title^?, headings^?, body^1）は、**qrelsに対する線形回帰では
  うまく推定できなかった**（クエリ間でBM25生スコアのスケールが揃っていないため）。実際の
  パイプラインで評価指標を直接最大化する探索（座標降下法、および4×4の総当たりグリッドサーチ
  で同じ結論を再確認）の方が信頼できた。
  - recall@1000を優先するなら **title=2, headings=1, body=1**
  - nDCG@10を優先するなら **title=3, headings=3, body=1**
  - どちらも既定の(title=3, headings=2, body=1)に近いが、意味のある差でどちらの指標も
    さらに改善できる。
- title/headings/bodyを単体で使った場合はheadings単体が最も強く、次いでtitle、bodyの順
  だった。ただし3フィールドを合算すると単体のどれよりも大幅に強くなり、**単体で強いフィールド
  にブーストを寄せれば良いわけではない**（headingsの重みを上げると全指標で悪化した）ことも
  分かった。

---

## 1. 背景・目的

TREC 2025 RAG Track の narrative形式クエリに対し、`experiments/gemini_fewshot/`で
Subquery単位のQuery2doc（LLM生成疑似文書）× BM25の実験を重ねてきた
（詳細は `query2doc_bm25_experiment_summary.md` 参照）。本フォルダでは、その延長として

1. 疑似文書を**web風に構造化**（title/headings/body）して生成する
2. 生成した各パートを**対応するインデックスフィールドへ個別にBM25検索**する
3. フィールドごとの**ブースト比を最適化**する

という3点を検証した。

## 2. パイプライン

```
narrative（元クエリ） --[decompose_narrative.py]--> Subquery（複数、簡潔な質問文）
                                                          |
                        [decomposed_query2doc_webstyle_expansion.py]
                                                          |
                                                          v
                        疑似参照文（JSON: title / headings / body）
                                                          |
                    narrative×5 + Subquery + 各フィールドのテキスト
                                                          |
              title -> titleフィールド、headings -> headingsフィールド、
              body -> bodyフィールドへ個別にBM25 match（bm25_fielded, retriever.py）
                                                          |
                    3フィールド分のスコアを線形和（bool/should）
                                                          |
              Subquery単位の結果をRRF融合（k=60）してトピックの最終順位に
```

検索対象は OpenSearch `msmarco-v21-doc`（MS MARCO v2.1 **文書レベル**、約1096万件。
セグメント単位のインデックスではない）。

## 3. 使用ファイル

| ファイル | 役割 |
|---|---|
| `decompose_narrative.py` | narrative → Subquery への分解 |
| `decomposed_queries.json` | 上記の出力（105トピック分） |
| `decomposed_query2doc_webstyle_expansion.py` | Subquery単位でweb風疑似文書（title/headings/body）をLLM生成 |
| `multi_query2doc_decomposed_webstyle_L200.json` | 上記の出力（451本、1トピック平均4〜5本） |
| `retriever.py` | BM25検索関数一式（`bm25_body`, `bm25_title`, `bm25_headings`, `bm25_keyterms`, `bm25_fielded`, `rrf_fuse`） |
| `evaluate_decomposed_webstyle.py` | 各手法の評価スクリプト（baseline / フィールド別検索 / アブレーション） |
| `learn_field_boosts.py` | ブースト比を線形回帰で推定する試み（`field_boost_regression.json`。**うまくいかなかった手法**として記録） |
| `search_field_boosts.py` | ブースト比を実パイプラインで直接探索するスクリプト（`field_boost_search_result.json` / `field_boost_search_result_coordinate_ascent_5x5.json`） |
| `decomposed_query2doc_webstyle_eval_summary_rep5.json` | baseline・既定ブースト・recallopt・ndcgoptの105トピック評価結果 |
| `decomposed_query2doc_webstyle_field_ablation_rep5.json` | title/headings/body単体のアブレーション結果（105トピック） |

すべて `QUERY_REPEAT=5`、qrelsは `keystone_qrels_coverage.txt`（22トピック、厳密寄り）と
`keystone_qrels_consensus.txt`（105トピック、より広くカバー）の両方で評価している。

## 4. 結果

### 4.1 基本手法（既定ブースト title=3, headings=2, body=1）

| method | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline（narrativeそのままbm25_body） | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| **webstyle_narrative_fielded** | 0.0620 | 0.2603 | 0.5028 | 0.7020 |

（consensus、105クエリ。coverageでも同じ傾向: baseline recall@1000=0.2884 → fielded 0.5182）

narrativeを検索クエリに含めない、あるいは単純なフィールド無視のBM25検索（`bm25_body`）
に比べて、全指標で大幅な改善が確認できた。

### 4.2 title/headings/body単体のアブレーション

同じtitle_q/headings_q/body_qを使い、対応するフィールド1つだけをmatchした場合
（`decomposed_query2doc_webstyle_field_ablation_rep5.json`）。

| method | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| title単体 | 0.0363 | 0.1421 | 0.2966 | 0.4033 |
| **headings単体** | **0.0448** | **0.1645** | **0.4146** | **0.5096** |
| body単体 | 0.0334 | 0.1383 | 0.3001 | 0.3849 |
| 3フィールド合算（webstyle_narrative_fielded） | 0.0620 | 0.2603 | 0.5028 | 0.7020 |

（consensus、105クエリ）

- 単体性能は headings > title > body の順。title・bodyは単体だとbaselineのrecall@1000
  すら下回る（0.1421, 0.1383 < 0.1495）。
- しかし3フィールドを合算すると、単体のどれよりも大幅に強い。単一フィールドの強さの
  単純な合計以上の相乗効果が出ている（「複数facetに同時にマッチする文書ほど強く浮上する」
  という線形和の設計が効いている）。

### 4.3 フィールドブースト比の最適化

**試した手法1: 線形回帰（`learn_field_boosts.py`）** — 各(トピック, Subquery,
候補文書)について(title_score, headings_score, body_score)を特徴量、qrelsのrelevance
gradeを目的変数にした最小二乗法。

- 推定比率: title=3.44, headings=2.49, body=1.00（既定の3,2,1とほぼ同じ）
- R²=0.126と当てはまりが弱く、実際に105トピックで評価するとむしろ既定よりわずかに悪化した
  （nDCG@10: 0.5028 → 0.4974）
- 原因: クエリ間でBM25生スコアのスケールが揃っておらず、回帰の目的関数（relevanceの二乗誤差）
  が実際に見たい評価指標（nDCG@10, recall@1000）とズレていたため。**この手法は不採用。**

**試した手法2: 実パイプラインでの直接探索（`search_field_boosts.py`）** — 105トピックの
1/3（35トピック）のサブセットで、recall@1000 / nDCG@10 それぞれを直接最大化する
ブースト比を探索。座標降下法（2ラウンド、計13通り）と、title・headings ∈ {1,2,3,4}の
全16通りグリッドサーチの両方で**同じ最適点**に到達した。

| title＼headings | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | R=0.2449 N=0.4492 | R=0.2399 N=0.4691 | R=0.2283 N=0.4590 | R=0.2179 N=0.4588 |
| 2 | **R=0.2582** N=0.4710 | R=0.2484 N=0.4786 | R=0.2380 N=0.4774 | R=0.2276 N=0.4676 |
| 3 | R=0.2534 N=0.4681 | R=0.2473 N=0.4787 | R=0.2376 **N=0.4834** | R=0.2296 N=0.4695 |
| 4 | R=0.2431 N=0.4626 | R=0.2407 N=0.4691 | R=0.2350 N=0.4634 | R=0.2283 N=0.4611 |

（R=recall@1000, N=nDCG@10。35トピックのサブセットでの値。bodyは常に1固定）

見つかった最適比率を **105トピック全体で確認評価**した結果：

| method | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| 既定（title=3, headings=2, body=1） | 0.0620 | 0.2603 | 0.5028 | 0.7020 |
| **recallopt（title=2, headings=1, body=1）** | 0.0621 | **0.2707** | 0.4970 | 0.7048 |
| **ndcgopt（title=3, headings=3, body=1）** | 0.0609 | 0.2512 | **0.5030** | 0.6891 |

（consensus、105クエリ。coverageでも同じ方向: recallopt recall@1000=0.5425、
ndcgopt nDCG@10=0.4076、いずれも既定を上回る）

recall@1000を優先する場合は`recallopt`、上位順位の精度（nDCG@10）を優先する場合は
`ndcgopt`が既定より優れている。headingsの重みを上げるほど良くなるわけではなく、
**recall重視ならheadingsを下げ、nDCG重視ならheadingsを上げる**という逆方向のトレード
オフになっている点は、4.2節の「headings単体が最強」という結果だけからは予測できない
（線形和で組み合わせたときの挙動は単体性能から単純に外挿できない）ことを示している。

## 5. 今後の課題

- 今回の探索範囲は title, headings ∈ {1,2,3,4}, body=1固定。より広い範囲や、bodyも
  含めた3変数の同時最適化は未検証。
- 本フォルダの手法はすべて**文書レベル**（`msmarco-v21-doc`）のインデックスに対するもので、
  TRECの本来の評価対象であるセグメント単位のインデックスでは未検証。
- narrative文脈を生成プロンプトに含める版（`decomposed_query2doc_webstyle_ctx_expansion.py`、
  `experiments/gemini_fewshot/`側に残っている）は効果がほぼゼロだったため、本フォルダには
  含めていない。
