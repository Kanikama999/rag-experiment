# document版 実験まとめ — 最高性能の分析（2026-09-09、2026-09-10更新）

対象: `rag-experiment-doc`（検索対象 = OpenSearch `msmarco-v21-doc`、MS MARCO v2.1 文書レベル
約1,096万件）。`experiments/` 配下の全評価結果 JSON を横断集計し、どの設定が最良かを整理した。

先行文書との関係:

- [`query2doc_bm25_experiment_summary.md`](query2doc_bm25_experiment_summary.md)（2026-08-30）—
  分解 vs 非分解、zero-shot vs few-shot まで
- [`experiments/gemini_fewshot_each_boost/REPORT.md`](experiments/gemini_fewshot_each_boost/REPORT.md)（2026-09-01）—
  web風構造化疑似文書 × フィールド別検索まで
- 本ドキュメント — 上記2つ以降に積み上がった位置ブースト・項重み付け系まで含めた全体像

---

## 0. 結論だけ先に

- **現行の最良設定（champion）は `webstyle_narrative_equalweight_posboost_only`。**
  全4指標・2種類の qrels すべてで baseline の約2〜2.6倍。
- **改善の主役は LLM ではなく位置ブースト。** 105トピックのフル分解（§3.1）では、
  総改善のうち位置ブーストが consensus で73〜93%、coverage で76〜83%を占め、
  構造化疑似文書（LLM由来）の正味の寄与は指標により7〜23%にとどまる。
- **その位置ブーストが効く理由は「先頭が重要」ではなく「窓による希釈低減」だった
  （2026-09-10、§7.12）。** FirstP（先頭窓）と MaxP（最良窓・位置不問）を直接比較したところ
  両qrelsで MaxP がわずかに上回り、「文書の要点は冒頭に書かれやすい」という当初の
  想定（FirstPの直感）は支持されない。§7.10の多段窓実験（冒頭を強く優遇するほど悪化）
  と独立に同じ結論に至っており、論文の位置ブーストの説明を書き換える必要がある。
- **オラクルを現行パイプラインで測り直した結果、逆転は解消したが伸びしろは小さい
  （2026-09-10、§4）。** 旧オラクルが champion に負けていたのは検索側が古い実装
  （平文・フィールド無視・位置ブーストなし）だったためで、現行パイプラインに載せ替えると
  正しく champion を上回る。ただし人間が正解判定した本物の本文を使っても champion比
  +0.03〜0.04（nDCG@10）にとどまり、判定は「機構で頭打ち」。疑似文書の質を上げる路線の
  投資対効果は低い。
- **BM25F は2段階の不公平を修正しても、nDCG@10 では線形和・championに負ける
  （§3.4）。** 1段目（narrative×5の反映）に加え、2026-09-10に `_bm25f_avgdl()` の
  プローブ語バグ（ストップワード"the"でavgdlが1.0にフォールバックし、bodyの寄与が
  ほぼ消えていた）を発見・修正し、bの3軸グリッドとqtf/k3を最初からやり直した。
  最終的な結論（線形和が上）は変わらず、根拠が強化された。ただしrecall@1000は
  両qrelsでBM25Fの方が上。
- **discourseboost は不採用でよい。** 単体では効くが、位置ブーストと併用すると寄与が
  ほぼゼロ（nDCG@10 はむしろ悪化）で、**しかも検索時間が 2.09 倍**になる。
- **IDF項重み付けの強さ α には内点に最適値がある（§7.9）。** 従来のα=1（idf²相当）は
  最適ではなく、α=1.5〜3が両qrels・複数指標で一貫して上回る。α=5では折り返して悪化する
  ことも確認済みで、探索は完結している。
- **多段窓・リング窓は不採用（§7.10）。** 位置減衰を付けるほど悪化し、短文書バイアスを
  除去するリング窓もnDCG@10では改善しない上に検索コストが最大5.68倍になる。

---

## 1. 最高性能の設定

`webstyle_narrative_equalweight_posboost_only`（= `spanterms_all` = `posboost_sb15`。
いずれも同一設定・同一数値で、別スクリプトから重複して測られたもの）。

条件: n=105、`RETRIEVE_K=1000`、`QUERY_REPEAT=5`、`span_end=100`、`span_boost=15`、
疑似文書は gemini-3.7-flash few-shot。

| qrels | | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|---|
| consensus (105topic) | baseline | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| | **champion** | **0.0689** | **0.3095** | **0.5251** | **0.7769** |
| | 倍率 | ×2.11 | ×2.07 | ×1.98 | ×2.08 |
| coverage (22topic) | baseline | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| | **champion** | **0.2373** | **0.5995** | **0.4049** | **0.3768** |
| | 倍率 | ×2.52 | ×2.08 | ×2.58 | ×2.50 |

**全指標・両 qrels で方向が揃っている**点が主張として強い。coverage は TREC 公式判定を
文書単位に集約した厳密寄りの22トピック、consensus は複数チームの合意で作った緩め広めの
105トピックで、性質が違う2つで同じ結論が出ている。

### パイプライン

```
narrative (元の長いクエリ)
  └─ decompose_narrative.py → Subquery（簡潔な質問文、平均4〜5個/トピック）
       └─ decomposed_query2doc_webstyle_expansion.py
            → web風構造化疑似文書 {title, headings, body}
                 │
                 ├─ title    フィールドへ  "narrative×5 + Subquery + 生成title"     を match (boost 1)
                 ├─ headings フィールドへ  "narrative×5 + Subquery + 生成headings"  を match (boost 1)
                 ├─ body     フィールドへ  "narrative×5 + Subquery + 生成body"      を match (boost 1)
                 └─ body先頭100語への span_first 位置ブースト (boost 15)
                      判定語 = analyze_terms(Subquery)
                 │
                 ↓ bool/should でスコア線形和
            Subquery単位で上位1000件
                 └─ RRF (k=60) でトピックの最終ランキングへ
```

実装は [`experiments/gemini_fewshot_each_boost/retriever.py`](experiments/gemini_fewshot_each_boost/retriever.py)
の `bm25_equalweight_posboost_discourseboost(markers=[])`。

---

## 2. 「最高」は指標ごとに違う — 3つの勝者

集計すると単一の勝者ではなく、トレードオフを持つ3クラスタに分かれる（すべて n=105）。

| 手法 | consensus nDCG@10 | consensus R@1000 | coverage nDCG@10 | coverage R@1000 |
|---|---|---|---|---|
| champion (`spanterms_all`) | 0.5251 | **0.3095** | 0.4049 | **0.5995** |
| `spanterms_plus_narrative` | **0.5368** | 0.2988 | **0.4086** | 0.5707 |
| `..._recallopt_confidence_t0.2` | 0.5006 | 0.2787 | **0.4297** | 0.5592 |
| champion + `discourseboost` | 0.5225 | 0.3098 | 0.3995 | 0.6005 |

### 2.1 `spanterms_plus_narrative` が nDCG@10 で champion を上回っている

位置ブーストの判定語（span_terms）に、Subquery の語だけでなく narrative の語も足した版。
**両 qrels とも nDCG@10 で champion を上回る**（consensus +0.0117、coverage +0.0037）一方、
recall@1000 は落ちる（consensus 0.3095→0.2988）。

これは [`evaluate_spanterms_narrative.py`](experiments/gemini_fewshot_each_boost/evaluate_spanterms_narrative.py)
が「seg版では narrative 語を足すと4指標すべて悪化した」という前提の**否定的検証**として
回したもので、doc版では上位精度に限って逆の結果が出た。想定していた機構（span_or は選言なので
語を足すほど発火率が上がり、飽和すると全候補に定数を足すだけになって選別力を失う。実測で
Subqueryのみ 74.7% → +narrative 94.4%）は recall の低下としては当たっているが、
nDCG@10 では逆に働いている。**seg版の知見が doc版にそのまま移らなかった事例**として記録に値する。

2026-09-07 の発表スクリプトはこれを反映しておらず、提案手法を 0.5251 のままにしている。

### 2.2 coverage nDCG 最高値 0.4297 は直接比較できない

`webstyle_narrative_fielded_recallopt_confidence_t0.2` は coverage nDCG@10 で全体最高値だが、
**2つの交絡がある**。

1. **検索深度が違う。** この系列だけ `RETRIEVE_K=3000`（champion は 1000）。
2. **疑似文書の生成モデルが違う。** confidence 重み付けは生成時の token logprob を必要とし、
   gemini-3.7-flash は logprobs が null のため、この系列だけ **gpt-5.6-terra** の疑似文書を
   使っている（`multi_query2doc_decomposed_webstyle_gpt56terra_L200.json`）。

ただし同じ gpt-5.6-terra・同じ深度の自分の対照（`recallopt` coverage 0.4094 / consensus 0.4972）
に対しては +0.0203 / +0.0034 なので、効果そのものは本物。またモデル間の差は
`recallopt` 同士の比較（gemini 0.4970 vs gpt56terra 0.4972 consensus）で見る限り小さい。

---

## 3. 性能の内訳 — 何が効いているのか

### 3.1 位置ブーストが主役（同一35トピック）

`long_unstructured_posboost_result.json` と `equalweight_posboost_two_points_full_metrics.json` より。

| 条件 | consensus nDCG@10 | consensus R@1000 |
|---|---|---|
| narrative のみ（疑似文書なし・位置ブーストなし） | 0.2664 | 0.1438 |
| ＋位置ブーストのみ | 0.4418 (+0.1754) | 0.2874 |
| ＋構造化疑似文書も（= champion） | 0.5007 (+0.0589) | 0.2982 |

**総改善の約75%を、LLM を一切使わない位置ブーストが稼いでいる。** 研究の新規性は
「疑似文書の構造化」に置かれているが、数字上の主役は位置ブースト。コスト対効果の観点で
最も突かれやすい箇所であり、発表の想定問答でも自認している通り。

### 3.2 discourseboost は併用すると寄与しない（n=105）

`decomposed_query2doc_webstyle_eval_summary_equalweight_ablation_rep5.json` より。

| 条件 | consensus nDCG@10 | consensus R@1000 | coverage nDCG@10 |
|---|---|---|---|
| posboost のみ | **0.5251** | 0.3095 | **0.4049** |
| discourseboost のみ | 0.4751 | 0.2569 | 0.3958 |
| 両方 | 0.5225 | 0.3098 | 0.3995 |

discourseboost は単体では baseline を大きく上回るが、posboost と併用すると
**nDCG@10 は -0.0026 で悪化、recall@1000 は +0.0003 で誤差**。

**さらに検索コストが 2.09 倍になる**（2026-09-09 実測、10トピック・同一クエリでの比較）。

| 条件 | 合計 | 中央値 | 最大 |
|---|---|---|---|
| posboost のみ（champion） | 36.41s | 3.63s | 4.55s |
| posboost + discourseboost | 77.81s | 7.29s | 10.35s |
| posboost のみ（再測） | 37.95s | 3.54s | 4.86s |

posboost 単体を前後2回測って挟み込んでおり（36.41s → 37.95s とほぼ一致）、
キャッシュの順序効果ではなく実質的な差である。

原因は節数。`bm25_equalweight_posboost_discourseboost` は discourse marker と span_terms の
**直積**で `span_near` 節を作る:

```python
span_near_clauses = [
    {"span_near": {"clauses": [{"span_term": {"body": mk}}, {"span_term": {"body": t}}],
                    "slop": slop, "in_order": True}}
    for mk in markers for t in span_terms
]
```

マーカーは6個（`summari` / `summar` / `conclus` / `conclud` / `overal` / `short`）、
span_terms は平均10.5語なので、**平均63節**（最小42 / 最大72）が追加される。champion 本体が
`match` 3節 + `span_first` 1節の計4節であることを考えると十数倍で、しかも追加分は
単純な `match` ではなく位置情報を走査する `span_near`（slop=20・順序指定つき）なので
1節あたりのコストも高い。

**結論: 性能は下がり、検索時間は倍。champion に含めない判断は正しく、論文では
「試したが不採用」として効果とコストの両方を示すべき。**

### 3.3 比較対象との差（n=105、consensus）

| 手法 | nDCG@10 | recall@1000 |
|---|---|---|
| baseline（narrative そのまま bm25_body） | 0.2649 | 0.1495 |
| 構造なし疑似文書・narrative繰り返しなし | 0.2706 | 0.1214 |
| Query2doc 再現（構造なし・narrative×5） | 0.3034 | 0.1442 |
| BM25F・qtf無視（旧・不公平） | 0.3476 | 0.2242 |
| **BM25F・narrative×5反映（qtf線形, 公平・最終値）** | **0.4533** | **0.2834** |
| フィールド線形和 recallopt (2,1,1)・位置ブーストなし | 0.4970 | 0.2707〜0.2715 |
| **champion** | **0.5251** | **0.3095** |

**2026-09-09〜10 に2段階で公平化・再測定済み（§3.4）。BM25F の行は主張に使ってよい。**
1段目（narrative×5 の反映）は §3.4-(1)〜(2)、2段目（`retriever.py` の avgdl バグ修正と
b の再探索）は §3.4-(3) を参照。上表は最終値（両方修正後）。

### 3.4 BM25F を公平化して再測定した

BM25F の測定には **2つの不公平**があり、2026-09-09〜10 に順に見つけて直した。
最終的な数値は「最終値（2026-09-10、avgdlバグ修正後）」の節にまとめてある。

**不公平その1: narrative×5 がスコアに反映されていなかった。**
`retriever.py` の `bm25f_prepare()` はクエリ語を重複除去し、`bm25f_score()` は各語を1回しか
回していなかったため、**BM25F だけ `QUERY_REPEAT=5`（narrative×5）が無視されていた**。
`match` クエリを使う他手法は同じ語が5回出れば BooleanQuery 節が5つでき5倍寄与するので、
比較が非対称だった。修正: [`evaluate_bm25f_narrative_rep5.py`](experiments/gemini_fewshot_each_boost/evaluate_bm25f_narrative_rep5.py)
でクエリ語の出現回数（qtf）をスコアに掛けるようにした。

**不公平その2: `_bm25f_avgdl()` のプローブ語バグ（2026-09-10 発見、影響が大きい）。**
`retriever.py` の `_bm25f_avgdl()` はフィールドの平均長をプローブ語 `"the"` の
termvectors から取得していたが、`english_search` アナライザが `"the"` を
ストップワードとして除去するため `term_vectors` が空で返り、`field_statistics` が
取れず **avgdl が 1.0 にフォールバックしていた**（正しい値は title 7.2 / headings 50.9 /
body 1096.9）。BM25F の長さ正規化 `B_f = (1−b_f) + b_f×(dl_f/avgdl_f)` の avgdl が
1.0 だと、body の典型的な dl≈1097 に対し `dl/avgdl≈1097` となり、`b_body` を上げるほど
**body の寄与がほぼゼロまで潰れる**（実質「title だけを見る BM25F」になる）。
修正はプローブ語を非ストップワードに変え、取得できなければ例外を投げるようにした
（`retriever.py`、コミット時にコメントで経緯を記録）。

このバグは **b の最適値の探索そのものを歪めていた**。バグ下の座標降下で見つかった
最適値 `b_title=0.6, b_headings=0.2, b_body=0.2` は、「bodyの寄与を潰すほど良い」という
バグ由来の見かけの最適化であり、無効。avgdl 修正後に3軸同時グリッド
（[`search_bm25f_b_3way.py`](experiments/gemini_fewshot_each_boost/search_bm25f_b_3way.py)、
`b∈{0.0,0.4,0.8,1.0}` の4×4×4=64セル、35トピック）で探索し直した結果、
`b_title=0.0, b_headings=0.8, b_body=1.0`（nDCG@10=0.3777、旧座標降下の0.3620から+0.0157）
が最良と判明した。**b_title=0.0 と b_body=1.0 はいずれも b の定義域 [0,1] の端**なので、
これ以上広げる余地はなく探索は完結している。

qtf の掛け方も、avgdl 修正後に105トピックで
[`evaluate_bm25f_k3_sweep.py`](experiments/gemini_fewshot_each_boost/evaluate_bm25f_k3_sweep.py)
で作り直した新しい `b`（0.0/0.8/1.0）を使ってスイープし直した
（k3=0,1,2,4,8,16,32,64 + linear の9条件、`RETRIEVE_K=3000`）。

### 最終値（2026-09-10、avgdlバグ修正後、n=105）

| k3 | consensus recall@1000 | consensus nDCG@10 | coverage recall@1000 | coverage nDCG@10 |
|---|---|---|---|---|
| 0（dedup） | 0.2242 | 0.3476 | 0.4806 | 0.3063 |
| 1 | 0.2367 | 0.3753 | 0.5057 | 0.3444 |
| 2 | 0.2441 | 0.3945 | 0.5230 | 0.3594 |
| 4 | 0.2537 | 0.4072 | 0.5405 | 0.3656 |
| 8 | 0.2629 | 0.4187 | 0.5566 | 0.3682 |
| 16 | 0.2703 | 0.4324 | 0.5662 | 0.3737 |
| 32 | 0.2754 | 0.4393 | 0.5754 | 0.3784 |
| 64 | 0.2789 | 0.4463 | 0.5814 | 0.3829 |
| **linear（k3=∞）** | **0.2834** | **0.4533** | **0.5890** | **0.3910** |

**両qrels・全4指標の8セルすべてで k3 に対して単調増加し、linear が最良。**
avgdl バグ下では「線形 qtf は効きすぎで飽和が必要」（k3=8 が最良）という結論だったが、
**これはバグの産物だった。** 修正後は飽和させるほど悪化し、線形が最良になる。
§7.11 に書いていた「クエリ側の重みには飽和が必要」という考察（α探索との類推）は撤回する。

BM25F（linear）と線形和 recallopt の最終比較:

| 指標 | BM25F (linear) | 線形和 recallopt | 差 |
|---|---|---|---|
| **consensus** | | | |
| recall@100 | 0.0594 | 0.0621 | −0.0027 |
| recall@1000 | **0.2834** | 0.2707 | **+0.0127** |
| nDCG@10 | 0.4533 | **0.4970** | −0.0437 |
| precision@100 | 0.6732 | **0.7048** | −0.0316 |
| **coverage** | | | |
| recall@100 | 0.2044 | **0.2090** | −0.0046 |
| recall@1000 | **0.5890** | 0.5425 | **+0.0465** |
| nDCG@10 | 0.3910 | **0.4109** | −0.0199 |
| precision@100 | 0.3195 | **0.3309** | −0.0114 |

線形和は OpenSearch 内部の BM25 スコアリングで計算しており `_bm25f_avgdl()` を
使わないため、この比較値自体は avgdl バグの影響を受けていない。

**(1) 公平化しても BM25F は nDCG@10 で負ける — 結論は維持、根拠は強化された。**
avgdlバグ下の最良値（k3=8, nDCG@10=0.4456）は「bodyの寄与を潰した」状態での見かけの
好成績であり、正しく長さ正規化すると最良値は **0.4533** に変わる（下がるのではなく
qtfの掛け方が変わったことで実は上がる。ただし線形和・championとの差は −0.0437 と
バグ下の −0.0514 よりむしろ縮んだ）。**「真の BM25F より、フィールドごとに独立に
BM25 を計算して線形和する方が nDCG@10 で強い」という結論は公平な比較でも生き残った。**

**(2) ただし recall では BM25F の方が強い。** consensus recall@1000 は +0.0127、
coverage は +0.0465 で、**両qrelsでフィールド線形和を上回る**。BM25F の弱点は
上位の精度であって網羅性ではない。後段にリランカーを置く構成ならBM25Fを候補生成に
使う選択肢は残る。

**(3) narrative×5 の効果は consensus と coverage で非対称。** dedup→linear の改善幅:

| | consensus nDCG@10 | coverage nDCG@10 |
|---|---|---|
| dedup→linear | 0.3476→0.4533 (+0.1057) | 0.3063→0.3910 (+0.0847) |

avgdlバグ下では coverage nDCG の改善が+0.0012とほぼゼロだったが、バグを直すと
両qrelsで大きく改善するようになった（旧記述「narrative×5の効果はconsensus限定」は
バグの副作用だったことになる）。

結果: [`bm25f_k3_sweep_result.json`](experiments/gemini_fewshot_each_boost/bm25f_k3_sweep_result.json)
（qtf/k3、最終値）、
[`bm25f_b_3way_grid_result.json`](experiments/gemini_fewshot_each_boost/bm25f_b_3way_grid_result.json)
（b の3軸グリッド）。avgdlバグ下の旧結果一式は
`bm25f_field_b_search_result.json`（座標降下）等に残るが**無効**。

---

## 4. オラクルを現行パイプラインで測り直した（2026-09-10 実施）

### 4.1 旧オラクルが champion に負けていた理由

`experiments/oracle/` は「疑似文書が理想的に書けたらどこまで伸びるか」の上限測定のはず
だったが、**champion がその上限を超えていた**（coverage nDCG@10: オラクル 0.3070 vs
champion 0.4049）。原因はオラクル側が平文・フィールド無視・位置ブーストなしの古い
パイプラインで測られていたことで、**上限が低いのではなく上限の測り方が古かった**。

[`evaluate_oracle_current_pipeline.py`](experiments/gemini_fewshot_each_boost/evaluate_oracle_current_pipeline.py)
でオラクル素材を現行パイプライン（フィールド別match + 位置ブースト + RRF）に載せ替えた。

**クエリ量を揃える必要があった。** オラクルは平文なので素朴に3フィールドすべてへ入れると
champion よりクエリ量が2〜6倍になり、素材の質ではなく量の差を測ってしまう（実測の
TermQuery 節数: champion 633、セグメント1本を3フィールド 1152、3本で 3702）。そこで
`oracle_matched` は**オラクル本文を body だけに入れ、語数もその Subquery の疑似文書 body に
揃えた**。title/headings は `long_pos_fielded` と同じく narrative×5 + Subquery のみ。

### 4.2 結果（22トピック。オラクル素材があるのはこの22件のみ）

| 条件 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| **consensus** | | | | |
| baseline | 0.0412 | 0.1673 | 0.2863 | 0.3909 |
| `oracle_flat`（旧方式の再現） | 0.0473 | 0.2057 | 0.3620 | 0.4427 |
| `long_pos_fielded`（LLM不要） | 0.0779 | 0.3034 | 0.4947 | 0.7377 |
| `champion` | 0.0801 | 0.3367 | 0.5074 | 0.7709 |
| **`oracle_matched`（量を揃えた本物）** | 0.0827 | 0.3488 | **0.5344** | 0.7950 |
| `oracle_body`（本物・全文） | **0.0830** | **0.3541** | 0.5283 | **0.7982** |
| `oracle_rich`（本物×3・3フィールド） | 0.0786 | 0.3464 | 0.4933 | 0.7414 |
| **coverage** | | | | |
| baseline | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| `oracle_flat`（旧方式の再現） | 0.1235 | 0.3922 | 0.2735 | 0.1914 |
| `long_pos_fielded`（LLM不要） | 0.2110 | 0.5311 | 0.3613 | 0.3409 |
| `champion` | 0.2373 | 0.5995 | 0.4049 | 0.3768 |
| **`oracle_matched`（量を揃えた本物）** | 0.2479 | 0.6030 | 0.4442 | 0.3973 |
| `oracle_body`（本物・全文） | **0.2517** | **0.6142** | 0.4417 | **0.4005** |
| `oracle_rich`（本物×3・3フィールド） | 0.2361 | 0.5987 | **0.4517** | 0.3709 |

### 4.3 わかったこと

**(1) 逆転は解消した。** coverage nDCG@10 は旧方式 0.2735 → 現行パイプライン 0.4442 で、
champion（0.4049）を上回る正常な上限に戻った。§4.1 の診断どおり、測り方の問題だった。

**(2) 伸びしろは小さい。** クエリ量を揃えた `oracle_matched` は champion 比で
consensus **+0.0270**、coverage **+0.0394**（nDCG@10）。**人間が正解判定した本物の本文を
使ってもこの程度**で、baseline → champion の +0.2211 に対して1割強にすぎない。

**(3) 素材を増やしても伸びない。** `oracle_body`（全文）は `oracle_matched` とほぼ同じ、
`oracle_rich`（3倍）は **consensus nDCG で champion を下回る**（-0.0141）。量を増やすと
むしろ悪化する。

**(4) LLM 不要条件が上限の93%に達している。** `long_pos_fielded`（LLMもオラクルも使わない）
の consensus nDCG@10 = 0.4947 は、オラクル上限 0.5344 の **93%**。素材を LLM 生成から
人間の正解本文まで最大限に良くしても、残りは7%しか動かない。

**判定は「機構で頭打ち」。** 疑似文書の質を上げる路線への投資対効果は低く、密検索・
リランカーへ移る根拠が実測で得られた。

## 5. 数値を扱う上での注意

- **同一設定の再実行で recall が揺れる。** `webstyle_narrative_fielded_recallopt` の
  consensus recall@1000 はファイル間で 0.2707 / 0.2710 / 0.2715 / 0.2749 とばらつく
  （一部は `RETRIEVE_K` 差、残りは run 差）。一方 nDCG@10 は 0.4970 で完全に安定。
  **recall@1000 の 0.001〜0.004 の差は有意ではない。** champion vs +discourseboost の
  「0.3095 vs 0.3098」はこのノイズレベルの中にある。
- **ハイパーパラメータが探索範囲の端で頭打ちしていない。** 現行の `span_end=100, span_boost=15`
  に対し `span_end=200, span_boost=30` が nDCG@10 0.5007→0.5018、recall@1000 0.2982→0.3052
  と単調に伸び続けている（35トピック）。現行 champion は最適点ではなく、グリッドの内側で
  止めた点にすぎない。
- **n=35 のサブセット結果を n=105 の表に混ぜないこと。** `champion_localdensity` /
  `posdensity_*` 系は35トピックのサブセットで、集計時に coverage recall@100 の上位へ
  紛れ込む。
- **Query2doc 再現行だけ 103 トピック**で評価されている（他は105）。同じ103トピックでの
  baseline nDCG@10 は 0.2634（105では 0.2649）なので差は小さいが、厳密には揃っていない。

---

## 6. 命名の修正（2026-09-09）

`notitle` → `equalweight` に改名した。

**理由**: `bm25_notitle_*` / `webstyle_narrative_notitle_*` という名前は「title フィールドを
使わない」という意味に読めるが、実際は **title/headings/body を `title_boost = headings_boost
= body_boost = 1` の均等重みで線形和している**だけで、title は重み1で使っている。
フィールド重み付け（recallopt = title 2:1:1 など）を**しない**という意図の名前だったが、
誤解を招くため実態に合わせた。

**挙動の変更は一切ない**（識別子・ファイル名・結果JSONのキー名のみの変更）。

| 旧 | 新 |
|---|---|
| `bm25_notitle_posboost_discourseboost` | `bm25_equalweight_posboost_discourseboost` |
| `bm25_notitle_localdensity` ほか4関数 | `bm25_equalweight_*` |
| `webstyle_narrative_notitle_posboost_only` | `webstyle_narrative_equalweight_posboost_only` |
| `evaluate_decomposed_webstyle_notitle_ablation.py` | `evaluate_decomposed_webstyle_equalweight_ablation.py` |
| `decomposed_query2doc_webstyle_eval_summary_notitle_*.json` | `..._equalweight_*.json` |

`retriever.py` の該当関数群の直前に改名の経緯を注記済み。

`rag-experiment-seg`（セグメント版）側にも doc版の関数名を参照している箇所が2ファイル
（`REPORT.md`、`evaluate_webstyle_posboost_discourseboost.py`）あったので、そちらも新名に
追従させた。seg版に `bm25_notitle_*` の実装自体は無く、doc版への言及のみ。

---

## 7. 追加検証: 項重み付け × 位置ブースト（2026-09-09 実施）

### 7.1 なぜやったか

これまで「項(term)重み付け」と「位置ブースト」は別々の系列で評価され、**一度も組み合わせて
測られていなかった**。項重み付け側が coverage nDCG@10 で全体最高値（0.4297）を出しているのに
併用結果が無いため、「足し合わさるのか、それとも同じ情報を二重に使っているだけなのか」が
未解決だった。

さらに §2.2 の通り、両系列は**疑似文書の生成モデルまで違っていた**（位置ブースト側 =
gemini-3.7-flash、項重み付け側 = gpt-5.6-terra）ため、既存数値の直接比較にはモデル交絡もあった。

### 7.2 設計

スクリプト: [`evaluate_termweight_posboost_2x2.py`](experiments/gemini_fewshot_each_boost/evaluate_termweight_posboost_2x2.py)

4条件すべてを**同一の検索関数** `bm25_fielded_weighted_posboost` で回し、
**重みの値と `span_boost` だけ**を変える2×2にした。

| 条件 | 項重み付け | 位置ブースト |
|---|---|---|
| `tw_uniform` | なし（全語 weight=1.0） | なし |
| `tw_uniform_posboost` | なし | あり |
| `tw_weighted` | あり | なし |
| `tw_weighted_posboost` | あり | あり |

対照に `bm25_fielded`（フィールド全体を1つの match 節にまとめる）を使わなかったのは、
term単位に分解すると TF の効き方が変わり、重み付け以外の差が混入するため。
参考行として既存 champion そのもの（`champion_ref`）も同時に回した。

条件は champion に揃えた（`RETRIEVE_K=1000`、`QUERY_REPEAT=5`、`span_end=100`、
`span_boost=15`、n=105）。既存の項重み付け実験は `RETRIEVE_K=3000` だったので直接比較できず、
ここで揃え直している。

**項重み付けの方式**: gemini-3.7-flash は token logprob を返さないため（§8-5 参照）、
confidence 方式は使えない。代わりに logprob 不要の **IDF ベース**（`term_weights.idf_term_weights`、
`weight = idf / mean(idf)` で平均1に正規化）を使った。

### 7.3 結果（gemini-3.7-flash + IDF項重み付け、n=105）

| 条件 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| **consensus** | | | | |
| baseline | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| champion_ref | 0.0689 | 0.3095 | **0.5251** | 0.7769 |
| tw_uniform | 0.0587 | 0.2556 | 0.4765 | 0.6644 |
| tw_uniform_posboost | 0.0690 | 0.3095 | 0.5257 | 0.7770 |
| tw_weighted | 0.0584 | 0.2605 | 0.4658 | 0.6610 |
| **tw_weighted_posboost** | 0.0690 | **0.3176** | 0.5182 | **0.7788** |
| **coverage** | | | | |
| baseline | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| champion_ref | 0.2373 | 0.5995 | 0.4049 | 0.3768 |
| tw_uniform | 0.1922 | 0.5342 | 0.4092 | 0.2973 |
| tw_uniform_posboost | 0.2378 | 0.5984 | 0.4049 | 0.3777 |
| tw_weighted | 0.1973 | 0.5537 | 0.4053 | 0.3050 |
| **tw_weighted_posboost** | 0.2375 | **0.6208** | **0.4154** | **0.3782** |

### 7.4 わかったこと

**(1) champion が完全に再現した。** `champion_ref` の consensus 値
0.0689 / 0.3095 / 0.5251 / 0.7769 は既知の champion 値と**小数点以下4桁まで一致**。
環境・データ・パイプラインの妥当性が確認できた。

**(2) 語ごとの節分解それ自体は無害。** `tw_uniform_posboost`（0.0690 / 0.3095 / 0.5257 / 0.7770）は
`champion_ref` とほぼ同一。フィールド全体を1節にまとめる方式と、語ごとに節を分ける方式で
結果が変わらないことを確認できたので、2×2の差分は**純粋に重み付けの効果**と読んでよい。
（narrative×5 + Subquery の plain_text 部分が両条件で共通かつ支配的なため、疑似文書側の
節の切り方の違いが希釈されていると考えられる。）

**(3) 両者は「足し算」で、重複していない。** これが本検証の主目的への答え。

| 指標 | 位置ブーストの改善幅（重み付けなし） | 同（重み付けあり） | 差 |
|---|---|---|---|
| recall@100 | +0.0103 | +0.0105 | +0.0003 |
| recall@1000 | +0.0539 | +0.0571 | +0.0032 |
| nDCG@10 | +0.0491 | +0.0524 | +0.0032 |
| precision@100 | +0.1127 | +0.1178 | +0.0051 |

（consensus）位置ブーストの効果は、項重み付けの有無でほとんど変わらない（差はすべて
+0.0003〜+0.0051で、§5 のノイズ水準と同程度）。**相乗効果も打ち消し合いもなく、単純に
加算される**。両者が別々の情報を捉えていることを意味する。

**(4) IDF項重み付けは recall を上げ、nDCG を下げる。**
（**2026-09-09 追記: この節の nDCG 低下は実装バグの副作用だった。§7.9 の訂正を参照。**
正しく正規化すると nDCG の低下は消え、適切な α では全指標で champion を上回る。）
効果の符号が指標で逆になる。

| | recall@1000 への効果 | nDCG@10 への効果 |
|---|---|---|
| consensus・位置ブーストなし | +0.0049 | -0.0107 |
| consensus・位置ブーストあり | +0.0081 | -0.0075 |
| coverage・位置ブーストなし | +0.0195 | -0.0039 |
| coverage・位置ブーストあり | +0.0224 | +0.0105 |

**recall@1000 は4条件すべてで一貫して改善**する一方、nDCG@10 は consensus では一貫して悪化する
（coverage の +0.0105 は22トピックなので弱い証拠）。レア語を重くすると幅広く拾えるが、
上位10件の並びは犠牲になる、という素直な解釈ができる。

**(5) recall@1000 の新記録。** `tw_weighted_posboost` の recall@1000 は
**consensus 0.3176（champion 0.3095 比 +0.0081）、coverage 0.6208（同 0.5995 比 +0.0213）**で、
両 qrels とも既存最高値を更新した。§5 で述べた run 間ノイズ（0.001〜0.004）を超える差であり、
coverage の +0.0213 は明確に有意。precision@100 も両 qrels で最高値（0.7788 / 0.3782）。

ただし nDCG@10 は consensus で 0.5182 < champion 0.5251 なので、**champion を全面的に
置き換えるものではない**。既存の recallopt / ndcgopt と同じ「指標優先度で選ぶ」関係になる。

**(6) 検索コストはほぼ増えない。** 条件ごとの実測（105トピック）:

| 条件 | 所要時間 | クエリの節数 |
|---|---|---|
| champion_ref | 1336秒 | 4 |
| tw_uniform | 749秒 | 平均 208 |
| tw_uniform_posboost | 1599秒 | 平均 208 |
| tw_weighted | 661秒 | 平均 166 |
| tw_weighted_posboost | 1458秒 | 平均 166 |

節数が4→166と40倍になっても、champion 比で**+9%**（1336→1458秒）にしか増えない。
むしろ `tw_weighted`（661秒）は `tw_uniform`（749秒）より**速い** — IDF=0 のストップワードを
落として節数を20%削減しているため。**項重み付けは実質タダで recall を買える。**
支配的なコストは節数ではなく位置ブースト（`span_first`）の方で、これがおよそ2倍にしている。

### 7.5 副実験: confidence 項重み付け（gpt-5.6-terra、n=105）

同じ2×2を、logprob が取れる唯一の疑似文書セットである gpt-5.6-terra 上で、
**confidence 方式**（生成時の token logprob を softmax、temperature=0.2）で回した。

| 条件 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| **consensus** | | | | |
| baseline | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| champion_ref | 0.0689 | 0.3141 | 0.5174 | 0.7744 |
| tw_uniform | 0.0591 | 0.2601 | 0.4792 | 0.6662 |
| tw_uniform_posboost | 0.0688 | 0.3140 | 0.5185 | 0.7742 |
| tw_weighted | 0.0603 | 0.2664 | 0.4841 | 0.6792 |
| **tw_weighted_posboost** | **0.0691** | **0.3160** | **0.5197** | **0.7774** |
| **coverage** | | | | |
| baseline | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| champion_ref | 0.2375 | 0.6120 | 0.4039 | 0.3768 |
| tw_uniform | 0.1964 | 0.5459 | 0.3994 | 0.3059 |
| tw_uniform_posboost | 0.2374 | 0.6112 | 0.4031 | 0.3764 |
| tw_weighted | 0.2017 | 0.5570 | **0.4105** | 0.3155 |
| **tw_weighted_posboost** | **0.2379** | **0.6123** | 0.4044 | **0.3791** |

**(1) confidence は4指標すべてを改善する。** IDF が「recall を上げ nDCG を下げる」トレードオフ
だったのに対し、confidence は**8セル（2 qrels × 2 posboost条件 × 4指標のうち符号を見た全て）で
一貫してプラス**。重みの作り方として confidence の方が素直に効く。

| | recall@1000 | nDCG@10 |
|---|---|---|
| consensus・位置ブーストなし | +0.0063 | +0.0049 |
| consensus・位置ブーストあり | +0.0020 | +0.0012 |
| coverage・位置ブーストなし | +0.0111 | +0.0111 |
| coverage・位置ブーストあり | +0.0011 | +0.0013 |

**(2) ただし位置ブーストと重複する。** 上表の通り、confidence の効果は位置ブーストを
足すと **1/3〜1/10 に縮む**（consensus nDCG +0.0049 → +0.0012、coverage recall@1000
+0.0111 → +0.0011）。交互作用の差も **8指標すべてで負**（-0.0009 〜 -0.0100）。

これは IDF と**正反対の性質**である。

| 重み付け方式 | 位置ブーストとの関係 | 交互作用の差（consensus） |
|---|---|---|
| IDF（§7.4） | **加算的**（重複しない） | +0.0003 〜 +0.0051（すべて正） |
| confidence（本節） | **部分的に重複** | -0.0009 〜 -0.0098（すべて負） |

解釈: confidence は「LLM が自信を持って書いた語」を重くするが、そうした語は文書の中心的な
話題語であり、**文書の冒頭に出現しやすい**。位置ブーストが捉えているのもまさに
「冒頭に出る語」なので、両者は同じ情報を別経路で見ていることになる。一方 IDF が捉える
「コーパスでの珍しさ」は出現位置とは独立な軸なので、位置ブーストと足し合わさる。

**(3) 実務的な結論。** 位置ブーストを使う前提なら、confidence の上乗せはほぼ無意味
（consensus nDCG +0.0012 はノイズ水準）。**位置ブーストと併用して意味があるのは IDF の方**で、
recall@1000 を +0.0081 押し上げて全体新記録を出したのは IDF 側だった（§7.4-5）。
logprob が取れないモデルでも使えるという実用上の利点と合わせて、**IDF 方式を採る根拠になる**。

**(4) モデル間の差。** `champion_ref` は疑似文書の生成モデルだけが違う同一設定なので、
モデル差の直接比較になる。

| | consensus nDCG@10 | consensus recall@1000 | coverage recall@1000 |
|---|---|---|---|
| gemini-3.7-flash | **0.5251** | 0.3095 | 0.5995 |
| gpt-5.6-terra | 0.5174 | **0.3141** | **0.6120** |

gemini は nDCG@10 で、gpt-5.6-terra は recall で上回る。差は小さいが、§2.2 で
「モデル間の差は小さい」と書いた見立てはおおむね妥当だった。

### 7.6 現時点の最良（全実験を通じて、n=105）

| 基準 | 設定 | 値 |
|---|---|---|
| nDCG@10 | gemini champion（均等重み + 位置ブースト） | consensus **0.5251** |
| nDCG@10（別解） | `spanterms_plus_narrative`（§2.1） | consensus **0.5368** ※recall は落ちる |
| recall@1000 | **gemini + IDF項重み付け + 位置ブースト** | consensus **0.3176** / coverage **0.6208** |
| recall@1000（+Subquery×5） | 同上 + Subquery×5（§7.7.1） | consensus 0.3178 / coverage **0.6277** |
| precision@100 | gemini + IDF項重み付け + 位置ブースト | consensus **0.7788** / coverage 0.3782 |

---

## 7.7 追加検証: Subquery を繰り返すと効くのか（2026-09-09 実施）

### 経緯

narrative は Query2doc 論文の知見に従って `QUERY_REPEAT=5` で繰り返してきた（繰り返さないと
元クエリの語が疑似文書の語に TF で埋もれる。1→5 で全指標が明確に改善）。一方 **Subquery は
一貫して1回しか入れていなかった**。繰り返しの効果が narrative 固有なのか、Subquery にも
効くのかは未検証だった。

スクリプト: [`evaluate_subquery_repeat.py`](experiments/gemini_fewshot_each_boost/evaluate_subquery_repeat.py)

「narrative の有無」×「Subquery の繰り返し回数」の 2×2。Subquery×5 だけを champion と
比べると、差が「narrative を外した効果」なのか「Subquery を5回にした効果」なのか判別できない
ため `n0_s1` を対照に入れてある。検索側は champion 構成に固定（均等重み + 位置ブースト、
`span_end=100`/`span_boost=15`、`RETRIEVE_K=1000`、gemini 疑似文書、n=105）。

なお位置ブーストの判定語 `span_terms` は `analyze_terms(Subquery)` で作られ、この関数は
重複を除去するので、**Subquery を何回繰り返しても `span_terms` は変わらない**。
本実験で動くのは match 節側の項頻度(TF)だけである。

### 結果（gemini + champion 構成、n=105）

| 条件 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| **consensus** | | | | |
| baseline | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| `n5_s1`（= champion） | 0.0689 | 0.3095 | **0.5251** | 0.7769 |
| `n5_s5` | **0.0693** | **0.3111** | 0.5177 | **0.7792** |
| `n0_s1` | 0.0598 | 0.2737 | 0.3906 | 0.6664 |
| `n0_s5` | 0.0641 | 0.2847 | 0.4324 | 0.7128 |
| **coverage** | | | | |
| baseline | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| `n5_s1`（= champion） | 0.2373 | 0.5995 | 0.4049 | 0.3768 |
| `n5_s5` | **0.2429** | **0.6068** | **0.4149** | **0.3882** |
| `n0_s1` | 0.2014 | 0.5695 | 0.3359 | 0.3191 |
| `n0_s5` | 0.2254 | 0.5843 | 0.3930 | 0.3573 |

`n5_s1` は champion 値を4桁まで完全再現（0.0689 / 0.3095 / 0.5251 / 0.7769）。

### わかったこと

**(1) 繰り返しの効果は narrative 固有ではないが、narrative があると飽和する。**
Subquery を1回→5回にしたときの改善幅:

| 指標 | narrative あり | narrative なし |
|---|---|---|
| **consensus** | | |
| recall@1000 | +0.0016 | **+0.0110** |
| nDCG@10 | **-0.0074** | **+0.0418** |
| precision@100 | +0.0024 | **+0.0464** |
| **coverage** | | |
| recall@1000 | +0.0074 | **+0.0148** |
| nDCG@10 | +0.0101 | **+0.0572** |
| precision@100 | +0.0114 | **+0.0382** |

**narrative が無いときは Subquery×5 が劇的に効く**（consensus nDCG +0.0418、coverage +0.0572）。
一方 **narrative があるとほぼ効かない**（consensus nDCG は -0.0074 とむしろ微減）。

機構的な解釈: 繰り返しの目的は「元の情報要求側の語の TF を、疑似文書の語に対して確保する」
ことである。narrative×5 がすでにその役目を果たしているので、Subquery を重ねても
**追加の効果が飽和している**。逆に narrative を外すと Subquery だけが情報要求の担い手に
なるため、繰り返しが本来の効き方をする。繰り返しは「Subquery 固有の工夫」ではなく
「元クエリ側の TF 確保」という単一の機構だった、と読める。

**(2) narrative は Subquery の繰り返しでは代替できない。** narrative を外したときの
consensus nDCG@10 の低下は Subquery×1 で -0.1344、×5 でも -0.0852。Subquery×5 は
**低下幅を約4割埋めるが、埋め切らない**。narrative は繰り返し回数では補えない情報
（複数観点の文脈、条件、背景）を持っている。

**(3) champion を置き換えるほどではない。** `n5_s5` は coverage では4指標すべてで
champion を上回る（nDCG@10 0.4049→0.4149 など）が、consensus では
recall@100 / recall@1000 / precision@100 が微増（+0.0003 〜 +0.0024、ノイズ水準）の一方
**nDCG@10 は -0.0074 と低下**する。105トピックの consensus を重く見るなら、
**Subquery×5 は明確な勝ちではない**。

**(4) 所要時間は逆転している（ただし実行順の交絡あり）。**

| 条件 | 所要時間 |
|---|---|
| `n5_s1` | 1520秒 |
| `n5_s5` | 1393秒 |
| `n0_s1` | 1261秒 |
| `n0_s5` | 1128秒 |

クエリが長い `n5_s5` の方が `n5_s1` より速い。実行順が n5_s1 → n5_s5 → n0_s1 → n0_s5 で
単調に短くなっていることから、**OpenSearch のページキャッシュが温まった順序効果**と考えられ、
条件間の純粋な速度比較には使えない。narrative を外すと速い（1261s / 1128s）のは
クエリ文字列が短くなるためで、こちらは実質的な差だろう。


### 7.7.1 同じ2×2を idfweighted 構成でも回した（検証: 希釈仮説）

§7.7 の結果を受けて「narrative があると Subquery×5 が効かないのは、Subquery が疑似文書の
本文（約200語）と同じ match 節に連結されて薄められるからではないか」という希釈仮説を立て、
`idfweighted` 構成（疑似文書の語を語ごとの独立した重み付き節に分離するので、連結される側に
残るのは narrative と Subquery だけ）で同じ2×2を回した。仮説が正しければ、こちらでは
Subquery×5 がより強く効くはずである。

| 条件 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| **consensus** | | | | |
| `n5_s1`（IDF項重み付けのみ） | 0.0690 | 0.3176 | **0.5182** | 0.7788 |
| `n5_s5`（IDF + Subquery×5） | 0.0691 | **0.3178** | 0.5166 | **0.7791** |
| `n0_s1` | 0.0591 | 0.2763 | 0.3815 | 0.6572 |
| `n0_s5` | 0.0636 | 0.2936 | 0.4233 | 0.7073 |
| **coverage** | | | | |
| `n5_s1`（IDF項重み付けのみ） | 0.2375 | 0.6208 | **0.4154** | 0.3782 |
| `n5_s5`（IDF + Subquery×5） | **0.2433** | **0.6277** | 0.4060 | **0.3873** |
| `n0_s1` | 0.1932 | 0.5630 | 0.3266 | 0.3055 |
| `n0_s5` | 0.2193 | 0.5968 | 0.3840 | 0.3491 |

**希釈仮説は否定された。** narrative がある条件での Subquery×5 の効果を2構成で比べると:

| 指標（consensus） | champion 構成 | idfweighted 構成 |
|---|---|---|
| recall@1000 | +0.0016 | **+0.0002** |
| nDCG@10 | -0.0074 | **-0.0015** |
| precision@100 | +0.0024 | +0.0004 |

疑似文書の語を分離しても Subquery×5 は強くならず、**むしろゼロに近づいた**。

決定的なのは narrative なし条件の一致である。Subquery×5 による consensus nDCG@10 の改善幅は
**champion 構成 +0.0418 / idfweighted 構成 +0.0419** と、検索の組み立てが全く違うのに
ほぼ同一だった。つまりこの現象は検索側の構造とは独立で、**「narrative×5 が既にクエリ側の
TF を飽和させている」という説明（§7.7-1）が正しい**と確認できた。

**新記録（ただし限定的）。** `n5_s5` + IDF は coverage recall@1000 = **0.6277**（従来最高
0.6208）、coverage recall@100 = **0.2433**（同 0.2429）を更新した。ただし consensus 側は
recall@1000 +0.0002 とノイズ水準の横ばい。**実質的な改善は IDF 項重み付けが担っており、
Subquery×5 の上乗せが見えるのは22トピックの coverage だけ**である。論文で主張するなら
「Subquery の繰り返しは narrative がある限り不要」という否定的知見の方が確度が高い。

## 7.8 全手法の一覧比較（両qrels × 全4指標、すべて n=105）

これまでの全実験を1つの表に統合したもの。**太字が各列の最高値**。すべて105トピック、
`TOPK=1000`、`QUERY_REPEAT=5`。注記のない行は gemini-3.7-flash 疑似文書・`RETRIEVE_K=1000`。
★は現時点の推奨2設定（nDCG重視 / recall重視）。

### consensus
| 手法 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline（narrativeそのまま bm25_body） | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| 構造なし疑似文書・繰り返しなし | 0.0287 | 0.1214 | 0.2706 | 0.3322 |
| Query2doc再現（構造なし・narrative×5） | 0.0342 | 0.1442 | 0.3034 | 0.3925 |
| BM25F (tuned)・qtf無視<br><sub>narrative×5が効かない旧実装（§3.4）</sub> | 0.0569 | 0.2889 | 0.3833 | 0.6387 |
| BM25F (tuned) + narrative×5（線形 qtf） | 0.0594 | 0.2834 | 0.4243 | 0.6594 |
| BM25F (tuned) + narrative×5（飽和 qtf, k3=8） | 0.0622 | 0.2958 | 0.4456 | 0.6946 |
| フィールド線形和 recallopt (2,1,1) | 0.0621 | 0.2715 | 0.4970 | 0.7046 |
| フィールド線形和 + 位置ブースト | 0.0672 | 0.2949 | 0.5223 | 0.7584 |
| discourseboost 単体（均等重み） | 0.0590 | 0.2569 | 0.4751 | 0.6674 |
| ★champion（均等重み + 位置ブースト） | 0.0689 | 0.3095 | 0.5251 | 0.7769 |
| champion + discourseboost | 0.0690 | 0.3098 | 0.5225 | 0.7773 |
| champion + span_termsにnarrative追加 | 0.0680 | 0.2988 | **0.5368** | 0.7711 |
| champion + Subquery×5 | **0.0693** | 0.3111 | 0.5177 | **0.7792** |
| ★champion + IDF項重み付け | 0.0690 | 0.3176 | 0.5182 | 0.7788 |
| ★champion + IDF項重み付け + Subquery×5 | 0.0691 | **0.3178** | 0.5166 | 0.7791 |
| champion + confidence項重み付け<br><sub>疑似文書は gpt-5.6-terra</sub> | 0.0691 | 0.3160 | 0.5197 | 0.7774 |
| フィールド線形和 + confidence (t=0.2)<br><sub>RETRIEVE_K=3000・gpt-5.6-terra</sub> | 0.0634 | 0.2787 | 0.5006 | 0.7144 |

### coverage
| 手法 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline（narrativeそのまま bm25_body） | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| 構造なし疑似文書・繰り返しなし | 0.0898 | 0.2583 | 0.2071 | 0.1382 |
| Query2doc再現（構造なし・narrative×5） | 0.1026 | 0.3052 | 0.2508 | 0.1609 |
| BM25F (tuned)・qtf無視<br><sub>narrative×5が効かない旧実装（§3.4）</sub> | 0.1875 | 0.5957 | 0.3214 | 0.3055 |
| BM25F (tuned) + narrative×5（線形 qtf） | 0.1948 | 0.5538 | 0.2796 | 0.3186 |
| BM25F (tuned) + narrative×5（飽和 qtf, k3=8） | 0.2076 | 0.5899 | 0.3226 | 0.3382 |
| フィールド線形和 recallopt (2,1,1) | 0.2090 | 0.5463 | 0.4109 | 0.3309 |
| フィールド線形和 + 位置ブースト | 0.2263 | 0.5776 | 0.4080 | 0.3627 |
| discourseboost 単体（均等重み） | 0.1943 | 0.5395 | 0.3958 | 0.3014 |
| ★champion（均等重み + 位置ブースト） | 0.2373 | 0.5995 | 0.4049 | 0.3768 |
| champion + discourseboost | 0.2375 | 0.6005 | 0.3995 | 0.3773 |
| champion + span_termsにnarrative追加 | 0.2262 | 0.5707 | 0.4086 | 0.3650 |
| champion + Subquery×5 | 0.2429 | 0.6068 | 0.4149 | **0.3882** |
| ★champion + IDF項重み付け | 0.2375 | 0.6208 | 0.4154 | 0.3782 |
| ★champion + IDF項重み付け + Subquery×5 | **0.2433** | **0.6277** | 0.4060 | 0.3873 |
| champion + confidence項重み付け<br><sub>疑似文書は gpt-5.6-terra</sub> | 0.2379 | 0.6123 | 0.4044 | 0.3791 |
| フィールド線形和 + confidence (t=0.2)<br><sub>RETRIEVE_K=3000・gpt-5.6-terra</sub> | 0.2185 | 0.5592 | **0.4297** | 0.3468 |

### 読み方

**(1) 最高値は指標ごとに別の手法に散っている。** 単一の勝者はいない。

| 指標 | consensus の最高 | coverage の最高 |
|---|---|---|
| recall@100 | champion + Subquery×5 (0.0693) | **IDF項重み + Subquery×5 (0.2433)** |
| recall@1000 | **IDF項重み + Subquery×5 (0.3178)** | **IDF項重み + Subquery×5 (0.6277)** |
| nDCG@10 | span_termsにnarrative追加 (0.5368) | 線形和+confidence (0.4297) |
| precision@100 | champion + Subquery×5 (0.7792) | champion + Subquery×5 (0.3882) |

ただし consensus の recall@1000 は 0.3176（IDF単体）→ 0.3178（+Subquery×5）で差は +0.0002 と
ノイズ水準。**実質的な改善を担っているのは IDF 項重み付けの方**であり、Subquery×5 の
上乗せが見えるのは22トピックの coverage だけである（§7.7.1）。

**(2) 両qrelsで一致するのは recall@1000 だけ。** IDF項重み付けが両方で最高値を取っており、
これが**最も信頼できる改善**である。nDCG@10 は consensus と coverage で勝者が食い違う
（0.5368 の手法は coverage では 0.4086、0.4297 の手法は consensus では 0.5006）。
nDCG@10 での順位付けは qrels の選び方に依存するので、単独の根拠にしないこと。

**(3) champion 系の内部差は小さい。** champion 以降の6手法は consensus nDCG@10 が
0.5177〜0.5368、recall@1000 が 0.2988〜0.3176 の狭い帯に収まる。§5 の run 間ノイズ
（recall で 0.001〜0.004）を考えると、この帯の中の順位はあまり意味を持たない。
**大きな差がついているのは champion より手前**（baseline 0.2649 → 線形和 0.4970 →
champion 0.5251）であり、伸びしろは微調整ではなく構造側にある。

**(4) coverage の最高 nDCG@10 (0.4297) は条件が揃っていない。** `RETRIEVE_K=3000`
かつ疑似文書が gpt-5.6-terra で、他行（1000・gemini）と直接比較できない。
同一条件で測り直した `champion + confidence項重み付け` は coverage nDCG@10 = 0.4044 で、
champion (0.4049) と差が無かった（§7.5）。**この行を主張に使わないこと。**


## 7.9 IDF項重み付けの強さ α の探索（2026-09-09〜10 実施）

### 経緯と問題意識

IDF 重み（`weight = idf/平均idf`）を boost に渡すと、BM25 がスコア計算時に既に idf を
掛けているため実効的な重みは **idf の2乗**になる。BM25 の idf は Robertson-Spärck Jones
重みの導出を持つ量だが、**それを2乗する導出は存在しない**。従来の設定は理論的に
正当化された値ではなく、試した2点のうちの片方にすぎなかった。

一方、クエリ側に項ごとの重みを置くこと自体は BM25 の定義に含まれる（完全形は
`score = Σ_t [クエリ側の重み] × idf(t) × [文書側のtf正規化]` で、クエリ側の重みには
通常クエリ内項頻度の飽和項が入る。narrative×5 が効くのはこのスロットの操作）。
また拡張語に重みを付けるのは Rocchio / RM3 など擬似適合性フィードバックの標準的な実務。
**争点は「IDF で重み付けすること」ではなく「その強さ」**である。

そこで指数 α を導入した:

```
weight ∝ (idf/平均idf)**α   を平均1に再正規化
→ 実効的な重み ∝ idf**(1+α)
   α=0   → idf**1（素のBM25。全語が重み1）
   α=1   → idf**2（従来）
```

平均1への再正規化が重要で、α は「配る boost の総量」ではなく**語間の強弱だけ**を変える。
実データ（125語の疑似文書 body）での boost 値:

| 語 | α=0 | α=0.5 | α=1 | α=1.5 |
|---|---|---|---|---|
| `undercompensation` | 1.00 | 2.01 | 3.77 | 6.65 |
| `workers` | 1.00 | 0.98 | 0.89 | 0.76 |
| `from` | 1.00 | 0.25 | 0.06 | 0.01 |

スクリプト: [`evaluate_idf_alpha_sweep.py`](experiments/gemini_fewshot_each_boost/evaluate_idf_alpha_sweep.py)
検索構成は `tw_weighted_posboost` と完全に同一（均等重み、`span_end=100`、`span_boost=15`、
`RETRIEVE_K=1000`、`QUERY_REPEAT=5`、gemini疑似文書、n=105）。変えるのは α だけ。

### 訂正: 旧 `idf_term_weights` に正規化のバグがあった

α 引数を追加する際に判明した。旧実装は `mean_idf` を**ストップワードを含む全語**で
計算しておきながら、重みは **idf>0 の語**にだけ配っていた。そのため残った語の重みの
平均が1ではなく **1.25 程度に膨らみ**、IDF重み付けの条件だけ総ブースト量が2割超過していた。

| | 旧（過剰ブースト） | 新 α=1（正規化済み） |
|---|---|---|
| consensus nDCG@10 | 0.5182 | **0.5251** |
| consensus recall@1000 | 0.3176 | 0.3160 |

**§7.4 で報告した「IDF は recall を上げ nDCG を下げる」というトレードオフの正体は、
「IDF強調」ではなく「ブースト量の過剰」だった。** 正規化を直すと nDCG の低下は消える。

### 結果（105トピック）

| α | consensus R@1000 | consensus nDCG@10 | consensus P@100 | coverage R@1000 | coverage nDCG@10 |
|---|---|---|---|---|---|
| 0 | 0.3095 | 0.5257 | 0.7770 | 0.5984 | 0.4049 |
| 0.25 | 0.3114 | 0.5245 | 0.7796 | 0.6026 | 0.4037 |
| 0.5 | 0.3131 | 0.5243 | 0.7797 | 0.6078 | 0.4061 |
| 0.75 | 0.3147 | 0.5256 | 0.7825 | 0.6095 | 0.4123 |
| 1（従来） | 0.3160 | 0.5251 | 0.7844 | 0.6119 | 0.4144 |
| **1.5** | 0.3185 | **0.5271** | **0.7871** | 0.6182 | **0.4266** |
| 2 | 0.3201 | 0.5231 | 0.7862 | 0.6229 | 0.4246 |
| **3** | **0.3208** | 0.5208 | 0.7824 | **0.6269** | 0.4178 |
| 5 | 0.3097 ↓ | 0.5131 ↓ | 0.7575 ↓ | 0.6042 ↓ | 0.4219 |

**α=5 で明確に折り返した**ので、他のパラメータ（`span_end` 等）と違い**内点に最適値がある**。

### わかったこと

**(1) 当初の予想は逆だった。** 「α=1（idf²）は強すぎるので中間に最適点がある」と予想したが、
正しく正規化すると α は 1.5〜3 まで効き続けた。「α=1 に理論的根拠が無い」という問題意識は
妥当で、探索の結果**より強い方が良かった**という決着。

**(2) 指標ごとに最適 α が分かれる。** 両 qrels で同じ形。

| 指標 | 最適 α |
|---|---|
| recall@1000 | **3** |
| nDCG@10 | **1.5** |
| precision@100 | 1.5〜2 |

「レア語を強調すると幅広く拾えるが上位10件の並びは荒れる」という §7.4 の解釈自体は、
正しい正規化のもとで**より緩やかな形で**再現した（旧実装のバグで見えていた急な
トレードオフとは別物）。

**(3) nDCG@10 でも champion を上回った。** α=1.5 は consensus nDCG@10 = 0.5271 で
champion（0.5251）を超え、4指標すべてで champion 以上。これまで「IDF項重み付けは
recall 専用」と整理していたが、**適切な α なら全指標で champion を超える**。

### 推奨設定の更新

| 優先指標 | 設定 | consensus | coverage |
|---|---|---|---|
| バランス／nDCG重視 | **α=1.5** | R@1000 0.3185 / nDCG 0.5271 | R@1000 0.6182 / nDCG 0.4266 |
| recall重視 | **α=3** | R@1000 **0.3208** / nDCG 0.5208 | R@1000 **0.6269** / nDCG 0.4178 |

## 7.10 多段窓・リング窓（2026-09-09 実施、不採用）

### 発端: span_first の実装上の制約

実装を実測して2つの制約が確定した。

**(a) 位置を連続値として使えていない。** 現行の位置ブーストは OpenSearch の `span_first`
クエリを `bool/should` に1節足して boost を掛けるだけで、`_script_score` も `rescore` も
使っていない（`retriever.py` 全体で grep して0件）。`span_first` は Lucene の
`SpanFirstQuery`（`SpanPositionRangeQuery(0, end)`）で、explain を取ると
**窓内の span 頻度を tf とした BM25** で採点している。

```
end= 50 → score(freq= 15.0) = boost * idf * tf
end=100 → score(freq= 28.0)
end=400 → score(freq=145.0)
```

つまり「**先頭100語以内に何個ヒットしたか**」であって「**どれだけ早く出たか**」ではない。
位置5でマッチしても位置99でマッチしても同じ扱い。また**窓を広げるほど freq が増えて
スコアが上がる**方向に働く。

**(b) 固定窓は全文包含バイアスを生む。** 無作為サンプル300文書の body 語数は中央値938語。
窓が全文を覆って位置情報が消える文書の割合:

| span_end | 全文が窓に収まる文書 |
|---|---|
| 100（現行） | 2.0% |
| 200 | 6.0% |
| 400 | 16.0% |
| 800 | 42.7% |
| 1600 | 75.0% |

`span_end` を広げるほど位置ブーストは「語を含むかどうか」に退化する。**単純に窓を
広げる方向の探索は解釈できない。**

補足: `headings` フィールドとの関係も実測した。1文書あたりのユニーク見出しは平均21.8個
だが、**body 先頭100語に含まれる見出しは平均29.2%（中央値18.8%）**、200語でも37.2%。
全見出しを覆えた文書は10〜12%のみ。つまり位置ブーストは見出しの約7割を見ておらず、
`headings` フィールドへの match とは**カバー範囲が違う**（両者は重複しきっていない）。

### 比較した3案

スクリプト: [`evaluate_multiwindow_posboost.py`](experiments/gemini_fewshot_each_boost/evaluate_multiwindow_posboost.py)
実装: `retriever.py` の `bm25_equalweight_multiwindow`

- **cascade（累積多段窓）**: 入れ子の `span_first` を `[(50,b1),(100,b2),(200,b3),(400,b4)]`
  と重ねる。位置減衰関数の階段近似。ただし窓が入れ子なので**全文が窓に収まる短い文書は
  全段を総取り**し、問題(b)を解決しない
- **ring（排他リング窓）**: `span_not(include=span_first(b), exclude=span_first(a))` で
  `[0,50),[50,100),[100,200),[200,400)` の排他区間を作る。80語の文書は `[200,400)` に
  原理的にマッチできないため総取りが起きない。**問題(b)に直接対処するのはこちら**
  （実測で `ring[200,400)` にマッチした文書の body 語数は最小350語、200語未満は0件）
- **スケジュール**: `flat`（全段同じ）/ `geom0.5`（8:4:2:1。窓幅が倍々なので
  「boost×end 一定＝面積一定」と数学的に同一）/ `geom0.25`（64:16:4:1、急減衰）。
  いずれも合計が SCALE（15 または 30）になるよう正規化

### 結果（35トピック、consensus）

| 条件 | recall@1000 | nDCG@10 | 検索コスト |
|---|---|---|---|
| **champion（単一窓）** | 0.2982 | **0.5007** | **1.00倍** |
| `ring_flat_s15` | **0.3033** (+0.0052) | 0.4918 (−0.0089) | 5.68倍 |
| `cascade_flat_s15` | 0.2990 (+0.0009) | 0.4962 (−0.0045) | 3.08倍 |
| `cascade_geom0.5_s15` | 0.2968 | 0.4873 (−0.0133) | 3.48倍 |
| `ring_geom0.25_s30` | 0.2812 | 0.4649 (−0.0358) | 5.75倍 |

**nDCG@10 は全12条件で champion を下回った。**

### わかったこと

**(1) 減衰を付けるほど悪化する。** `flat` が最良で、`geom0.5` → `geom0.25` と減衰を
急にするほど単調に悪くなる。**冒頭を強く優遇するほど悪い**という、位置減衰の発想とは
逆の結果。

これは制約(a)と整合する。`span_first` は窓内で位置を区別せず、窓を広げると freq が
増えてスコアも上がる（freq 15→145）。減衰スケジュールはその freq 増加を打ち消す方向に
働くので、**効いている要因をわざわざ潰していた**ことになる。`flat` が最良ということは、
効いているのは「位置による重み付け」ではなく「**窓を広げて対象文書を増やすこと**」である。

**(2) スケール30は全条件で悪化。** 位置ブーストが BM25 本体を圧倒すると崩れる。
既存の `span_boost` 探索で50が明確に悪化したのと整合。

**(3) リング窓は recall だけ改善。** `ring_flat_s15` は recall@1000 +0.0052、
recall@100 +0.0014、precision@100 +0.0183 の3指標が改善する一方 **nDCG@10 は −0.0089**。
短文書バイアスの除去は recall 側には効くが、上位10件には効かない。

**(4) 不採用。** recall の +0.0052 は run 間ノイズ（0.001〜0.004）をわずかに超える程度で、
**5.68倍のコスト**と nDCG の低下を正当化できない。discourseboost と同じ
「効果なし・コスト増」の類型。

**論文への使い方**: 「単一固定窓のままが最良」という否定的知見として書ける。加えて
問題(b)の実測（end=800で43%、1600で75%が全文包含）は、**`span_end` を延長する方向の
改善が筋の悪いことの根拠**になる。


## 7.12 位置ブーストは「位置」ではなく「窓」で効いている（2026-09-10 実施）

### 問い

現行の位置ブーストが効く理由には2つの説明があり、区別できていなかった。

- **説A（位置）**: 要点は文書の冒頭に書かれやすいから当たる（FirstP の直感）
- **説B（希釈）**: 窓で区切ると長い文書での語の希釈が減るから当たる。**冒頭である必要はない**

発表資料（2026-09-07）の位置ブーストの根拠は説A（FirstP）なので、この切り分けは避けられない。

### 方法

FirstP（先頭の窓のみ）と **MaxP（最良の窓を文書のどこからでも探す）** を同じ機構で計算して
比較する。**MaxP は位置を一切見ない**（Dai & Callan 2019 らのパッセージ最大値法。解いて
いるのは「長い文書で語が希釈される」問題であって位置の問題ではない）ので、説A/説Bの
直接の対照になる。

実装は [`evaluate_position_rerank.py`](experiments/gemini_fewshot_each_boost/evaluate_position_rerank.py)。
`_mtermvectors` の `positions: True` で語ごとの全出現位置を取得し、Python 側でリランクする。
painless の `_index` API（expert scripting）は ES7 以降で削除されており使えないため、これが
唯一の経路。位置ありのペイロードは 78.7 KB/文書と重いので、35トピック・リランク深度100件・
docid 単位のキャッシュで軽量化した（18分で完了）。

条件はすべて同一の候補プール（位置ブーストなしの fielded 検索）に対するリランクで、
加点の式だけが違う。W=100、BOOST=15（champion に合わせた）。

### 結果（35トピック）

| 条件 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| **consensus** | | | | |
| `nopos`（加点なし） | 0.0587 | 0.2455 | 0.4492 | 0.6660 |
| `firstp`（先頭窓） | 0.0609 | 0.2455 | 0.4871 | 0.6940 |
| **`maxp`（最良窓・位置不問）** | **0.0612** | 0.2455 | **0.4927** | **0.6974** |
| `maxp_wlen`（窓長で正規化） | 0.0593 | 0.2455 | 0.4560 | 0.6754 |
| `decay`（初出位置の連続減衰） | 0.0606 | 0.2455 | 0.4776 | 0.6897 |
| `champion_ref`（実クエリ） | 0.0677 | 0.2982 | 0.5007 | 0.7669 |
| **coverage**（採点対象5件） | | | | |
| `nopos` | 0.2303 | 0.5506 | 0.3305 | 0.3040 |
| `firstp` | 0.2353 | 0.5506 | 0.3769 | 0.3120 |
| **`maxp`** | 0.2363 | 0.5506 | **0.3866** | 0.3140 |
| `maxp_wlen` | 0.2320 | 0.5506 | 0.3462 | 0.3080 |
| `decay` | **0.2366** | 0.5506 | 0.3200 | **0.3140** |
| `champion_ref` | 0.2537 | 0.6142 | 0.3634 | 0.3380 |

### わかったこと

**(1) 説Aは支持されない。** FirstP − MaxP（正なら位置が有利）は**両 qrels とも負**:

| 指標 | consensus | coverage |
|---|---|---|
| recall@100 | -0.0003 | -0.0010 |
| nDCG@10 | **-0.0056** | **-0.0096** |
| precision@100 | -0.0034 | -0.0020 |

差は小さいので「ほぼ同じ」と読むのが妥当だが、**「先頭が有利」という証拠は無い**。
むしろ最良窓を探す方がわずかに良い。

**(2) 効いているのは窓そのもの。** `nopos`（0.4492）→ `firstp`（0.4871）で +0.0379、
`maxp`（0.4927）で +0.0435。窓を導入した時点で改善の大半が得られ、その窓を先頭に
固定するかどうかは本質的でない。

**これは §7.10 の多段窓実験（減衰なしの `flat` が最良、冒頭を強く優遇するほど悪化）と
完全に整合する。独立した2つの実験が同じ結論を示した。**

**(3) 論文の記述を変える必要がある。** 位置ブーストの説明を「文書の要点は冒頭に
書かれやすい」ではなく「**窓で区切ることで長い文書における語の希釈が減る**」に
書き換えるべき。FirstP を根拠として引くのも不適切になる。

**(4) 連続減衰・窓長正規化はいずれも不発。** `decay`（`1/(1+p/100)`）は nDCG で `firstp` に
負け（consensus 0.4776 vs 0.4871）、`span_end`/`span_boost` を減衰定数1個に置き換える案は
成立しない。`maxp_wlen`（窓長で正規化）は全指標で最下位。

### 注意: この実験は並べ替え能力の比較であって champion の再現ではない

`firstp` は `champion_ref` を再現していない（consensus nDCG -0.0136、recall@1000 -0.0527）。
リランクは**上位100件の並べ替えしかできない**のに対し、実際の `span_first` は**検索時に
効いて候補集合そのものを変える**ため。全リランク条件で recall@1000 が 0.2455 と同一なのが
その証拠である。FirstP vs MaxP の相対比較は両条件が同じ制約を受けるので妥当だが、
絶対値を champion と比べてはいけない。

## 8. 優先すべき次の一手

1. ~~**オラクルを現行パイプラインで測り直す**~~ — **2026-09-10 に完了、§4 参照。** 逆転は
   解消したが伸びしろは小さく（champion比 +0.03〜0.04）、「機構で頭打ち」の判定。
2. ~~**`span_end` / `span_boost` のグリッド拡張**~~ — 代わりに位置ブースト自体の根拠を
   問い直した（§7.12）。「先頭が重要」という仮説は支持されず、効いているのは窓による
   希釈低減であることが判明したため、単純なグリッド拡張よりこちらを優先した。
3. ~~**BM25F の再実装と再測定**~~ — **2026-09-09〜10 に2段階で完了、§3.4 参照。**
   1段目（narrative×5反映）の後に `_bm25f_avgdl()` のプローブ語バグ（"the"がストップ
   ワードで除去され avgdl=1.0 にフォールバック）を発見し、2段目として b の3軸グリッドと
   qtf/k3 スイープをやり直した。最終値: consensus nDCG@10 = 0.4533（linear qtf,
   b=0.0/0.8/1.0）、champion との差は −0.0437。「線形和 > 真の BM25F」という結論は
   公平な比較でも生き残ったので、論文の主張に使ってよい。ただし recall@1000 は両 qrels
   で BM25F の方が線形和より上（consensus 0.2834 vs 0.2707、coverage 0.5890 vs 0.5425）
   である点は注記が要る。avgdlバグ修正後は qtf を**線形**に掛けるのが最良（k3=∞の極限）で、
   飽和させるほど悪化する — バグ下の結論（飽和が必要）とは逆転している。
4. **`spanterms_plus_narrative` の採否を決める** — nDCG 優先なら champion を差し替える判断も
   あり得る。ただし recall@1000 は下がるので、後段にリランカーを置く前提かどうかで結論が変わる。
5. ~~**gemini で logprob が取れるようになったか再確認**~~ — **2026-09-09 に確認済み、取れない。**
   `google/gemini-3.7-flash` に `logprobs: true, top_logprobs: 1` を付けて実リクエストを投げたが、
   本文が正常に生成された応答（`finish_reason: stop`）でも `logprobs` は `null` のままだった。
   当時の観測から仕様は変わっていない。**したがって confidence 重み付けを gemini に統一する
   ための再生成（約 $0.2）はやる意味がない**（払っても null が返るだけ）。confidence 方式は
   gpt-5.6-terra の疑似文書専用のままとし、gemini 側の項重み付けは IDF ベースを使う。

   副産物として、生成スクリプトが使っている `reasoning: {"effort": "none"}` は gemini では
   **400 エラーで拒否される**ことも分かった（`Reasoning is mandatory for this endpoint and
   cannot be disabled.`）。実際 295 完了トークン中 277 が reasoning トークンだった。
   gemini でこの実験系を回すときは reasoning を無効化できない前提でコストを見積もること。
