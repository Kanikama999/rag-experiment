# フィールドの強さ実験（title/headings/body アブレーション）まとめ

構造化擬似参照文（title/headings/bodyのJSON、`../multi_query2doc_decomposed_webstyle_L200.json`）
のうち、どのフィールドがどれだけ検索性能に効いているかを切り分けた一連の実験をここに集約する。
共有パイプライン本体（`decompose_narrative.py`、`decomposed_query2doc_webstyle_expansion.py`、
`retriever.py`、生成済み疑似文書データ）は親ディレクトリ（`../`）に残したまま、
このディレクトリのスクリプトは起動時に親ディレクトリを`sys.path`へ追加して読みにいく。

## 0. baselineとは

どの表にも登場する`baseline`は、疑似参照文もSubquery分解も一切使わず、**narrative
（トピックの元の長いクエリ文）をそのまま1回bm25_body検索するだけ**の条件（比較の基準点）。
`evaluate_field_ablation.py`では以下の1行のみ:

```python
if method == "baseline":
    run[qid] = fuse([search_body(q)], TOPK)
```

`q`は繰り返し・分解・生成いずれも行っていない生のnarrative。他の全条件（title_only、
fieldedなど）はこのbaselineに対する改善幅（表の括弧内`+0.0xxx`）として比較している。

## 1. 単体 → ペア → 3フィールドのアブレーション（`evaluate_field_ablation.py`）

同じ入力テキスト（narrative×5 + Subquery + 対応フィールドのテキスト）を使い、
1フィールド・2フィールド・3フィールドの組み合わせを比較。3フィールド条件は
「既定boost（title^3+headings^2+body^1）」と「等倍boost（title^1+headings^1+body^1）」の
両方を用意し、単体・ペア条件（すべて等倍）と公平に比較できるようにした。

### coverage（22クエリ）

| method | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| title_only | 0.1101 (+0.0159) | 0.3091 (+0.0206) | 0.2149 (+0.0580) | 0.1795 (+0.0291) |
| headings_only | 0.1312 (+0.0370) | 0.3312 (+0.0428) | 0.3023 (+0.1454) | 0.2055 (+0.0550) |
| body_only | 0.0987 (+0.0044) | 0.2952 (+0.0068) | 0.2545 (+0.0976) | 0.1559 (+0.0055) |
| title_headings | 0.1599 (+0.0657) | 0.3929 (+0.1045) | 0.3105 (+0.1536) | 0.2573 (+0.1068) |
| title_body | 0.1610 (+0.0668) | 0.4972 (+0.2087) | 0.3391 (+0.1822) | 0.2468 (+0.0964) |
| headings_body | 0.1488 (+0.0546) | 0.4504 (+0.1619) | 0.3318 (+0.1749) | 0.2291 (+0.0786) |
| fielded（title^3+headings^2+body^1） | 0.2047 (+0.1105) | 0.5213 (+0.2328) | 0.4051 (+0.2482) | 0.3300 (+0.1795) |
| **fielded_equal（title^1+headings^1+body^1）** | **0.1925 (+0.0982)** | **0.5344 (+0.2459)** | **0.4091 (+0.2522)** | 0.2977 (+0.1473) |

### consensus（105クエリ）

| method | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| title_only | 0.0363 (+0.0035) | 0.1433 (-0.0062) | 0.2966 (+0.0317) | 0.4033 (+0.0296) |
| headings_only | 0.0448 (+0.0120) | 0.1651 (+0.0157) | 0.4146 (+0.1497) | 0.5096 (+0.1359) |
| body_only | 0.0334 (+0.0006) | 0.1406 (-0.0089) | 0.2999 (+0.0350) | 0.3845 (+0.0108) |
| title_headings | 0.0495 (+0.0167) | 0.1926 (+0.0431) | 0.4237 (+0.1588) | 0.5592 (+0.1855) |
| title_body | 0.0508 (+0.0180) | 0.2222 (+0.0727) | 0.4082 (+0.1433) | 0.5777 (+0.2040) |
| headings_body | 0.0467 (+0.0140) | 0.2044 (+0.0549) | 0.4123 (+0.1474) | 0.5315 (+0.1578) |
| fielded（title^3+headings^2+body^1） | 0.0620 (+0.0292) | 0.2617 (+0.1122) | 0.5028 (+0.2379) | 0.7020 (+0.3283) |
| fielded_equal（title^1+headings^1+body^1） | 0.0587 (+0.0260) | 0.2559 (+0.1065) | 0.4772 (+0.2122) | 0.6648 (+0.2910) |

（生データ: `decomposed_query2doc_webstyle_field_ablation_rep5.json`〈単体3条件、105クエリ〉、
`field_ablation_eval_summary_rep5.json`〈単体3条件+fielded既定boost、本ディレクトリで再実行〉、
`field_ablation_eval_summary_rep5_baseline-fielded_equal.json`、
`field_ablation_eval_summary_rep5_baseline-title_headings-title_body-headings_body.json`）

### 分かったこと

1. **単体の強さは headings > title > body**（両qrels共通）。
2. **フィールドを増やすほど単調に強くなる**（単体 → ペア → 3フィールド）。ペアはどの組み合わせも
   単体2つを足した以上の伸びを示しており、単体同士の相乗効果はペアの時点で既に出ている。
3. **ペア3種の間に大きな優劣はない**（consensus nDCG@10: title_headings 0.4237、title_body 0.4082、
   headings_body 0.4123と近い）。どの2つを組み合わせても、単体最強のheadings_only（0.4146）と
   同程度かやや上回る水準まで伸びる。
4. **既定boost（3,2,1）と等倍boost（1,1,1）の差は小さい**。既定boostの方が全指標でわずかに
   良い（consensus nDCG@10: 0.5028 vs 0.4772）が、coverageのnDCG@10・recall@1000では
   むしろequalの方が良い（0.4091 vs 0.4051、0.5344 vs 0.5213）。つまり「フィールドを分けて
   線形和で合算する」という構造自体が効果の大部分を占めており、title/headingsを重めにする
   という重み付け自体の寄与は相対的に小さい。
   - 単体条件（title_only等）は重み1で計測しているため、fieldedと公平に比較するには
     本来fielded_equalと比べるべき（既定boostのfieldedと比べると、フィールド分割の効果に
     重み付けの効果が混ざる）。

## 1.5 生のnarrativeだけをtitle/headingsフィールドに与えたら（`evaluate_field_ablation.py`）

上記の`title_only`/`headings_only`/`body_only`は「narrative×5 + Subquery + 疑似文書パート」を
対応フィールドにmatchする条件だった。ここでは疑似文書もSubquery分解も一切使わず、**baselineと
同じ生のnarrative（繰り返しなし、単一クエリ）を、bodyの代わりにtitle/headingsフィールドだけに
match**したらどうなるかを見る（baseline自体が「narrativeをbodyだけに」に相当するので、
narrative_title_only / narrative_headings_only の2条件を追加すれば3フィールド分がそろう）。

### coverage（22クエリ）

| method | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline（narrative→body） | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| narrative_title_only | 0.0972 (+0.0030) | 0.2710 (-0.0174) | 0.2088 (+0.0519) | 0.1582 (+0.0077) |
| **narrative_headings_only** | **0.1231 (+0.0289)** | **0.3074 (+0.0189)** | **0.2972 (+0.1403)** | **0.1936 (+0.0432)** |

### consensus（105クエリ）

| method | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|
| baseline（narrative→body） | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| narrative_title_only | 0.0318 (-0.0010) | 0.1218 (-0.0277) | 0.2821 (+0.0172) | 0.3619 (-0.0118) |
| **narrative_headings_only** | **0.0417 (+0.0090)** | **0.1515 (+0.0020)** | **0.4042 (+0.1392)** | **0.4796 (+0.1059)** |

（生データ: `field_ablation_eval_summary_rep5_baseline-narrative_title_only-narrative_headings_only.json`）

### 分かったこと

- **narrativeを与えるフィールドを変えるだけでも、headingsが最も強い**（narrative_headings_onlyは
  4指標すべてでbaselineを上回るが、narrative_title_onlyは4指標中3つで悪化気味）。疑似文書パート
  を使った1節の実験と同じ「headings > title > body」の順位が、疑似文書なしの生narrativeでも
  再現された。
- ただし改善幅は疑似文書ありの`headings_only`（consensus nDCG@10 +0.1497）の方が
  narrative単体（+0.1392）よりわずかに大きい。疑似文書やSubquery分解を足すことで
  headingsフィールドの効きがさらに強まっている。
- titleフィールドは生のnarrativeとの相性が悪い（consensus recall@1000 -0.0277、
  precision@100 -0.0118とbaselineより悪化）。疑似文書パートを使った`title_only`は
  全指標でbaselineを上回っていたので、titleフィールドは「narrativeそのもの」より
  「LLM生成のtitleらしい短い文」の方が合っている可能性が高い。

## 2. 繰り返し効果 vs フィールド構造効果の分離（`evaluate_decomposed_webstyle_ablation.py`）

「narrativeをN回繰り返す効果」と「title/headings/bodyに分けて線形和する効果」を分離した
アブレーション（`baseline` / `webstyle_flat_norepeat` / `webstyle_flat_repeat` /
`webstyle_narrative_fielded_recallopt`の4条件）。結果は
`decomposed_query2doc_webstyle_eval_summary_ablation_rep5.json`。

## 3. フィールド統合方式: 線形和 vs RRF（`evaluate_decomposed_webstyle_fieldrrf.py`）

title/headings/bodyへの3つのmatchを「1クエリ内でスコア線形和」ではなく「独立クエリ+RRF融合」に
した場合の比較。結果は`decomposed_query2doc_webstyle_eval_summary_fieldrrf_rep5.json`。

## 4. フィールドブースト比の最適化（`learn_field_boosts.py` / `search_field_boosts.py`）

- `learn_field_boosts.py`: qrelsに対する線形回帰でtitle/headings/bodyのブースト比を推定
  （`field_boost_regression.json`）。R²が弱く、実際の評価指標とはズレていたため不採用と判断。
- `search_field_boosts.py`: 実パイプラインでrecall@1000/nDCG@10を直接最大化するブースト比を
  座標降下法・グリッドサーチで探索（`field_boost_search_result.json`,
  `field_boost_search_result_coordinate_ascent_5x5.json`）。recallopt=title2:headings1:body1、
  ndcgopt=title3:headings3:body1が既定（3,2,1）よりわずかに良い。

## 5. 真のBM25F（フィールド横断統計合算）でのbパラメータ探索（`search_bm25f_field_b.py`）

線形和方式ではなく、真のBM25F（フィールド間でTF/IDF統計を合算してから1回だけBM25計算する
方式）でのフィールド別長さ正規化パラメータb_title/b_headings/b_bodyを探索。
結果は`bm25f_field_b_search_result.json`, `bm25f_field_b_search_result_coorddesc.json`
（ログ: `run_bm25f_field_b_refix.log`）。

## 6. title欠損文書の取りこぼし診断（`evaluate_field_absence.py`）

title欠損の正解文書がfielded線形和方式で拾われにくい問題を、BM25F/cross_fieldsなど
統合方式を変えて回収できるか検証。結果は`field_absence_result.json`
（タイトル有無キャッシュ: `_cache_relevant_title_presence.json`）。

## ディレクトリ構成

```
field_strength/
├── README.md                                            このファイル
├── evaluate_field_ablation.py                            単体/ペア/3フィールドアブレーション（新規）
├── decomposed_query2doc_webstyle_field_ablation_rep5.json 単体3条件の元データ（105クエリ）
├── field_ablation_eval_summary_rep5.json                 上記の再実行結果（単体3条件+fielded）
├── field_ablation_eval_summary_rep5_baseline-fielded_equal.json         等倍boost追加分
├── field_ablation_eval_summary_rep5_baseline-title_headings-title_body-headings_body.json  ペア3条件
├── field_ablation_eval_summary_rep5_baseline-narrative_title_only-narrative_headings_only.json  生narrative→title/headings単体
├── evaluate_decomposed_webstyle_ablation.py              繰り返し効果 vs フィールド構造効果
├── decomposed_query2doc_webstyle_eval_summary_ablation_rep5.json
├── evaluate_decomposed_webstyle_fieldrrf.py               線形和 vs RRF統合
├── decomposed_query2doc_webstyle_eval_summary_fieldrrf_rep5.json
├── learn_field_boosts.py / field_boost_regression.json    ブースト比: 線形回帰（不採用）
├── search_field_boosts.py / field_boost_search_result*.json  ブースト比: 直接探索
├── search_bm25f_field_b.py / bm25f_field_b_search_result*.json / run_bm25f_field_b_refix.log  真のBM25Fのb探索
└── evaluate_field_absence.py / field_absence_result.json / _cache_relevant_title_presence.json  title欠損診断
```

親ディレクトリの`REPORT.md`（4.2節・4.3節）はこのディレクトリ整理前の記述のままなので、
数値は一致するが「ファイルはこのフォルダに移動済み」である点に注意。
