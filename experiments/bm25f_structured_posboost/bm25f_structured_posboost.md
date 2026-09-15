# bm25f_structured_posboost — document版の最高性能パイプライン

真のBM25F（k1=3.0）に、LLM生成の構造化疑似文書と位置ブーストを組み合わせた設定。
`experiments/` 配下の全結果（JSON 209個＋結果Markdown）を横断して、**全トピックで評価した
結果の中で両qrels・全4指標の1位**（2026-09-14時点）。学習済みリランカーは使っていない。

このフォルダ単体で完結する（データ・コード・生成済み疑似文書・結果を同梱）。元フォルダ
`gemini_fewshot_each_boost/` 等からのコピーで、元ファイルには手を加えていない。

---

## 1. 性能

n=105（consensus）/ n=22（coverage）、TOPK=1000。

| qrels | 設定 | recall@100 | recall@1000 | nDCG@10 | precision@100 |
|---|---|---|---|---|---|
| **consensus** | baseline（narrativeそのまま bm25_body） | 0.0327 | 0.1495 | 0.2649 | 0.3737 |
| | champion（線形和＋位置ブースト） | 0.0689 | 0.3095 | 0.5251 | 0.7769 |
| | 旧記録 BM25F k1=3.0（`structured_base`） | 0.0739 | 0.3468 | 0.5510 | 0.8446 |
| | **bm25f_structured_posboost** | **0.0753** | **0.3631** | **0.5533** | **0.8612** |
| **coverage** | baseline | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| | champion | 0.2373 | 0.5995 | 0.4049 | 0.3768 |
| | 旧記録 BM25F k1=3.0（`structured_base`） | 0.2771 | 0.6835 | 0.4750 | 0.4423 |
| | **bm25f_structured_posboost** | **0.2796** | **0.6970** | **0.4760** | **0.4491** |

baseline比で recall@1000 は consensus ×2.43 / coverage ×2.42、nDCG@10 は ×2.09 / ×3.03。

結果: [`bm25f_posboost_full_result.json`](bm25f_posboost_full_result.json)（`structured_posboost`）

---

## 2. パイプライン

```
narrative（TRECトピックの長い質問文）
  │
  │ ① decompose_narrative.py
  │    gemini-3.7-flash（batch, reasoning=low, temperature=0.0）
  ▼
Subquery × 平均4.3個/トピック（計451個）          → decomposed_queries.json
  │
  │ ② decomposed_query2doc_webstyle_expansion.py
  │    gemini-3.7-flash（batch, reasoning=low, temperature=1.0, few-shot 3例, 200語）
  ▼
構造化疑似文書 {title, headings, body}（Subquery 1個につき1本）
                                                   → multi_query2doc_decomposed_webstyle_L200.json
  │
  │ ③ ここから検索。Subqueryごとに独立に行う（evaluate_bm25f_posboost_full.py）
  ▼
クエリテキスト（narrativeは5回繰り返す）
  title    : narrative×5 + Subquery + 生成title
  headings : narrative×5 + Subquery + 生成headings
  body     : narrative×5 + Subquery + 生成body
  │
  ├─ Stage 1（OpenSearch）候補プール取得 ─────────────────────────────
  │    bool/should
  │      match title    (boost 2)
  │      match headings (boost 1)
  │      match body     (boost 1)
  │      span_first(span_or(analyze_terms(Subquery)), body, end=100, boost=15)  ← 位置ブースト
  │    → 上位 3000 件
  │
  ├─ Stage 2（Python）BM25Fで採点し直す ──────────────────────────────
  │    _mtermvectors で各候補の title/headings/body の tf とフィールド長を取得
  │    擬似tf(t,d) = Σ_f w_f · tf(t,f,d) / ((1−b_f) + b_f · dl_f/avgdl_f)
  │    score(d)    = Σ_t idf(t) · 擬似tf/(k1 + 擬似tf) · qtf(t)
  │      w = (title 2, headings 1, body 1)
  │      b = (title 0.0, headings 0.8, body 1.0)
  │      k1 = 3.0
  │      idf = log(1 + (N−df+0.5)/(df+0.5))、df は3フィールド横断（二重計上なし）
  │      qtf = クエリ全文中の語の出現回数（線形。narrative×5 がここで効く）
  │    → 上位 3000 件
  │
  ▼
Subqueryごとのランキング
  │ ④ RRF（k=60）で融合
  ▼
トピックの最終ランキング（上位1000件）
```

### 位置ブーストの置き場所がポイント

Stage 2 は tf とフィールド長しか持たないので、位置情報を使えない。そこで位置ブーストを
**Stage 1 の候補取得クエリ**に入れた。「Subqueryの語が本文の先頭100語に出る文書」が
3000件の枠に入ってくるので、候補集合そのものが変わり recall@1000 が動く。

- 位置ブーストあり／なしで候補プールの重複は63%（qid=1001の1 Subqueryで実測、3000件中1888件）。
- 事後リランク（上位100件の並べ替え）では recall@1000 は原理的に動かない
  （`EXPERIMENT_SUMMARY.md` §7.12 で全条件 0.2455 に固定）。

### これは「正規の」BM25Fではない

Robertson・Zaragoza・Taylor（2004）のBM25Fは全文書をこの式で採点するもので、候補の
事前絞り込みは含まない。OpenSearch 3.0.0 にはBM25Fをネイティブに計算するクエリがない
（`combined_fields` は `unknown query` になる）ため、`retriever.py` は
「フィールド線形和BM25で候補を取り、BM25Fの式で並べ替える」2段構成で代用している。

- Stage 1 で3000件に入らなかった文書は、BM25Fなら上位に来るはずでも拾えない。
  recall@1000 は `RETRIEVE_K` に律速される。
- Stage 2 の採点式自体は正確（`retriever.bm25f_score` とスコア差0を確認済み）。
- 並べ替えは語彙ベースの式で、学習済みモデルは使っていない。

---

## 3. 何がどれだけ効いているか

本番実行（n=105、RETRIEVE_K=3000）の5条件。すべて同じトピック・同じ深さ。

| 条件（疑似文書 / 位置ブースト） | consensus R@100 | consensus R@1000 | consensus nDCG@10 | consensus P@100 | coverage R@100 | coverage R@1000 | coverage nDCG@10 | coverage P@100 |
|---|---|---|---|---|---|---|---|---|
| `baseline`（— / —） | 0.0327 | 0.1495 | 0.2649 | 0.3737 | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| `narrative_only_base`（なし / なし） | 0.0718 | 0.3158 | 0.5378 | 0.8197 | 0.2392 | 0.5908 | 0.4356 | 0.3855 |
| `narrative_only_posboost`（なし / あり） | 0.0728 | 0.3278 | 0.5469 | 0.8288 | 0.2445 | 0.6101 | 0.4155 | 0.3927 |
| `structured_base`（構造化 / なし） | 0.0739 | 0.3468 | 0.5510 | 0.8446 | 0.2771 | 0.6835 | 0.4750 | 0.4423 |
| `structured_posboost`（構造化 / あり） | **0.0753** | **0.3631** | **0.5533** | **0.8612** | **0.2796** | **0.6970** | **0.4760** | **0.4491** |

- `baseline`：narrativeを1回だけ `bm25_body` に投げる素のBM25。
- `narrative_only_*`：3フィールドとも同じテキスト `narrative×5 + Subquery` を入れる。
  LLM生成テキストを使わない以外は、BM25F・k1・qtf・RRFとも本番と同じ。

### 積み上げ

| 追加したもの | consensus R@100 | consensus R@1000 | consensus nDCG@10 | consensus P@100 | coverage R@100 | coverage R@1000 | coverage nDCG@10 | coverage P@100 |
|---|---|---|---|---|---|---|---|---|
| baseline（値） | 0.0327 | 0.1495 | 0.2649 | 0.3737 | 0.0942 | 0.2884 | 0.1569 | 0.1505 |
| ＋BM25F・k1=3.0・Subquery分解・narrative×5（`narrative_only_base`） | +0.0390 | +0.1664 | +0.2729 | +0.4460 | +0.1450 | +0.3023 | +0.2787 | +0.2350 |
| ＋構造化疑似文書（`structured_base`） | +0.0022 | +0.0310 | +0.0132 | +0.0249 | +0.0379 | +0.0927 | +0.0394 | +0.0568 |
| ＋位置ブースト（`structured_posboost`） | +0.0014 | +0.0162 | +0.0023 | +0.0167 | +0.0025 | +0.0135 | +0.0010 | +0.0068 |
| **合計（値）** | **0.0753** | **0.3631** | **0.5533** | **0.8612** | **0.2796** | **0.6970** | **0.4760** | **0.4491** |

### 位置ブーストの効果（`structured_posboost` − `structured_base`）

| 指標 | consensus | coverage |
|---|---|---|
| recall@100 | +0.0014 | +0.0025 |
| recall@1000 | +0.0162 | +0.0135 |
| nDCG@10 | +0.0023 | +0.0010 |
| precision@100 | +0.0167 | +0.0068 |

両qrels・全8セルでプラス。consensus recall@1000 の +0.0162 は、プロジェクトで定めている
run間ノイズ（recall@1000 で 0.001〜0.004、`EXPERIMENT_SUMMARY.md` §5）の4倍以上。
nDCG@10 はほぼ横ばい（下がってはいない）。

### 構造化疑似文書の効果（`structured_base` − `narrative_only_base`）

| 指標 | consensus | coverage |
|---|---|---|
| recall@100 | +0.0022 | +0.0379 |
| recall@1000 | +0.0310 | +0.0927 |
| nDCG@10 | +0.0132 | +0.0394 |
| precision@100 | +0.0249 | +0.0568 |

---

## 4. パラメータがどう決まったか

いずれも同梱のスクリプトと結果JSONで再確認できる。

### b（長さ正規化）: `search_bm25f_b_3way.py` → `bm25f_b_3way_grid_result.json`

b∈{0.0, 0.4, 0.8, 1.0} の3軸同時グリッド（64セル）。探索条件は **consensus の qrels のみ・35トピック（valid_qids[::3]）・
candidate_k=200・k1=0.9・qtf未反映（`retriever.bm25f_score`）**。coverage では評価していない。

採用したのは **nDCG@10 が最良のセル** `b_title=0.0, b_headings=0.8, b_body=1.0`。

| b（title / headings / body） | consensus recall@100 | consensus recall@1000 | consensus nDCG@10 | consensus precision@100 |
|---|---|---|---|---|
| 0 / 0.8 / 1 | **0.0588** | **0.1435** | **0.3777** | **0.6700** |
| 0.4 / 0.8 / 1 | 0.0589 | 0.1435 | 0.3777 | 0.6703 |
| 0 / 1 / 1 | 0.0589 | 0.1435 | 0.3766 | 0.6711 |
| 0.4 / 1 / 1 | 0.0589 | 0.1435 | 0.3766 | 0.6711 |
| 0.8 / 0.8 / 1 | 0.0589 | 0.1435 | 0.3763 | 0.6703 |
| 0.4 / 0.4 / 0.4（既定値） | 0.0538 | 0.1435 | 0.3299 | 0.6146 |

- **recall@1000 は64セルすべて同じ値（0.1435）**。候補200件の並べ替えしか起きないので、b では動かない。
  つまり b は recall では選ばれていない。
- 採用セルの順位は nDCG@10 で 1/64（0.4 / 0.8 / 1.0 と小数点4桁まで同点で、グリッドの走査順で先に来た方）、recall@100 と precision@100 では 8/64・8/64（1位との差はそれぞれ 0.0001・0.0014）。
- b_title=0.0 と b_body=1.0 はどちらも定義域 [0, 1] の端。title は平均7.2語でほぼ長さが揃っているので割り引く意味が薄く、
  body は平均1,097語で長さの差が大きいので全面的に割り引くのが効く、と解釈できる。

`_bm25f_avgdl()` のプローブ語バグ（ストップワード "the" で avgdl が 1.0 に落ちていた）
を 2026-09-10 に修正した後の値。バグ下で決めた b（0.6/0.2/0.2）は無効。

### qtf（クエリ側の重み）: `evaluate_bm25f_k3_sweep.py` → `bm25f_k3_sweep_result.json`

k3∈{0,1,2,4,8,16,32,64} と線形の9条件（n=105、k1=0.9）。両qrels・全指標で k3 に対し
単調増加し、**線形（k3=∞）が最良**。narrative×5 の繰り返しはそのまま重みとして掛ける。

| k3 | consensus R@100 | consensus R@1000 | consensus nDCG@10 | consensus P@100 | coverage R@100 | coverage R@1000 | coverage nDCG@10 | coverage P@100 |
|---|---|---|---|---|---|---|---|---|
| 0（重複除去＝qtf無視） | 0.0454 | 0.2242 | 0.3476 | 0.5230 | 0.1488 | 0.4806 | 0.3063 | 0.2323 |
| 1 | 0.0485 | 0.2367 | 0.3753 | 0.5560 | 0.1629 | 0.5057 | 0.3444 | 0.2541 |
| 2 | 0.0505 | 0.2441 | 0.3945 | 0.5767 | 0.1704 | 0.5230 | 0.3594 | 0.2650 |
| 4 | 0.0527 | 0.2537 | 0.4072 | 0.6017 | 0.1788 | 0.5405 | 0.3656 | 0.2791 |
| 8 | 0.0550 | 0.2629 | 0.4187 | 0.6248 | 0.1901 | 0.5566 | 0.3682 | 0.2955 |
| 16 | 0.0566 | 0.2703 | 0.4324 | 0.6434 | 0.1948 | 0.5662 | 0.3737 | 0.3041 |
| 32 | 0.0577 | 0.2754 | 0.4393 | 0.6553 | 0.1999 | 0.5754 | 0.3784 | 0.3123 |
| 64 | 0.0584 | 0.2789 | 0.4463 | 0.6626 | 0.2020 | 0.5814 | 0.3829 | 0.3150 |
| 線形（k3=∞） | **0.0594** | **0.2834** | **0.4533** | **0.6732** | **0.2044** | **0.5890** | **0.3910** | **0.3195** |

### k1（TF飽和）: `evaluate_bm25f_k1_length_sweep.py` → `bm25f_k1_length_sweep_result.json`

k1_base × 文書長依存の減衰α のグリッド（n=105、RETRIEVE_K=3000）。α=0（k1定数）が最良で、
k1 は大きいほど良い。

| k1 | consensus R@100 | consensus R@1000 | consensus nDCG@10 | consensus P@100 | coverage R@100 | coverage R@1000 | coverage nDCG@10 | coverage P@100 |
|---|---|---|---|---|---|---|---|---|
| 0.9 | 0.0594 | 0.2834 | 0.4533 | 0.6732 | 0.2044 | 0.5890 | 0.3910 | 0.3195 |
| 1.5 | 0.0672 | 0.3170 | 0.5061 | 0.7634 | 0.2389 | 0.6380 | 0.4377 | 0.3773 |
| 2.0 | 0.0704 | 0.3320 | 0.5246 | 0.8025 | 0.2590 | 0.6611 | 0.4499 | 0.4118 |
| 3.0 | **0.0739** | **0.3468** | **0.5510** | **0.8446** | **0.2771** | **0.6835** | **0.4750** | **0.4423** |

**3.0 はグリッドの上端で、まだ頭打ちしていない。**

同梱の `evaluate_k1_decay_sweep.py`（k1 0.1〜2.0 の定数スイープ）は「k1を下げると悪化する」
という動機付けになった実験。ただし**バグ下の b（0.6/0.2/0.2）で回っている**ので、
絶対値は他と比較できない。

### Stage 2 の寄与: `evaluate_bm25f_norerank.py` → `bm25f_norerank_result.json`

Stage 1 のヒット順をそのまま使い、BM25F再採点をしない場合（位置ブーストなし）:

|  | consensus R@100 | consensus R@1000 | consensus nDCG@10 | consensus P@100 | coverage R@100 | coverage R@1000 | coverage nDCG@10 | coverage P@100 |
|---|---|---|---|---|---|---|---|---|
| Stage 1 のみ（norerank） | 0.0621 | 0.2707 | 0.4970 | 0.7048 | 0.2090 | 0.5425 | 0.4109 | 0.3309 |
| ＋BM25F再採点 k1=3.0（`structured_base`） | **0.0739** | **0.3468** | **0.5510** | **0.8446** | **0.2771** | **0.6835** | **0.4750** | **0.4423** |

Stage 1 のみの値はフィールド線形和 recallopt（title 2:1:1）と一致する。BM25Fの式で
並べ替え直すことが recall@1000 を +0.076 押し上げている。

### 位置ブースト: パイロット → 本番

| 位置ブーストの改善幅（structured_posboost − structured_base） | consensus R@100 | consensus R@1000 | consensus nDCG@10 | consensus P@100 | coverage R@100 | coverage R@1000 | coverage nDCG@10 | coverage P@100 |
|---|---|---|---|---|---|---|---|---|
| パイロット（consensus 35 / coverage 9トピック、RETRIEVE_K=1000） | +0.0019 | +0.0350 | +0.0068 | +0.0223 | +0.0110 | +0.0199 | +0.0094 | +0.0156 |
| 本番（consensus 105 / coverage 22トピック、RETRIEVE_K=3000） | +0.0014 | +0.0162 | +0.0023 | +0.0167 | +0.0025 | +0.0135 | +0.0010 | +0.0068 |

候補を3000件まで取ると、位置ブーストが引き込んでいた文書の一部が元から入るため、
効果は小さくなる。パイロットのトピック集合はプロジェクト標準（stride-3）と違うので、
他のn=35実験とは比較できない。

### 参考: 疑似文書なし／フラット疑似文書（`evaluate_bm25f_nopseudo.py` / `evaluate_bm25f_flatpseudo.py`）

k1=3.0、RETRIEVE_K=3000、n=105。

| 条件（k1=3.0、qtf未反映） | consensus R@100 | consensus R@1000 | consensus nDCG@10 | consensus P@100 | coverage R@100 | coverage R@1000 | coverage nDCG@10 | coverage P@100 |
|---|---|---|---|---|---|---|---|---|
| nopseudo（narrative×5 + Subquery） | 0.0692 | 0.3096 | 0.5199 | 0.7903 | 0.2331 | 0.5863 | 0.4207 | 0.3755 |
| flatpseudo（＋疑似文書を連結して3フィールド共通） | 0.0577 | 0.2820 | 0.4148 | 0.6570 | 0.1969 | 0.5882 | 0.3654 | 0.3105 |

**この2つは qtf を反映していない**（`retriever.bm25f_score` を直接呼んでおり、narrative×5 が
採点に効かない）。qtf線形の系列（`structured_base`、`narrative_only_base`）と直接比べると
qtf の差が混ざる。nopseudo と `narrative_only_base`（0.3158）の違いは主にこれ。

---

## 5. 妥当性チェック

- **旧記録の再現**：`structured_base` は `k1b3_a0`（`bm25f_k1_length_sweep_result.json`）の
  値を、両qrels・全4指標で小数点4桁まで再現した（ログ末尾の「妥当性チェック」）。
- **採点式**：Stage 2 の再実装は `retriever.bm25f_score`（qtf=1のとき）とスコア差0、順位完全一致。
- **位置ブーストの節**：`retriever.bm25_fielded_posboost` と同一の `span_first`/`span_or` を生成。
- **最適化の等価性**：termvectorsをトピック単位で1回だけ取得する版は、Subquery単位で取得する
  旧版と5条件すべてで結果が一致（1000件の順位一致、スコア差0）。
- **フォルダ単体での動作**：フォルダ外から実行してもパスはすべてフォルダ内に解決され、
  qid=100 の再計算が同梱チェックポイントと5条件すべてで一致した。

---

## 6. 未解決・注意点

- **`span_boost=15`, `span_end=100` はchampion（線形和）向けの値**で、BM25F向けには探索していない。
- **k1 はグリッド上端（3.0）で最良**。3.0 より大きい値は未探索。
- **b は現行条件で再探索していない**。探索は k1=0.9・qtf未反映・候補200件・consensusのみ・35トピックで、
  recall@1000 が b で動かない設定だった。現行（k1=3.0・qtf線形・候補3000件・位置ブーストあり）では、
  Stage 2 の並べ替えがRRF後の上位1000件の顔ぶれに影響するので、b は recall@1000 にも効きうる。
- **位置ブーストと疑似文書の交互作用は読み取れない**。固定の `span_boost=15` はベーススコアに
  対する比率が条件によって違う（qid=1001実測で structured 2.6% / narrative_only 3.5%）。
  `narrative_only_posboost` で coverage nDCG@10 が下がった（−0.020）のが交互作用なのか、
  この比率の差なのかは区別できない。
- **Stage 1 の深さに依存**する（§2）。RETRIEVE_K を変えると recall@1000 も変わる。
- coverage は22トピックなので、coverage側の小さい差は弱い証拠として扱うこと。

---

## 7. 再現方法

### 必要なもの（フォルダ外）

- Python 環境：`opensearch-py`, `pytrec_eval`, `requests`（開発時は `/home/takeuchi/venvs/rag`）
- OpenSearch：`http://localhost:9200`、インデックス `msmarco-v21-doc`（MS MARCO v2.1 文書、10,960,555件）
- 疑似文書を作り直す場合のみ：環境変数 `OPENROUTER_API_KEY`

### 評価（生成済みの疑似文書を使う。約2時間）

```bash
cd experiments/bm25f_structured_posboost
python -u evaluate_bm25f_posboost_full.py > run_bm25f_posboost_full.log 2>&1
```

チェックポイント（`bm25f_posboost_full_ckpt.json`）があれば完了済みトピックは飛ばす。
**同梱のチェックポイントは全105トピック分なので、そのまま実行すると集計だけして終わる**。
最初から回し直すときはチェックポイントを別名に退避してから実行する。

所要時間の内訳（1 Subqueryあたり、実測）：`_mtermvectors` 64%、候補検索 32%
（位置ブーストありの検索はなしの2.8倍）、Pythonの計算 数%。termvectorsは本文の全語を返し、
使うのは約15%だけ。

### 疑似文書から作り直す場合

どちらもサブコマンドは `smoke`（数件だけ試す）/ `submit`（Batch投入）/ `poll`（進捗確認、既定）/ `fetch`（結果取得）。

```bash
python decompose_narrative.py submit
python decompose_narrative.py poll        # 完了まで
python decompose_narrative.py fetch       # → decomposed_queries.json

python decomposed_query2doc_webstyle_expansion.py submit
python decomposed_query2doc_webstyle_expansion.py poll
python decomposed_query2doc_webstyle_expansion.py fetch   # → multi_query2doc_decomposed_webstyle_L200.json
```

LLMの出力は毎回変わる（疑似文書生成は temperature=1.0）ので、同梱の疑似文書と完全には一致しない。

---

## 8. ファイル一覧

### パイプライン本体

| ファイル | 役割 |
|---|---|
| `data/trec_rag_2025_queries.jsonl` | トピック（narrative） |
| `data/keystone_qrels_consensus.txt` / `data/keystone_qrels_coverage.txt` | 正解判定（105 / 22トピック） |
| `decompose_narrative.py` → `decomposed_queries.json` | ① narrative → Subquery |
| `decomposed_query2doc_webstyle_expansion.py` → `multi_query2doc_decomposed_webstyle_L200.json` | ② Subquery → 構造化疑似文書 |
| `batch_ids_decomposed_query2doc_webstyle.json` | ②の生成時のBatch APIジョブID（記録用） |
| `retriever.py` | 検索・BM25F・RRFの共通関数 |
| `evaluate_bm25f_posboost_full.py` | ③④ 本番評価（5条件） |
| `bm25f_posboost_full_result.json` | 本番結果（集計値） |
| `bm25f_posboost_full_ckpt.json` | 本番結果（トピックごとの上位1000件ランキング） |
| `run_bm25f_posboost_full.log` | 本番実行ログ（途中で最適化版に切り替えて再開した跡を含む） |

### パラメータ決定・比較実験

| スクリプト | 結果 | ログ | 内容 |
|---|---|---|---|
| `search_bm25f_b_3way.py` | `bm25f_b_3way_grid_result.json` | `run_bm25f_b_3way.log` | b の3軸グリッド |
| `evaluate_bm25f_k3_sweep.py` | `bm25f_k3_sweep_result.json` | `run_bm25f_k3.log` 他 | qtf の掛け方 |
| `evaluate_bm25f_k1_length_sweep.py` | `bm25f_k1_length_sweep_result.json` | — | k1 の決定 |
| `evaluate_k1_decay_sweep.py` | `k1_decay_sweep_result.json` | — | k1 の動機付け（バグ下のb、参考） |
| `evaluate_bm25f_norerank.py` | `bm25f_norerank_result.json` | `run_bm25f_norerank.log` | Stage 2 の寄与 |
| `evaluate_bm25f_nopseudo.py` | `bm25f_nopseudo_result.json` | `run_bm25f_nopseudo.log` | 疑似文書なし（qtf未反映） |
| `evaluate_bm25f_flatpseudo.py` | `bm25f_flatpseudo_result.json` | `run_bm25f_flatpseudo.log` | フラット疑似文書（qtf未反映） |
| `evaluate_bm25f_posboost_structure_pilot.py` | `bm25f_posboost_structure_pilot_result.json` + `_ckpt.json` | `run_posboost_structure_pilot_n35.log` | 位置ブーストのパイロット |

### 同梱していないもの（元フォルダ `gemini_fewshot_each_boost/` にある）

- `bm25f_k1_length_sweep_ckpt.json`（133MB）、`bm25f_k3_sweep_ckpt.json`（50MB）：
  スイープのトピックごとのランキング。集計値は結果JSONに入っている。
- avgdlバグ下のBM25F結果一式（`field_strength/bm25f_field_b_search_result*.json` 等）：無効な値。

スクリプト内のパスは、このフォルダ内（`data/` と同階層の生成物）を指すように書き換えてある。
変更したのはパス定義の行だけで、処理内容は元と同じ。
