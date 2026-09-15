# 見出し役割（heading role）実験まとめ（2026-09-09）

> 結論: **シグナルは実在するが、チャンピオンに足しても効果はゼロ**。
> ただし「なぜ効かないか」まで確定できたので、以前のdiscourse marker実験の
> 曖昧な失敗とは違い、記録として残す価値のある negative result になっている。

## 1. 動機

これまでの discourseboost は body 中の談話標識（"in summary" / "in conclusion" 等）の
直後を加点するもので、チャンピオンに足しても効果がほぼ無かった
（consensus nDCG@10 0.5251 → 0.5225）。

着想を変え、**見出しは役割を持つ**（意味→定義、語源→起源、まとめ→要約、注意点→重要 …）
という仮説を検索に持ち込む。着目点は body から headings フィールドへ移る。
役割語彙は9種（definition / detail / origin / qa / summary / caution / comparison /
procedure / cause）を正規表現で定義し、`retriever.py` の `HEADING_ROLE_PATTERNS` に置いた。

要点は、**同じ役割語彙を文書側とクエリ側の両方に当てる**こと。

- 文書側: headings フィールドを改行で1本ずつに割り、各見出しに当てる → その見出しの役割
- クエリ側: Subquery の文字列に当てる → その Subquery の意図

Webの見出しも分解後のSubqueryも `What is X` / `Why does X ...` のような同じ形をしている
ので、1つの語彙で両方さばける。

## 2. 事前分析

### 2.1 無条件の役割保有率には識別力が無い（`measure_heading_role_distribution.py`）

「その文書が役割見出しを持つか」をクエリと無関係に測ると、relevant と
judged-nonrelevant がほぼ同じだった（summary: 10.55% vs 10.53%、caution に至っては
非relevantの方が高い 16.10% vs 12.42%）。コーパス平均との差は「判定対象になった
＝そのトピックの文書」という効果にすぎない。**役割を無条件に加点する設計は成立しない。**

### 2.2 話題一致した見出しの役割別で見ると差が出る（`measure_heading_role_conditional.py`）

見出し1本ごとに「Subqueryの内容語が2語以上入っているか（＝その見出しはクエリ話題か）」と
役割ラベルを出し、relevant / 非relevant で比較（consensus 105トピック、各15,750文書。
負例は narrative そのままのBM25上位から抽出）。

| signal | relevant | nonrel | lift |
|---|---|---|---|
| any（役割問わず話題一致見出し） | 41.23% | 24.71% | 1.668 |
| definition | 3.86% | 1.16% | **3.341** |
| cause | 4.56% | 1.45% | **3.149** |
| procedure | 1.42% | 0.51% | 2.765 |
| comparison | 1.42% | 0.53% | 2.687 |
| caution | 1.54% | 0.77% | 1.984 |
| **summary** | 0.34% | 0.16% | **2.120** |
| detail | 2.28% | 1.19% | 1.910 |
| qa | 0.30% | 0.17% | 1.778 |
| origin | 1.26% | 0.61% | 2.062 |

**以前の discourseboost が使っていた要約系マーカーが、役割の中で最も弱い部類**だった。
coverage 22トピックでは summary の lift は 1.192 で、役割非依存の 1.859 すら下回る。
discourseboost が効かなかった理由の説明になっている（マーカーの選び方が悪かった）。

### 2.3 意図と役割の対応（alignment）が最も強い（`measure_heading_role_alignment.py`）

同じ語彙を Subquery 側にも当てて意図ラベルを付け、3条件に分けた。

| signal | relevant | nonrel | lift (consensus) | lift (coverage) |
|---|---|---|---|---|
| plain（話題一致だが役割語なし） | 36.29% | 22.88% | 1.586 | 1.840 |
| crossed（役割はあるが意図と不一致） | 11.63% | 4.54% | 2.561 | 2.201 |
| **aligned（役割が意図と一致）** | 3.83% | 0.78% | **4.902** | **3.058** |

plain < crossed < aligned と単調。仮説はここまでは支持されている。

ただし弱点が2つある。意図ラベルが付く Subquery は451本中220本（48.8%）だけで、
内訳も definition 38.8% と cause 11.5% にほぼ集中している（detail / qa / summary /
procedure はそれぞれ2本以下）。つまり aligned の数字は実質この2役割で稼いでいる。

## 3. 検索側の実装と結果 — いずれもチャンピオンを超えなかった

チャンピオン = `bm25_equalweight_posboost_headingroleboost(role_markers=[])`
（title/headings/body均等 + body先頭100語へのposboost、span_boost=15）。
これは既存の `webstyle_narrative_equalweight_posboost_only` と同一クエリで、
35トピックでの再現値 nDCG@10=0.5007 が既存記録 0.5007 と一致することを確認済み。

### 3.1 span_near 加点（`search_headingrole_params.py`）

headings フィールド上で「意図に一致する役割語 × クエリ語」が slop 以内に近接したら加点。
consensus・18トピック（stride=6）:

| 条件 | recall@1000 | nDCG@10 |
|---|---|---|
| champion（役割ブースト無し） | 0.2653 | **0.4999** |
| aligned slop=15 boost=2.0 | 0.2656 | 0.4958 |
| aligned slop=15 boost=5.0 | 0.2644 | 0.4938 |
| aligned slop=8 boost=5.0 | 0.2647 | 0.4929 |
| aligned slop=4 boost=5.0 | 0.2652 | 0.4924 |
| aligned slop=15 boost=10.0 | 0.2593 | 0.4915 |

boost に対して単調に悪化し、boost→0 でチャンピオンに戻る。純粋に害。

### 3.2 再ランキング（`evaluate_headingrole_rerank.py`）

3.1 の失敗には条件のずれという説明がつく。分析側は「同一見出しの中に内容語2語以上」を
要求していたのに対し、検索式は「headings のどこかで役割語とクエリ語1語が近接」しか
要求しておらず、はるかに緩い。しかも headings への `match` 節と重複している。

そこで分析側とまったく同じ条件をチャンピオン上位100件に対してクライアント側で計算し、
RRFスコアへ加点した（`score' = rrf + alpha * feature`）。consensus・35トピック:

| feature | 最良のalphaでのnDCG@10 | champion差 |
|---|---|---|
| champion (alpha=0) | 0.5007 | — |
| aligned | 0.5007 (alpha<=3e-5) | ±0.0000 |
| crossed | 0.5007 | ±0.0000 |
| plain | 0.5028 (alpha=3e-4) | +0.0021 |
| any | 0.5028 (alpha=3e-4) | +0.0021 |

alpha を上げると全 feature が単調に悪化する。**aligned はどの alpha でもチャンピオンを
超えない**（+0.0021 は plain/any 側に出ており、役割とは無関係のノイズ）。

なお最初 alpha ∈ {0.001 … 0.05} で掃引して全滅したが、これはシグナルが無いのではなく
刻みが粗すぎたため。RRFスコアは 1/(60+rank) の和なので上位付近の隣接ランク差は 1e-4 程度で、
alpha=0.001 は既に約10ランク分の移動に相当していた。細かい格子で引き直した結果が上表で、
**効果はきれいにゼロ**というのが正しい結論。

## 4. なぜ効かなかったか（`measure_heading_role_within_champion.py`）

2.3 の lift 4.90 は、負例を **narrative そのままの BM25 上位**から取って測ったもの。
チャンピオンはそれよりはるかに強いランカーなので、その lift の大半は
**チャンピオンが既に捉えている情報**である可能性が高い。負例をチャンピオン上位100件の
非relevant文書に置き換えて測り直すと:

| signal | relevant | nonrel | lift（BM25負例のとき） |
|---|---|---|---|
| any | 91.51% | 88.85% | 1.030（← 1.668） |
| plain | 84.58% | 83.95% | 1.007（← 1.586） |
| crossed | 35.43% | 27.94% | 1.268（← 2.561） |
| **aligned** | 14.31% | 8.95% | **1.599**（← 4.902） |

（consensus 35トピック、pos n=2684 / neg n=816）

plain と any は 1.0 まで崩壊する＝チャンピオンに完全に吸収されている。
一方 **aligned だけは lift 1.599 を保っており、残余情報は確かに存在する**。
ただし絶対量が足りない。aligned を持つ文書の relevant 率は 84.0% で、
この母集団の基準率 76.7% に対し +7ポイントしかなく、しかも relevant の 14.31% しか
カバーしない。加点1本でこの差を nDCG@10 に変換するには弱すぎる、というのが実態。

## 5. 教訓

- **シグナルの lift は、比較する負例の強さでいくらでも変わる。**
  手法を実装する前に、`baseline` ではなく **その時点のチャンピオンが実際に取り違えている
  文書**を負例にして lift を測るべきだった。今回それをやっていれば、4.90 ではなく 1.60 が
  最初から見えていて、実装コストを払う前に見送れた。
- 以前の discourseboost の失敗も、同じ測定をしていれば予測できた可能性が高い。
- 逆に言えば、2.2 の「要約系マーカーは役割の中で最弱」は、discourse marker 系の
  アプローチを今後も要約系で続けても筋が悪い、という判断材料として残る。

## 6. 残っている可能性

- **リランカー導入時の素性として。** aligned は上位100件内でも lift 1.599 を保っている。
  加点1本では弱すぎるが、学習型リランカーの素性の1本としてなら意味がある可能性がある。
- **生成側での役割対応。** 今回は実文書の見出しにだけ役割を付けたが、疑似文書生成の
  プロンプトを変えて LLM に役割ラベル付きの headings を出させ、役割対役割で
  マッチさせる設計は未検証。今回の実験はこれを否定していない。
- **意図ラベルのLLM化。** 正規表現では Subquery の 48.8% にしか意図が付かず、
  内訳も definition/cause に偏っている。LLM に振ればカバー率と役割分布は広がるが、
  4節の残余 lift 1.599 が小さい以上、これだけで結論が覆る見込みは薄い。

## 7. ファイル

| ファイル | 役割 |
|---|---|
| `retriever.py` | `HEADING_ROLE_PATTERNS` / `HEADING_ROLE_MARKERS` / `heading_roles()` / `role_markers_for()` / `bm25_equalweight_posboost_headingroleboost()` |
| `measure_heading_role_distribution.py` | 2.1 無条件の役割保有率（→ `heading_role_distribution_result.json`） |
| `measure_heading_role_conditional.py` | 2.2 話題一致見出しの役割別 lift（→ `heading_role_conditional_{coverage,consensus}.json`） |
| `measure_heading_role_alignment.py` | 2.3 意図×役割の alignment（→ `heading_role_alignment_{coverage,consensus}.json`） |
| `search_headingrole_params.py` | 3.1 span_near加点のパラメータ探索（→ `headingrole_param_search_result.json`） |
| `evaluate_headingrole_rerank.py` | 3.2 再ランキング版の alpha 掃引（→ `headingrole_rerank_eval_summary_rep5.json`） |
| `measure_heading_role_within_champion.py` | 4 チャンピオン上位内での lift 再測定（→ `heading_role_within_champion.json`） |
| `evaluate_decomposed_webstyle_headingrole.py` | 105トピック版の評価スクリプト（**未実行**。3.1/3.2 が両方ゼロだったため走らせていない） |
| `heading_role_lexicon_df.json` | 役割語のheadingsフィールド文書頻度（マーカー選定の根拠） |
| `_cache_champion_run_rep5.json` | チャンピオンの融合済みランキングのキャッシュ（35トピック） |
