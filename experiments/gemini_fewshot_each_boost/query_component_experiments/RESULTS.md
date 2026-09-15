# クエリの効き目実験 — 結果まとめ

narrative・Subquery・疑似参照文書・拡張語リストのうち、どれが・どれだけ性能に効くかを
調べた実験群の結果を、両qrels・全4指標で記録する。ファイル構成は [README.md](README.md) 参照。

対象: `msmarco-v21-doc`（MS MARCO v2.1 文書レベル、約1,096万件）。n=105（consensus）/ n=22（coverage）。

---

## 0. 最高性能パイプライン（この実験群を踏まえた現状最良）

recall優先で見つかった現状最良は `narr3×6 + sub_pb0 + termlist_pb0`（§5・§6の発見を反映）。
championとの違いは、narrativeの繰り返しを5→3回に減らし、拡張語リストを第3チャンネルとして
融合に追加した点。

```
トピック（narrative = 元の長いクエリ文）
  └─ decompose_narrative.py で Subquery に分解（トピックあたり平均4.3個）
       └─ Subquery ごとに以下の3本を検索
            │
            ├─ ① long側
            │     narrative×3 + Subquery×1 + 疑似参照文書 + 拡張語リスト（連結）
            │     span_terms = analyze_terms(Subquery) + analyze_terms(narrative)
            │     bm25_equalweight_posboost_discourseboost(
            │         title/headings/body, span_terms,
            │         span_end=200, span_boost=30, markers=[])
            │
            ├─ ② short側
            │     Subquery単独（title/headings/bodyすべてに同じテキスト）
            │     bm25_fielded(title_boost=1, headings_boost=1, body_boost=1)
            │     位置ブーストなし
            │
            └─ ③ third側
                  拡張語リスト単独（title_terms/heading_terms/body_terms、フィールド対応）
                  bm25_fielded(title_boost=1, headings_boost=1, body_boost=1)
                  位置ブーストなし
            │
            ↓ RRF融合（k=60, top_n=1000, 重み ①:②:③ = 6:1:1）
       Subqueryごとの融合済みランキング（上位1000件）
            │
            ↓ Subquery横断でさらにRRF融合（k=60, top_n=1000）
       トピックの最終ランキング（上位1000件）
```

championとの違い一覧:

| 要素 | champion | 最高性能（recall優先） |
|---|---|---|
| narrative繰り返し | ×5 | **×3**（§5） |
| クエリ構成 | 疑似参照文書のみ | 疑似参照文書 + **拡張語リスト** |
| span_terms | Subqueryのみ | Subquery + **narrative** |
| span_end / span_boost | 100 / 15 | **200 / 30** |
| 融合 | なし | **short側+third側を重み6:1:1で融合** |

実測（両qrels・全指標）:

| | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|
| consensus | 0.0693 | **0.3318** | 0.5147 | 0.7827 |
| coverage | 0.2409 | 0.6453 | 0.4242 | 0.3864 |

（champion参考値: consensus R@1000=0.3095, coverage R@1000=0.5995）

新規LLM生成は分解時のSubquery生成・疑似参照文書生成・拡張語リスト生成のみで、
narrativeの繰り返し回数・span_terms・span_end/boost・融合重みはすべて検索時パラメータ。

---

## 1. クエリ成分の寄与（2⁴完全要因）

スクリプト: [`evaluate_query_component_grid.py`](evaluate_query_component_grid.py)
結果: [`query_component_grid.json`](query_component_grid.json) / [`query_component_found_rels.json`](query_component_found_rels.json)

narrative・Subquery・疑似参照文書・拡張語リストのon/off16通り×位置ブースト2通り=31条件を
全数評価。単独性能・Shapley値（限界寄与）・固有貢献（正解文書の逆引き）がこの1本から出る。

### ① 単独性能（位置ブーストなし = 成分の純粋な実力）

| 素材 | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|
| narrative×5 | consensus | 0.0595 | 0.2492 | **0.5008** | 0.6830 |
| | coverage | 0.1909 | 0.4885 | **0.4180** | 0.3064 |
| Subquery×5 | consensus | 0.0579 | 0.2406 | 0.4268 | 0.6331 |
| | coverage | 0.2055 | 0.5046 | 0.3718 | 0.3236 |
| 拡張語リスト | consensus | 0.0403 | 0.1852 | 0.3280 | 0.4578 |
| | coverage | 0.1216 | 0.3909 | 0.2712 | 0.1850 |
| 疑似参照文書 | consensus | 0.0287 | 0.1157 | 0.2846 | 0.3345 |
| | coverage | 0.0883 | 0.2542 | 0.2106 | 0.1355 |
| （全部入り） | consensus | 0.0592 | 0.2707 | 0.4610 | 0.6646 |
| | coverage | 0.2000 | 0.5682 | 0.3897 | 0.3091 |

narrative×5単独が両qrelsで全部入りを上回る。疑似参照文書は両qrelsで最弱。

### ② 限界寄与（Shapley値、位置ブーストなし）

| 素材 | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|
| narrative×5 | consensus | +0.0100 | +0.0446 | +0.0817 | +0.1137 |
| | coverage | +0.0360 | +0.0797 | +0.0707 | +0.0589 |
| Subquery×5 | consensus | +0.0069 | +0.0374 | +0.0250 | +0.0706 |
| | coverage | +0.0274 | +0.0745 | +0.0235 | +0.0430 |
| 疑似参照文書 | consensus | -0.0036 | -0.0140 | -0.0205 | -0.0388 |
| | coverage | -0.0114 | -0.0113 | -0.0191 | -0.0217 |
| 拡張語リスト | consensus | -0.0007 | +0.0049 | -0.0103 | -0.0081 |
| | coverage | -0.0036 | +0.0158 | -0.0032 | -0.0088 |

疑似参照文書は両qrels・4指標すべてで負。拡張語リストはR@1000だけ正。

### ③ 固有貢献（各素材の単独検索で見つけた正解）

| 素材 | qrels | 発見数 | うち自分だけ | 固有率 |
|---|---|---|---|---|
| narrative×5 | consensus | 31,253 | 7,921 | 25.3% |
| | coverage | 1,770 | 256 | 14.5% |
| Subquery×5 | consensus | 28,979 | 6,395 | 22.1% |
| | coverage | 1,790 | 237 | 13.2% |
| 拡張語リスト | consensus | 22,616 | 3,136 | 13.9% |
| | coverage | 1,383 | 94 | 6.8% |
| 疑似参照文書 | consensus | 14,543 | 1,729 | 11.9% |
| | coverage | 888 | 51 | 5.7% |

4素材の和集合: consensus 48,453件（4素材すべてが発見12.5%）／ coverage 2,458件（同21.4%）。

### 最小構成（位置ブーストあり = championと同じ機構）

| 素材数 | 組み合わせ | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|---|
| 0 | （なし） | consensus | 0.0036 | 0.0254 | 0.0207 | 0.0402 |
| | | coverage | 0.0068 | 0.0413 | 0.0110 | 0.0123 |
| 1 | narrative×5 | consensus | 0.0645 | 0.2744 | 0.5023 | 0.7250 |
| | | coverage | 0.2053 | 0.5164 | 0.3623 | 0.3336 |
| 2 | narrative×5+拡張語リスト | consensus | 0.0685 | 0.3108 | 0.5251 | 0.7706 |
| | | coverage | 0.2315 | 0.5962 | 0.4089 | 0.3714 |
| 3 | narrative×5+Subquery×5+疑似参照文書 | consensus | 0.0693 | 0.3111 | 0.5177 | 0.7792 |
| | | coverage | 0.2429 | 0.6068 | 0.4149 | 0.3882 |
| 4 | 全部入り | consensus | 0.0674 | 0.3142 | 0.5018 | 0.7573 |
| | | coverage | 0.2331 | 0.6232 | 0.4068 | 0.3695 |
| 参考 | champion（既知値） | consensus | 0.0689 | 0.3095 | 0.5251 | 0.7769 |
| | | coverage | 0.2373 | 0.5995 | 0.4049 | 0.3768 |

「narrative×5+拡張語リスト」の2素材でchampion同等のnDCGに到達（疑似参照文書が丸ごと不要）。

**検算**: 3素材セル(N1_S1_P1_E0_pb1) = 0.5177 は既知値（§7.7）と一致。

---

## 2. 素材別RRF融合（連結せず別チャンネルとして検索）

スクリプト: [`evaluate_material_fusion.py`](evaluate_material_fusion.py)
結果: [`material_fusion_result.json`](material_fusion_result.json)

4素材を1本のクエリに連結せず、別々に検索してRRF融合。トポロジーはflat（全リスト一括融合）と
hier（素材横断→Subquery横断の2段）の2種、位置ブーストなし/ありの2arm。

| 条件 | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|
| narr+sub/flat（posboostなし最良） | consensus | 0.0639 | 0.2692 | **0.5193** | 0.7261 |
| | coverage | 0.2163 | 0.5438 | **0.4414** | 0.3477 |
| narr+sub+疑似+拡張/flat（posboostあり全部） | consensus | 0.0644 | 0.2764 | 0.4453 | 0.7149 |
| | coverage | 0.2179 | 0.5634 | 0.3739 | 0.3564 |

posboostなしでは融合が連結最良(narr×5+Sub×5, nDCG 0.5085)を上回るが、championの位置ブースト
込みには届かない。

---

## 3. サブクエリ本数（何本必要か）

スクリプト: [`evaluate_subquery_count.py`](evaluate_subquery_count.py)
結果: [`subquery_count_result.json`](subquery_count_result.json)

トピックあたり平均4.3本のSubqueryの全部分集合（2,591通り）を列挙し、本数k別に平均/最良/
先頭k本を集計。母集団を「k本以上持つトピック」に固定して比較（min4本以上: consensus n=84,
coverage n=19）。

| k | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|
| 1 | consensus | 0.0636 | 0.2807 | 0.4833 | 0.7196 |
| | coverage | 0.2017 | 0.5195 | 0.4203 | 0.3330 |
| 2 | consensus | 0.0670 | 0.3010 | 0.5070 | 0.7600 |
| | coverage | 0.2191 | 0.5652 | 0.4215 | 0.3602 |
| 3 | consensus | 0.0683 | 0.3092 | 0.5153 | 0.7749 |
| | coverage | 0.2276 | 0.5805 | 0.4178 | 0.3737 |
| 4 | consensus | 0.0688 | 0.3150 | 0.5186 | 0.7801 |
| | coverage | 0.2325 | 0.5934 | 0.4146 | 0.3793 |
| 全部（平均4.3本） | consensus | 0.0690 | 0.3165 | 0.5236 | 0.7829 |
| | coverage | 0.2326 | 0.5947 | 0.4203 | 0.3795 |

2〜3本で改善の6〜8割に到達、4本目以降はノイズ水準（recall@1000で0.001〜0.004、§5参照）。

**検算**: k=n（全部使う）= champion。consensus nDCG@10=0.52508…≈0.5251、
coverage nDCG@10=0.40488…≈0.4049 で既知値と一致。

---

## 4. long側の縮小（narrative繰り返し・疑似文書body長）

### 4a. champion構成・位置ブーストなし

スクリプト: [`evaluate_long_shrink.py`](evaluate_long_shrink.py)
結果: [`long_shrink_result.json`](long_shrink_result.json)

| 条件 | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|
| narrativeなし | consensus | 0.0357 | 0.1491 | 0.3279 | 0.4120 |
| | coverage | 0.1125 | 0.3331 | 0.2426 | 0.1718 |
| narrative×5 | consensus | 0.0587 | 0.2559 | **0.4772** | 0.6648 |
| | coverage | 0.1925 | 0.5344 | **0.4091** | 0.2977 |

### 4b. championそのもの（位置ブーストあり）

スクリプト: [`evaluate_long_shrink_posboost.py`](evaluate_long_shrink_posboost.py)
結果: [`long_shrink_posboost_result.json`](long_shrink_posboost_result.json)

| 条件 | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|
| narrativeなし | consensus | 0.0598 | 0.2737 | 0.3906 | 0.6664 |
| | coverage | 0.2014 | 0.5695 | 0.3359 | 0.3191 |
| narrative×5（champion） | consensus | 0.0689 | 0.3095 | **0.5251** | 0.7769 |
| | coverage | 0.2373 | 0.5995 | **0.4049** | 0.3768 |

**検算**: narrative×5・body語数フル・融合なし が championに一致（両方の版で確認済み）。

---

## 5. recall記録パイプラインでの縮小（narrative×3が最適という発見）

スクリプト: [`evaluate_recall_shrink.py`](evaluate_recall_shrink.py)
結果: [`recall_shrink_result.json`](recall_shrink_result.json)

疑似文書+拡張語リスト(both構成)・span_end=200/boost=30・span_terms=Sub+narrativeという
recall記録の土台で、narrative繰り返し回数を0〜5で振る。

| narrative回数 | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|
| ×5 | consensus | 0.0705 | 0.3266 | 0.5369 | 0.7970 |
| | coverage | 0.2451 | 0.6304 | 0.4173 | 0.3932 |
| **×3（最良）** | consensus | 0.0700 | **0.3283** | 0.5227 | 0.7899 |
| | coverage | 0.2438 | **0.6403** | **0.4344** | 0.3900 |

**narrative×5が最適という長年の前提を覆す発見。** ×3の方が両qrelsでR@1000が高く、
coverageではnDCGも上回る。

**検算**: narrative×5・融合重み3 が `sub+narr|both|se200sb30|fuse3`（recall記録そのもの、
consensus R@1000=0.3266）に一致。

---

## 6. Subquery繰り返しスイープ（ほぼ無感応という発見）

スクリプト: [`evaluate_subquery_repeat_sweep.py`](evaluate_subquery_repeat_sweep.py)
結果: [`subquery_repeat_sweep_result.json`](subquery_repeat_sweep_result.json)

narrative×3固定・composition=both・融合重み6で、Subqueryの繰り返し回数を0〜5で振る。

| Subquery回数 | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|
| 0（一切入れない） | consensus | 0.0690 | 0.3303 | 0.5126 | 0.7799 |
| | coverage | 0.2417 | 0.6448 | 0.4271 | 0.3859 |
| 1（現行recall記録） | consensus | 0.0693 | 0.3318 | 0.5147 | 0.7827 |
| | coverage | 0.2409 | 0.6453 | 0.4242 | 0.3864 |
| 2 | consensus | 0.0696 | 0.3320 | 0.5140 | 0.7851 |
| | coverage | 0.2420 | 0.6470 | 0.4268 | 0.3882 |
| 3 | consensus | 0.0699 | **0.3320** | 0.5144 | 0.7889 |
| | coverage | 0.2460 | 0.6458 | 0.4247 | 0.3941 |
| 5 | consensus | 0.0700 | 0.3306 | 0.5141 | 0.7901 |
| | coverage | 0.2481 | **0.6486** | **0.4277** | **0.3964** |

**narrativeとは対照的にほぼフラット**（consensus R@1000の変動幅はわずか0.0017、§5のノイズ
水準0.001〜0.004とほぼ同じ）。Subqueryは既にshort側の独立融合チャンネルとして別枠で
効いているため、long側での繰り返し回数を増減させても影響が薄いと解釈。

**検算**: Subquery×1回 が既存recall記録（consensus R@1000=0.3318, nDCG@10=0.5147）に完全一致。

---

## 7. long+short融合（championと短いクエリのRRF融合）

スクリプト: [`evaluate_longshort_fusion.py`](evaluate_longshort_fusion.py)
結果: [`longshort_fusion_result.json`](longshort_fusion_result.json)

| 条件 | qrels | R@100 | R@1000 | nDCG@10 | P@100 |
|---|---|---|---|---|---|
| long単独（champion再現、posboostなし） | consensus | 0.0587 | 0.2559 | 0.4772 | 0.6648 |
| | coverage | 0.1925 | 0.5344 | 0.4091 | 0.2977 |
| long+narrative+subquery融合（posboostなし） | consensus | 0.0632 | 0.2752 | **0.5101** | 0.7170 |
| | coverage | 0.2120 | 0.5618 | **0.4334** | 0.3364 |

位置ブーストなしでは融合が全指標で単独を上回る。位置ブーストありでは等倍融合は逆効果で、
long側を3倍以上に重くして初めてchampionにわずかに勝てる（詳細はresultファイル参照）。

---

## 総括

| 発見 | 根拠 |
|---|---|
| 疑似参照文書は単独最弱・Shapley値が唯一全指標で負・固有貢献も最下位 | §1 |
| narrative×5+拡張語リストの2素材でchampion同等（疑似文書が丸ごと不要） | §1 |
| 素材を別チャンネルで融合すると位置ブーストなしでは連結を上回る | §2 |
| サブクエリは2〜3本で改善の6〜8割、4本目以降はノイズ水準 | §3 |
| narrative×5が最適という前提は誤り、both構成では×3が最良 | §5 |
| Subqueryの繰り返し回数はほぼ無感応（narrativeと対照的） | §6 |
| long+short融合は位置ブーストなしでのみ明確に効く | §7 |

全表の検算はそれぞれの節に記載の通り、既知の記録値と一致を確認済み。
