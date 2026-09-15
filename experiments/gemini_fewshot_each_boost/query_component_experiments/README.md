# クエリの効き目実験 — まとめ

「narrative・Subquery・疑似参照文書・拡張語リストのうち、どれが・どれだけ性能に効くか」を
調べた実験群。champion本体（`retriever.py`）とは独立に読める形でコピーをまとめてある。

**注意**: ここに置いたのはコピー。元ファイルは `experiments/gemini_fewshot_each_boost/` 直下に
残っており、そちらが正本（他の実験からも参照される）。このディレクトリの `.py` を単体で
再実行する場合は `retriever.py` と `_cache_*.json.gz`（素材別融合キャッシュなど）が
親ディレクトリにある前提なので、親ディレクトリから実行するか `../` を通す必要がある。

## 実験一覧

| # | 実験 | スクリプト | 結果ファイル | 何を調べたか |
|---|---|---|---|---|
| 1 | 成分の寄与（2⁴完全要因） | `evaluate_query_component_grid.py` | `query_component_grid.json`<br>`query_component_found_rels.json` | narrative/Subquery/疑似文書/拡張語リストのon-off16通り×posboost2通り=31条件。単独性能・Shapley値・固有貢献（正解文書の逆引き）が全部これ1本から出る |
| 2 | 素材別RRF融合 | `evaluate_material_fusion.py` | `material_fusion_result.json` | 4素材を1本のクエリに連結せず、別々に検索してRRF融合したらどうなるか |
| 3 | サブクエリ本数 | `evaluate_subquery_count.py` | `subquery_count_result.json` | トピックあたり平均4.3本のSubqueryのうち、何本あれば十分か（全部分集合を列挙） |
| 4 | long縮小（posboostなし） | `evaluate_long_shrink.py` | `long_shrink_result.json` | champion構成でnarrative繰り返し回数・疑似文書body長を削る（位置ブーストなし） |
| 5 | long縮小（posboostあり） | `evaluate_long_shrink_posboost.py` | `long_shrink_posboost_result.json` | 同上、championと同じ位置ブースト込み |
| 6 | recall記録での縮小 | `evaluate_recall_shrink.py` | `recall_shrink_result.json` | 疑似文書+拡張語リスト(both構成)の土台でnarrative繰り返しを削る。narrative×3が最適という発見はここから |
| 7 | Subquery繰り返しスイープ | `evaluate_subquery_repeat_sweep.py` | `subquery_repeat_sweep_result.json` | narrative×3固定でSubqueryの繰り返し回数(0,1,2,3,5)を振る。ほぼ無感応という結果 |
| 8 | long+short融合 | `evaluate_longshort_fusion.py` | `longshort_fusion_result.json` | championの長いクエリと、Subquery単独の短いクエリをRRF融合したらどうなるか |

## 主な結論（詳細は各resultファイル・本体スレッド参照）

- **疑似参照文書は単独最弱・Shapley値が唯一全指標で負・固有貢献も最下位**。3つの独立した指標が
  揃って否定的（①の結果）
- **narrativeの繰り返しは5回が最適ではない**。champion構成(posboostあり)なら5回でも伸び続けるが、
  疑似文書+拡張語リストのboth構成では**3回がピークで5回はむしろ悪化**（6の発見）
- **Subqueryの繰り返し回数はほぼ無感応**（7の発見）。narrativeとは対照的で、既にshort側の
  独立融合チャンネルとしてSubqueryが別枠で効いているためと解釈
- **サブクエリは2〜3本で改善の6〜8割に達し、4本目以降はノイズ水準**（3の発見）
- **素材を連結せず別チャンネルとして融合すると、位置ブーストなしでは上回るが、championの
  位置ブースト込みには届かない**（2・8の発見）

## 検算（サニティチェック）

各スクリプトには、既知の記録値と一致するはずの条件が仕込んである（docstring参照）。
例: `evaluate_subquery_repeat_sweep.py` の `subrep1` は既存recall記録（consensus
R@1000=0.3318）に一致 — 実測値もこの通りで確認済み。
