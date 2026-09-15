"""
evaluate_querylength_confound.py が出力した querylength_confound_per_topic_scores.json
を読み込み、Tier（A: Subquery単体 / B: narrative単体 / C: 既存の長いパイプライン）ごとに
posboostなし/ありの差を、対応のあるbootstrap検定で評価する。

各Tier・各qrelsセット・各指標について:
- 平均差（絶対）
- 相対改善率（(あり-なし)/なし）
- 対応のあるbootstrap（トピックをresampleして平均差の分布を作る、10000回）による
  95%信頼区間とp値（両側、差の分布が0をまたぐ割合から算出）

Tier間で相対改善率がほぼ一定なら、posboostは一般的な手法（FirstP相当）であることが
示唆される。Tierが長いほど改善率が大きく、かつ短いTierで有意でないなら、narrative
特有の効果を主張できる材料になる。

使い方:
    python analyze_querylength_confound.py
"""

import json
import os
import random

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
IN_FILE = os.path.join(RAG_DIR, "querylength_confound_per_topic_scores.json")

N_BOOTSTRAP = 10000
METRIC_KEYS = ["recall_100", "recall_1000", "ndcg_cut_10", "P_100"]
METRIC_LABELS = {"recall_100": "recall@100", "recall_1000": "recall@1000",
                  "ndcg_cut_10": "nDCG@10", "P_100": "precision@100"}
TIER_LABELS = {
    "A_subquery": "Tier A: Subquery単体（短い、平均30語弱）",
    "B_narrative": "Tier B: narrative単体（中程度、平均40語）",
    "C_fullpipeline": "Tier C: 既存の長いパイプライン（300〜400語規模）",
}


def paired_bootstrap(diffs, n_bootstrap=N_BOOTSTRAP, seed=0):
    """diffs: 各トピックの(あり - なし)の差のリスト。
    戻り値: (平均差, 95%CI下限, 95%CI上限, 両側p値)"""
    rng = random.Random(seed)
    n = len(diffs)
    mean_diff = sum(diffs) / n
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_bootstrap)]
    hi = boot_means[int(0.975 * n_bootstrap)]
    # 両側p値: 0がbootstrap分布のどれくらい極端な位置にあるか
    n_le_0 = sum(1 for b in boot_means if b <= 0)
    n_ge_0 = sum(1 for b in boot_means if b >= 0)
    p = 2 * min(n_le_0, n_ge_0) / n_bootstrap
    p = min(p, 1.0)
    return mean_diff, lo, hi, p


def main():
    with open(IN_FILE) as f:
        all_scores = json.load(f)

    for tier_name, tier_scores in all_scores.items():
        print("\n" + "=" * 78)
        print(TIER_LABELS.get(tier_name, tier_name))
        print("=" * 78)
        scores_off = tier_scores["False"]
        scores_on = tier_scores["True"]
        for qrel_name in scores_off.keys():
            print(f"\n[{qrel_name}]")
            off = scores_off[qrel_name]
            on = scores_on[qrel_name]
            common_qids = sorted(set(off.keys()) & set(on.keys()))
            n = len(common_qids)
            header = f"{'指標':14s}{'posboostなし':>14s}{'posboostあり':>14s}{'絶対差':>10s}{'相対改善':>10s}{'95%CI':>22s}{'p値':>10s}"
            print(header)
            print("-" * len(header))
            for key in METRIC_KEYS:
                vals_off = [off[q][key] for q in common_qids]
                vals_on = [on[q][key] for q in common_qids]
                mean_off = sum(vals_off) / n
                mean_on = sum(vals_on) / n
                diffs = [vals_on[i] - vals_off[i] for i in range(n)]
                mean_diff, lo, hi, p = paired_bootstrap(diffs)
                rel = (mean_diff / mean_off * 100) if mean_off > 0 else float("nan")
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                print(f"{METRIC_LABELS[key]:14s}{mean_off:14.4f}{mean_on:14.4f}"
                      f"{mean_diff:+10.4f}{rel:+9.1f}%  [{lo:+.4f}, {hi:+.4f}]{p:9.4f}{sig}")
            print(f"  (n={n}トピック, {N_BOOTSTRAP}回bootstrap, *: p<0.05, **: p<0.01)")


if __name__ == "__main__":
    main()
