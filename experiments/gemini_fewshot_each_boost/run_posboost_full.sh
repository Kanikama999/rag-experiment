#!/bin/sh
# 位置ブーストのフル実験（105トピック）。発表資料で35トピックに間引いていた内訳を揃える。
set -u
cd "$(dirname "$0")" || exit 1
echo "[$(date +%H:%M:%S)] posboost full ablation 開始"
/home/takeuchi/venvs/rag/bin/python -u evaluate_posboost_full_ablation.py \
    > run_posboost_full.log 2>&1
echo "[$(date +%H:%M:%S)] posboost full ablation 終了 (exit $?)"
