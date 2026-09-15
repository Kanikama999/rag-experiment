#!/bin/sh
# オラクルを現行パイプラインで測り直す（22トピック、7条件）
set -u
cd "$(dirname "$0")" || exit 1
echo "[$(date +%H:%M:%S)] oracle current pipeline 開始"
/home/takeuchi/venvs/rag/bin/python -u evaluate_oracle_current_pipeline.py \
    > run_oracle_current.log 2>&1
echo "[$(date +%H:%M:%S)] oracle current pipeline 終了 (exit $?)"
