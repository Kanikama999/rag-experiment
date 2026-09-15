#!/bin/sh
# 先行実行(PIDで待つ)の終了後、long_pos_fielded だけを追加実行して既存結果にマージする。
# pgrep -f はコマンド文字列にパターンを含む自分自身にマッチして誤検知するので、PID で待つ。
set -u
cd "$(dirname "$0")" || exit 1
WAIT_PID="$1"
echo "[$(date +%H:%M:%S)] PID $WAIT_PID の終了を待機中..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] 先行実行の終了を確認"
echo "[$(date +%H:%M:%S)] long_pos_fielded 開始"
/home/takeuchi/venvs/rag/bin/python -u evaluate_posboost_full_ablation.py \
    --only long_pos_fielded > run_posboost_fielded.log 2>&1
echo "[$(date +%H:%M:%S)] long_pos_fielded 終了 (exit $?)"
