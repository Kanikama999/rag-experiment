#!/bin/sh
# 多段窓・リング窓の13条件を回す。先行ジョブ(PID指定)の終了を待ってから開始する。
# pgrep -f はコマンド文字列にパターンを含む自分自身へ誤マッチするので、必ずPIDで待つ。
set -u
cd "$(dirname "$0")" || exit 1
WAIT_PID="$1"
echo "[$(date +%H:%M:%S)] PID $WAIT_PID の終了を待機中..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] 先行ジョブの終了を確認"
echo "[$(date +%H:%M:%S)] multiwindow posboost 開始"
/home/takeuchi/venvs/rag/bin/python -u evaluate_multiwindow_posboost.py \
    > run_multiwindow.log 2>&1
echo "[$(date +%H:%M:%S)] multiwindow posboost 終了 (exit $?)"
