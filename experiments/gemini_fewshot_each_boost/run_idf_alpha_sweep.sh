#!/bin/sh
# 先行ジョブ(PIDで待つ)の終了後、IDF項重み付けの alpha スイープを回す。
# pgrep -f はコマンド文字列にパターンを含む自分自身に誤マッチするので、必ずPIDで待つ。
set -u
cd "$(dirname "$0")" || exit 1
WAIT_PID="$1"
echo "[$(date +%H:%M:%S)] PID $WAIT_PID の終了を待機中..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] 先行ジョブの終了を確認"
echo "[$(date +%H:%M:%S)] idf alpha sweep 開始"
/home/takeuchi/venvs/rag/bin/python -u evaluate_idf_alpha_sweep.py \
    > run_idf_alpha_sweep.log 2>&1
echo "[$(date +%H:%M:%S)] idf alpha sweep 終了 (exit $?)"
