#!/bin/sh
# 実行中の subquery_repeat ドライバ（run_subquery_repeat_all.sh, PID 42245）の終了を
# 待ってから、Subquery重み付けシグナルの診断を回す。
# OpenSearchを取り合って所要時間が歪むのを避けるため並列にはしない。
set -u
PY=/home/takeuchi/venvs/rag/bin/python
DRIVER_PID=42245
cd "$(dirname "$0")" || exit 1

echo "[$(date +%H:%M:%S)] 先行実験(subquery_repeat, PID $DRIVER_PID)の終了を待機中..."
while kill -0 "$DRIVER_PID" 2>/dev/null; do
    sleep 30
done
# ドライバ終了後、子プロセスが残っていないことも念のため確認する
while pgrep -f "evaluate_subquery_repeat" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] 先行実験の終了を確認"

echo "[$(date +%H:%M:%S)] 重み付けシグナル診断 開始（105トピック）"
"$PY" -u diagnose_subquery_weight_signal.py --model gemini \
    > run_weight_diagnosis_gemini.log 2>&1
echo "[$(date +%H:%M:%S)] 重み付けシグナル診断 終了 (exit $?)"
echo "[$(date +%H:%M:%S)] WEIGHT DIAGNOSIS DONE"
