#!/bin/sh
# 実行中のbm25f_nopseudo（PID 168289）の終了を待ってから、フラット疑似文書実験を回す。
# OpenSearchを取り合って所要時間が歪むのを避けるため並列にはしない。
set -u
PY=/home/takeuchi/venvs/rag/bin/python
DRIVER_PID=168289
cd "$(dirname "$0")" || exit 1

echo "[$(date +%H:%M:%S)] 先行実験(bm25f_nopseudo, PID $DRIVER_PID)の終了を待機中..."
while kill -0 "$DRIVER_PID" 2>/dev/null; do
    sleep 30
done
while pgrep -f "evaluate_bm25f_nopseudo" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] 先行実験の終了を確認"

echo "[$(date +%H:%M:%S)] フラット疑似文書実験 開始（105トピック）"
"$PY" -u evaluate_bm25f_flatpseudo.py > run_bm25f_flatpseudo.log 2>&1
echo "[$(date +%H:%M:%S)] フラット疑似文書実験 終了 (exit $?)"
echo "[$(date +%H:%M:%S)] FLATPSEUDO DONE"
