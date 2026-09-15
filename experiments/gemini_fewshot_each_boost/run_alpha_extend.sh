#!/bin/sh
# alpha スイープの延長（2, 3, 5）。alpha=1.5 がグリッドの端で最良だったため、
# 単調増加が続くのか飽和・悪化するのかを確かめる。
# 既存の idf_alpha_sweep_result.json を潰さないよう --tag で別ファイルに書く。
set -u
cd "$(dirname "$0")" || exit 1
WAIT_PID="$1"
echo "[$(date +%H:%M:%S)] PID $WAIT_PID の終了を待機中..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "[$(date +%H:%M:%S)] 先行ジョブの終了を確認"
echo "[$(date +%H:%M:%S)] alpha 延長スイープ (2,3,5) 開始"
/home/takeuchi/venvs/rag/bin/python -u evaluate_idf_alpha_sweep.py \
    --alphas 2,3,5 --tag _extend > run_idf_alpha_extend.log 2>&1
echo "[$(date +%H:%M:%S)] alpha 延長スイープ 終了 (exit $?)"
