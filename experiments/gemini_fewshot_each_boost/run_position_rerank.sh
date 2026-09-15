#!/bin/sh
set -u
cd "$(dirname "$0")" || exit 1
echo "[$(date +%H:%M:%S)] position rerank 開始"
/home/takeuchi/venvs/rag/bin/python -u evaluate_position_rerank.py > run_position_rerank.log 2>&1
echo "[$(date +%H:%M:%S)] position rerank 終了 (exit $?)"
