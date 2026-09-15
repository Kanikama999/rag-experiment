#!/bin/sh
# BM25F の k3 スイープ（9条件、105トピック）
set -u
cd "$(dirname "$0")" || exit 1
echo "[$(date +%H:%M:%S)] bm25f k3 sweep 開始"
/home/takeuchi/venvs/rag/bin/python -u evaluate_bm25f_k3_sweep.py > run_bm25f_k3.log 2>&1
echo "[$(date +%H:%M:%S)] bm25f k3 sweep 終了 (exit $?)"
