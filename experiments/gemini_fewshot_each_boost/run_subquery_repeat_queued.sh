#!/bin/sh
# 実行中の 2x2 実験（evaluate_termweight_posboost_2x2）が終わるのを待ってから、
# Subquery繰り返しの 2x2 を2構成ぶん回す。
# OpenSearchを取り合って所要時間が歪むのを避けるため並列にはしない。
#   champion    : nDCG@10 基準の現行最良（均等重み + 位置ブースト）
#   idfweighted : recall@1000 基準の現行最良（IDF項重み付け + 位置ブースト）
set -u
PY=/home/takeuchi/venvs/rag/bin/python
cd "$(dirname "$0")" || exit 1

echo "[$(date +%H:%M:%S)] 先行実験(termweight 2x2)の終了を待機中..."
while pgrep -f "evaluate_termweight_posboost_2x2" > /dev/null 2>&1; do
    sleep 30
done
echo "[$(date +%H:%M:%S)] 先行実験の終了を確認"

for R in champion idfweighted; do
    echo "[$(date +%H:%M:%S)] subquery_repeat retrieval=$R 開始"
    "$PY" -u evaluate_subquery_repeat.py --model gemini --retrieval "$R" \
        > "run_subquery_repeat_gemini_$R.log" 2>&1
    echo "[$(date +%H:%M:%S)] subquery_repeat retrieval=$R 終了 (exit $?)"
done
echo "[$(date +%H:%M:%S)] SUBQUERY REPEAT ALL DONE"
