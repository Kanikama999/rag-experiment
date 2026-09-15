#!/bin/sh
# 項重み付け x 位置ブースト の 2x2 を2モデル分まとめて回すドライバ。
# 主実験: gemini-3.7-flash + IDF項重み付け（logprob不要）
# 副実験: gpt-5.6-terra + confidence項重み付け（logprobが取れる唯一の疑似文書セット）
set -u
PY=/home/takeuchi/venvs/rag/bin/python
cd "$(dirname "$0")" || exit 1

echo "[$(date +%H:%M:%S)] gemini + idf 開始"
"$PY" -u evaluate_termweight_posboost_2x2.py --model gemini --weighting idf \
    > run_2x2_gemini_idf.log 2>&1
echo "[$(date +%H:%M:%S)] gemini + idf 終了 (exit $?)"

echo "[$(date +%H:%M:%S)] gpt56terra + confidence 開始"
"$PY" -u evaluate_termweight_posboost_2x2.py --model gpt56terra --weighting confidence \
    > run_2x2_gpt56terra_confidence.log 2>&1
echo "[$(date +%H:%M:%S)] gpt56terra + confidence 終了 (exit $?)"

echo "[$(date +%H:%M:%S)] ALL DONE"
