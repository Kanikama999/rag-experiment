#!/bin/sh
# avgdl バグ修正後の BM25F 再測定。b のグリッドサーチ → k3 スイープ の順。
# b の最適値がバグ下で選ばれていたため、b から測り直す必要がある。
set -u
cd "$(dirname "$0")" || exit 1
PY=/home/takeuchi/venvs/rag/bin/python
echo "[$(date +%H:%M:%S)] (1) b グリッドサーチ 開始"
"$PY" -u search_bm25f_field_b.py > run_bm25f_field_b_refix.log 2>&1
echo "[$(date +%H:%M:%S)] (1) b グリッドサーチ 終了 (exit $?)"
echo "[$(date +%H:%M:%S)] ALL DONE"
