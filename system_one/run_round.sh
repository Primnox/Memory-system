#!/usr/bin/env bash
# One training round of the memory head, on this machine's GPU, end to end.
#
#   bash run_round.sh r5 [r4]
#
# 1. training rows from every data/train_*.json (disputed labels skipped)
# 2. fine-tune Laya on the GPU (AMD ROCm build in .venv-rocm), Laya's own recipe
# 3. export to ONNX and patch it for DirectML (onnx_dml.py)
# 4. score once, each on data the round never trained or tuned on:
#      regression cases (and against the previous round), fresh Dialogue NLI,
#      LongMemEval knowledge-update pairs, LongMemEval end to end through the app,
#      blind v2 through the app (v3 is training data, not a test)
# Test misses are never printed.
set -euo pipefail
TAG=${1:?round tag, e.g. r5}
PREV=${2:-r4}
HERE=$(cd "$(dirname "$0")" && pwd)
APP=C:/project/.worktrees/memory
APPPY=C:/project/backend/venv/Scripts/python.exe
ROCM=$HERE/.venv-rocm/Scripts/python.exe
CPU=$HERE/.venv/Scripts/python.exe
MODEL=$HERE/models/laya-memory-$TAG
export PYTHONIOENCODING=utf-8
cd "$HERE"

echo "== 1. rows"
"$CPU" build_train.py
echo "== 2. train ($TAG) on the GPU"
# EXTRA_ROWS: more row files, e.g. data/dnli_rows_1500.jsonl
"$ROCM" train_laya.py --device cuda --rows data/train_rows.jsonl ${EXTRA_ROWS:-} --soft-ratio 2 --epochs 3 \
    --micro-batch 8 --grad-accum 4 --output-dir "$MODEL"
for leftover in train_items.pt checkpoint_latest; do rm -rf "$MODEL/$leftover"; done
echo "== 3. ONNX for DirectML"
"$CPU" laya_export_onnx_upstream.py --model "$MODEL" --output "$MODEL/raw.onnx"
"$CPU" onnx_dml.py --patch "$MODEL/raw.onnx" --onnx "$MODEL/laya-memory-$TAG.dml.onnx"
rm -f "$MODEL/raw.onnx" "$MODEL/raw.onnx.data"
ONNX="$MODEL/laya-memory-$TAG.dml.onnx"

echo "== 4a. regression cases"
"$CPU" regression.py --model "$MODEL" --onnx "$ONNX" --tag "$TAG" --against "$PREV" | tail -3 || true
echo "== 4b. fresh Dialogue NLI + LongMemEval knowledge-update pairs"
"$CPU" external_eval.py --model "$MODEL" --onnx "$ONNX" --tag "$TAG" \
    --files data/blind_external/pairs_dnli_fresh.jsonl data/blind_external/pairs_lme_ku.jsonl | tail -3
echo "== 4c. LongMemEval end to end"
"$CPU" blind_pairs.py --data data/lme_e2e/lme_e2e.json --model "$MODEL" --onnx "$ONNX" --tag "$TAG" --save-pairs | tail -1
"$APPPY" lme_e2e.py score --backend "$APP/backend" --decisions "results/blind_lme_e2e_$TAG.json" --tag "head-$TAG" | grep top-1
for SPLIT in v2; do   # v3 is training data now (its README): never score on it
  [ -f "$APP/scripts/blind_memory/$SPLIT/test.json" ] || continue
  echo "== 4d. blind $SPLIT through the app"
  (cd "$APP" && PRIMNOX2_MEMORY_HEAD_SUPERVISE=0 "$APPPY" scripts/bench_memory_blind.py --split test \
      --data "scripts/blind_memory/$SPLIT" --query oracle --head "$MODEL" \
      --json "scripts/blind_memory/results/test_$SPLIT/mem-head-$TAG.json" 2>&1 \
      | grep -aE "^answerable|recall|precision|retired something|memory head")
done
