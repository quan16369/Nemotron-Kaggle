#!/bin/bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-/kaggle/working}"
ART_DIR="${ART_DIR:-$WORK_DIR/artifacts}"
DATA_DIR="$ART_DIR/data"
MODEL_NAME="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
MERGED_DIR="${MERGED_DIR:-$WORK_DIR/merged_adapter}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
POPULATION_SIZE="${POPULATION_SIZE:-6}"
GENERATIONS="${GENERATIONS:-4}"
ELITE_SIZE="${ELITE_SIZE:-2}"
MUTATION_SCALE="${MUTATION_SCALE:-0.15}"
SVD_RANK="${SVD_RANK:-32}"
HISTORY_PATH="${HISTORY_PATH:-$ART_DIR/merge_history.jsonl}"

args=(
  -m src.evolve_merge_lora
  --model-name "$MODEL_NAME"
  --adapters "$WORK_DIR/adapter_a" "$WORK_DIR/adapter_b" "$WORK_DIR/adapter_c"
  --truth "$DATA_DIR/val_holdout.jsonl"
  --out-dir "$MERGED_DIR"
  --history-path "$HISTORY_PATH"
  --population-size "$POPULATION_SIZE"
  --generations "$GENERATIONS"
  --elite-size "$ELITE_SIZE"
  --mutation-scale "$MUTATION_SCALE"
  --combination-type svd
  --svd-rank "$SVD_RANK"
  --max-samples "$MAX_SAMPLES"
)

if [[ "$LOAD_IN_4BIT" == "1" ]]; then
  args+=(--load-in-4bit)
fi

if [[ "$DISABLE_THINKING" == "1" ]]; then
  args+=(--disable-thinking)
fi

python "${args[@]}"

python -m src.package_submission \
  --adapter-dir "$MERGED_DIR" \
  --out "$WORK_DIR/submission.zip"
