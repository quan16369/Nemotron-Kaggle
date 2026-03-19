#!/bin/bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-/kaggle/working}"
ART_DIR="${ART_DIR:-$WORK_DIR/artifacts}"
DATA_DIR="$ART_DIR/data"
MODEL_NAME="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"
CONFIG_PATH="${CONFIG_PATH:-config.json}"
RUN_BUILD_DATA="${RUN_BUILD_DATA:-1}"

if [[ "$RUN_BUILD_DATA" == "1" ]]; then
  bash src/scripts/build_specialist_data.sh
fi

train_one() {
  local dataset_path="$1"
  local output_dir="$2"

  local args=(
    -m src.train_lora
    --config "$CONFIG_PATH"
    --dataset "$dataset_path"
    --model_name "$MODEL_NAME"
    --output_dir "$output_dir"
    --max_seq_length "$MAX_SEQ_LENGTH"
  )

  if [[ "$LOAD_IN_4BIT" == "1" ]]; then
    args+=(--load_in_4bit)
  fi

  if [[ "$DISABLE_THINKING" == "1" ]]; then
    args+=(--disable_thinking)
  fi

  python "${args[@]}"
}

train_one "$DATA_DIR/specialist_a_train.jsonl" "$WORK_DIR/adapter_a"
train_one "$DATA_DIR/specialist_b_train.jsonl" "$WORK_DIR/adapter_b"
train_one "$DATA_DIR/specialist_c_train.jsonl" "$WORK_DIR/adapter_c"
