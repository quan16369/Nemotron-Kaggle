#!/bin/bash
set -euo pipefail

TRAIN_PATH="${TRAIN_PATH:-/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv}"
WORK_DIR="${WORK_DIR:-/kaggle/working}"
ART_DIR="${ART_DIR:-$WORK_DIR/artifacts}"
SYNTH_DIR="$ART_DIR/synth"
DATA_DIR="$ART_DIR/data"
MODEL_NAME="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/final_adapter}"

mkdir -p "$SYNTH_DIR" "$DATA_DIR"

python -m src.toolkit synth_all \
  --outdir "$SYNTH_DIR" \
  --sizes '{"roman_numeral":20000,"bit_binary":50000,"unit_conversion":25000,"gravity":25000,"equation":20000,"text_decrypt":20000}'

python -m src.toolkit build_comp_data \
  --train-csv "$TRAIN_PATH" \
  --out-train "$DATA_DIR/train_competitive.jsonl" \
  --out-val "$DATA_DIR/val_holdout.jsonl" \
  --real-repeat 2 \
  --real-style-mix '{"answer_only":0.60,"ultra_short":0.25,"concise":0.15}' \
  --synth-cap 60000 \
  --synth \
    "$SYNTH_DIR"/roman_numeral.jsonl \
    "$SYNTH_DIR"/bit_binary.jsonl \
    "$SYNTH_DIR"/unit_conversion.jsonl \
    "$SYNTH_DIR"/gravity.jsonl \
    "$SYNTH_DIR"/equation.jsonl \
    "$SYNTH_DIR"/text_decrypt.jsonl

python -m src.train_lora \
  --config config.json \
  --dataset "$DATA_DIR/train_competitive.jsonl" \
  --model_name "$MODEL_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 4096 \
  --load_in_4bit

python -m src.package_submission \
  --adapter-dir "$OUTPUT_DIR" \
  --out "$WORK_DIR/submission.zip"
