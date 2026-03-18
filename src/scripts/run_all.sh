#!/bin/bash
set -e

TRAIN_PATH="/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv"
WORK_DIR="/kaggle/working"
ART_DIR="$WORK_DIR/artifacts"
SYNTH_DIR="$ART_DIR/synth"

mkdir -p "$SYNTH_DIR"

python -m src.toolkit analyze --csv "$TRAIN_PATH"

python -m src.toolkit synth_all \
  --outdir "$SYNTH_DIR" \
  --sizes '{"roman_numeral":20000,"bit_binary":50000,"unit_conversion":25000,"gravity":25000,"equation":20000,"text_decrypt":20000}'

python -m src.toolkit make_mixture \
  --out "$ART_DIR/mixture.jsonl" \
  "$SYNTH_DIR"/roman_numeral.jsonl \
  "$SYNTH_DIR"/bit_binary.jsonl \
  "$SYNTH_DIR"/unit_conversion.jsonl \
  "$SYNTH_DIR"/gravity.jsonl \
  "$SYNTH_DIR"/equation.jsonl \
  "$SYNTH_DIR"/text_decrypt.jsonl

python -m src.toolkit export_chatml \
  --input "$ART_DIR/mixture.jsonl" \
  --out "$ART_DIR/train_chatml.jsonl"

python -m src.train_lora \
  --config config.json \
  --dataset "$ART_DIR/train_chatml.jsonl" \
  --model_name "YOUR_NEMOTRON_MODEL" \
  --output_dir "$WORK_DIR/final_adapter" \
  --max_seq_length 4096 \
  --load_in_4bit
