#!/bin/bash
set -euo pipefail

TRAIN_PATH="${TRAIN_PATH:-/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv}"
WORK_DIR="${WORK_DIR:-/kaggle/working}"
ART_DIR="${ART_DIR:-$WORK_DIR/artifacts}"
SYNTH_DIR="$ART_DIR/synth"
DATA_DIR="$ART_DIR/data"
HOLDOUT_IDS_PATH="$DATA_DIR/holdout_ids.json"
VAL_PATH="$DATA_DIR/val_holdout.jsonl"

mkdir -p "$SYNTH_DIR" "$DATA_DIR"

python -m src.toolkit synth_all \
  --outdir "$SYNTH_DIR" \
  --sizes '{"roman_numeral":20000,"bit_binary":50000,"unit_conversion":25000,"gravity":25000,"equation":20000,"text_decrypt":20000}'

python -m src.toolkit build_comp_data \
  --train-csv "$TRAIN_PATH" \
  --out-train "$DATA_DIR/specialist_a_train.jsonl" \
  --out-val "$VAL_PATH" \
  --holdout-ids-out "$HOLDOUT_IDS_PATH" \
  --seed 101 \
  --real-repeat 3 \
  --real-style-mix '{"answer_only":0.75,"ultra_short":0.20,"concise":0.05}' \
  --synth-cap 20000 \
  --synth \
    "$SYNTH_DIR"/roman_numeral.jsonl \
    "$SYNTH_DIR"/bit_binary.jsonl \
    "$SYNTH_DIR"/unit_conversion.jsonl \
    "$SYNTH_DIR"/gravity.jsonl \
    "$SYNTH_DIR"/equation.jsonl \
    "$SYNTH_DIR"/text_decrypt.jsonl

python -m src.toolkit build_comp_data \
  --train-csv "$TRAIN_PATH" \
  --out-train "$DATA_DIR/specialist_b_train.jsonl" \
  --holdout-ids-in "$HOLDOUT_IDS_PATH" \
  --seed 202 \
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

python -m src.toolkit build_comp_data \
  --train-csv "$TRAIN_PATH" \
  --out-train "$DATA_DIR/specialist_c_train.jsonl" \
  --holdout-ids-in "$HOLDOUT_IDS_PATH" \
  --seed 303 \
  --real-repeat 1 \
  --real-style-mix '{"answer_only":0.45,"ultra_short":0.20,"concise":0.35}' \
  --synth-cap 100000 \
  --synth \
    "$SYNTH_DIR"/roman_numeral.jsonl \
    "$SYNTH_DIR"/bit_binary.jsonl \
    "$SYNTH_DIR"/unit_conversion.jsonl \
    "$SYNTH_DIR"/gravity.jsonl \
    "$SYNTH_DIR"/equation.jsonl \
    "$SYNTH_DIR"/text_decrypt.jsonl

echo "Built specialist datasets:"
ls -1 "$DATA_DIR"/specialist_*_train.jsonl "$VAL_PATH"
