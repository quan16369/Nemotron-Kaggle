# Nemotron Kaggle Workflow

This repo is set up for the NVIDIA Nemotron Model Reasoning Challenge.

The recommended path is:

1. Build synthetic data and one shared holdout split.
2. Build three specialist training datasets.
3. Train three LoRA adapters on the same Nemotron base model.
4. Merge the adapters with evolutionary search.
5. Package the merged adapter as `submission.zip`.

## Manual Kaggle Commands

If you prefer writing and running bash manually on Kaggle, use the commands below in this exact order.

### 1. Set paths

```bash
export TRAIN_PATH=/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv
export WORK_DIR=/kaggle/working
export ART_DIR=$WORK_DIR/artifacts
export SYNTH_DIR=$ART_DIR/synth
export DATA_DIR=$ART_DIR/data
export MODEL_NAME=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

mkdir -p "$SYNTH_DIR" "$DATA_DIR"
```

### 2. Generate synthetic data

```bash
python -m src.toolkit synth_all \
  --outdir "$SYNTH_DIR" \
  --sizes '{"roman_numeral":20000,"bit_binary":50000,"unit_conversion":25000,"gravity":25000,"equation":20000,"text_decrypt":20000}'
```

### 3. Build holdout and specialist dataset A

```bash
python -m src.toolkit build_comp_data \
  --train-csv "$TRAIN_PATH" \
  --out-train "$DATA_DIR/specialist_a_train.jsonl" \
  --out-val "$DATA_DIR/val_holdout.jsonl" \
  --holdout-ids-out "$DATA_DIR/holdout_ids.json" \
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
```

### 4. Build specialist dataset B

```bash
python -m src.toolkit build_comp_data \
  --train-csv "$TRAIN_PATH" \
  --out-train "$DATA_DIR/specialist_b_train.jsonl" \
  --holdout-ids-in "$DATA_DIR/holdout_ids.json" \
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
```

### 5. Build specialist dataset C

```bash
python -m src.toolkit build_comp_data \
  --train-csv "$TRAIN_PATH" \
  --out-train "$DATA_DIR/specialist_c_train.jsonl" \
  --holdout-ids-in "$DATA_DIR/holdout_ids.json" \
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
```

### 6. Train adapter A

```bash
python -m src.train_lora \
  --config config.json \
  --dataset "$DATA_DIR/specialist_a_train.jsonl" \
  --model_name "$MODEL_NAME" \
  --output_dir "$WORK_DIR/adapter_a" \
  --max_seq_length 4096 \
  --load_in_4bit
```

### 7. Train adapter B

```bash
python -m src.train_lora \
  --config config.json \
  --dataset "$DATA_DIR/specialist_b_train.jsonl" \
  --model_name "$MODEL_NAME" \
  --output_dir "$WORK_DIR/adapter_b" \
  --max_seq_length 4096 \
  --load_in_4bit
```

### 8. Train adapter C

```bash
python -m src.train_lora \
  --config config.json \
  --dataset "$DATA_DIR/specialist_c_train.jsonl" \
  --model_name "$MODEL_NAME" \
  --output_dir "$WORK_DIR/adapter_c" \
  --max_seq_length 4096 \
  --load_in_4bit
```

### 9. Evolutionary merge

```bash
python -m src.evolve_merge_lora \
  --model-name "$MODEL_NAME" \
  --adapters "$WORK_DIR/adapter_a" "$WORK_DIR/adapter_b" "$WORK_DIR/adapter_c" \
  --truth "$DATA_DIR/val_holdout.jsonl" \
  --out-dir "$WORK_DIR/merged_adapter" \
  --history-path "$ART_DIR/merge_history.jsonl" \
  --population-size 6 \
  --generations 4 \
  --elite-size 2 \
  --mutation-scale 0.15 \
  --combination-type svd \
  --svd-rank 32 \
  --max-samples 256 \
  --load-in-4bit
```

### 10. Package submission

```bash
python -m src.package_submission \
  --adapter-dir "$WORK_DIR/merged_adapter" \
  --out "$WORK_DIR/submission.zip"
```

Submit:

```bash
$WORK_DIR/submission.zip
```

## Quick Notes

- All adapters must be trained on the same base model.
- Keep the final merged adapter at rank `<= 32`.
- If compute is tight, reduce `--synth-cap` and `--max-samples`.
- If verbose reasoning hurts greedy decoding, try `--disable_thinking` during training and merge evaluation.

## Related Files

- `src/train_lora.py`
- `src/toolkit.py`
- `src/evolve_merge_lora.py`
- `src/package_submission.py`
- `SPECIALIST_WORKFLOW.md`
- `COMPETITIVE_LORA.md`
