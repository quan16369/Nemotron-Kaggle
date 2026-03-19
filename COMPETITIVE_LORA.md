# Competitive LoRA Pipeline

This repo now has a Kaggle-friendly pipeline that still submits a plain LoRA adapter, but trains it with stronger data preparation and template alignment than the bare notebook baseline.

## What changed

- Real `train.csv` examples can be mixed with synthetic data.
- Training examples are serialized with the model tokenizer's chat template instead of manual `<|user|>/<|assistant|>` formatting.
- Holdout prompts can be built for local scoring.
- Adapter packaging is a one-command step that produces `submission.zip`.

## Main commands

Build a competition-style train/holdout mixture:

```bash
python -m src.toolkit build_comp_data \
  --train-csv /kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv \
  --out-train /kaggle/working/artifacts/data/train_competitive.jsonl \
  --out-val /kaggle/working/artifacts/data/val_holdout.jsonl \
  --real-repeat 2 \
  --real-style-mix '{"answer_only":0.60,"ultra_short":0.25,"concise":0.15}' \
  --synth /kaggle/working/artifacts/synth/*.jsonl
```

Train the LoRA:

```bash
python -m src.train_lora \
  --config config.json \
  --dataset /kaggle/working/artifacts/data/train_competitive.jsonl \
  --model_name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --output_dir /kaggle/working/final_adapter \
  --max_seq_length 4096 \
  --load_in_4bit
```

Package the adapter for submission:

```bash
python -m src.package_submission \
  --adapter-dir /kaggle/working/final_adapter \
  --out /kaggle/working/submission.zip
```

Score local predictions against the holdout file:

```bash
python -m src.metric score \
  --truth /kaggle/working/artifacts/data/val_holdout.jsonl \
  --pred /kaggle/working/preds.jsonl \
  --prediction-column prediction
```

Evolutionary-search a merged adapter from multiple LoRA checkpoints:

```bash
python -m src.evolve_merge_lora \
  --model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --adapters \
    /kaggle/working/adapter_a \
    /kaggle/working/adapter_b \
    /kaggle/working/adapter_c \
  --truth /kaggle/working/artifacts/data/val_holdout.jsonl \
  --out-dir /kaggle/working/merged_adapter \
  --population-size 6 \
  --generations 4 \
  --elite-size 2 \
  --combination-type svd \
  --svd-rank 32 \
  --load-in-4bit
```

## Notes

- `src/scripts/run_competitive.sh` strings the whole flow together.
- `src/scripts/build_specialist_data.sh`, `src/scripts/train_specialists.sh`, and `src/scripts/merge_specialists.sh` implement the recommended specialist-to-merge workflow.
- [SPECIALIST_WORKFLOW.md](/home/quan/nemotron-kaggle/SPECIALIST_WORKFLOW.md) gives the exact command order.
- `src.metric` defaults to `rel_tol=1e-4`. Adjust it if the official metric reveals a different numeric tolerance.
- The final artifact is still a standard LoRA adapter zip with `adapter_config.json` at the zip root.
- `src.evolve_merge_lora` keeps the final merged adapter at rank `<= svd_rank`, so set `svd_rank=32` for this contest.
