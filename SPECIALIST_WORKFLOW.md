# Specialist LoRA Workflow

This is the simplest recommended order for this repo:

1. Build one shared holdout split and three specialist training datasets.
2. Train three LoRA adapters on the same Nemotron base model.
3. Merge the three adapters with evolutionary search.
4. Package the merged adapter as `submission.zip`.

## Step 0: Set paths

On Kaggle, these defaults should usually work:

```bash
export TRAIN_PATH=/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv
export WORK_DIR=/kaggle/working
export MODEL_NAME=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

## Step 1: Build datasets

```bash
bash src/scripts/build_specialist_data.sh
```

This creates:

- `/kaggle/working/artifacts/data/specialist_a_train.jsonl`
- `/kaggle/working/artifacts/data/specialist_b_train.jsonl`
- `/kaggle/working/artifacts/data/specialist_c_train.jsonl`
- `/kaggle/working/artifacts/data/val_holdout.jsonl`

## Step 2: Train specialist adapters

```bash
bash src/scripts/train_specialists.sh
```

This creates:

- `/kaggle/working/adapter_a`
- `/kaggle/working/adapter_b`
- `/kaggle/working/adapter_c`

If you already built the datasets, skip rebuilding them:

```bash
RUN_BUILD_DATA=0 bash src/scripts/train_specialists.sh
```

## Step 3: Merge the specialists

```bash
bash src/scripts/merge_specialists.sh
```

This creates:

- `/kaggle/working/merged_adapter`
- `/kaggle/working/submission.zip`

## Step 4: Submit

Upload `/kaggle/working/submission.zip` to Kaggle.

## Useful knobs

- `LOAD_IN_4BIT=1` keeps memory lower during training and merge search.
- `MAX_SAMPLES=256` in `merge_specialists.sh` makes merge search cheaper. Increase it once things are stable.
- `DISABLE_THINKING=1` is worth trying as an ablation if verbose reasoning hurts greedy decoding.

## One-line version

If you want the shortest path:

```bash
bash src/scripts/train_specialists.sh
bash src/scripts/merge_specialists.sh
```
