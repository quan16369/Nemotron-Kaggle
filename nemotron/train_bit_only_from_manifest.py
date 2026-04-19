"""Train a bit_manipulation-only LoRA adapter from a CSV manifest on Kaggle.

This script is a standalone extraction of the main training path from
`training-with-unsloth-to-achieve-0-81-lb.ipynb`, reduced to:

- install the same offline dependencies
- load the same Nemotron base model
- filter the manifest down to `bit_manipulation`
- train a LoRA adapter with the same winning-default loss and LR schedule

Example:
    python train_bit_only_from_manifest.py \
      --training-manifest-csv /kaggle/input/winning-snapshot-delta-manifest/winning_snapshot_delta_manifest.csv \
      --output-dir /kaggle/working/bit_only_sft_adapter
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import shutil
import site
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MANIFEST_CSV_CANDIDATES = [
    Path("/kaggle/input/datasets/hiyato/winning-snapshot-delta-manifest/winning_snapshot_delta_manifest.csv"),
    Path("/kaggle/input/winning_snapshot_delta_manifest/manifest.csv"),
    Path("/kaggle/input/winning-snapshot-delta-manifest/winning_snapshot_delta_manifest.csv"),
    Path("/kaggle/input/winning-snapshot-delta-manifest/manifest.csv"),
    Path("/kaggle/input/nemotron-training-manifest/winning_snapshot_delta_manifest.csv"),
    Path("/kaggle/input/nemotron-training-manifest/manifest.csv"),
]

DEFAULT_BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_CATEGORY = "bit_manipulation"
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/bit_only_sft_adapter")
DEFAULT_TRAINING_ROOT = Path("/kaggle/working/bit_only_sft_output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-manifest-csv",
        type=Path,
        default=None,
        help="Manifest CSV in the notebook format. Defaults to common Kaggle input paths.",
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help="Category to train on. Default: bit_manipulation",
    )
    parser.add_argument(
        "--base-model-name",
        default=DEFAULT_BASE_MODEL_NAME,
        help="Reference base model name to write into the adapter config.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local Kaggle model path. If omitted, resolve through kagglehub.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the trained adapter.",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
        help="Trainer output directory.",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Must satisfy effective_batch_size = per_device_batch_size * gradient_accumulation_steps.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epoch-batch-order-seed", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--reasoning-token-budget", type=int, default=7680)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument(
        "--limit-examples",
        type=int,
        default=None,
        help="Optional debug cap after filtering by category.",
    )
    return parser.parse_args()


def find_default_manifest() -> Path | None:
    return next((path for path in DEFAULT_MANIFEST_CSV_CANDIDATES if path.exists()), None)


def install_offline_triton() -> None:
    candidates = glob.glob("/kaggle/input/**/*triton*.whl", recursive=True)
    print("Found Triton wheels:", candidates)
    if not candidates:
        print("No offline Triton wheel found. Continuing with runtime Triton.")
        return

    wheel = candidates[0]
    target = "/kaggle/working/pydeps"
    os.makedirs(target, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            target,
            "--upgrade",
            "--ignore-installed",
            wheel,
        ],
        check=True,
    )
    if target not in sys.path:
        sys.path.insert(0, target)
    site.addsitedir(target)
    import importlib.util

    print("Custom target added:", target)
    print("triton spec:", importlib.util.find_spec("triton"))


def apply_kaggle_training_env_fixes() -> None:
    sys.path.insert(0, "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script")

    ptxas_src = (
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/"
        "triton/backends/nvidia/bin/ptxas-blackwell"
    )
    ptxas_dst = "/tmp/ptxas-blackwell"
    if os.path.exists(ptxas_src) and not os.path.exists(ptxas_dst):
        shutil.copy2(ptxas_src, ptxas_dst)
        os.chmod(ptxas_dst, os.stat(ptxas_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        src_bin = os.path.dirname(ptxas_src)
        dst_bin = "/tmp/triton_nvidia_bin"
        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
        for filename in os.listdir(dst_bin):
            file_path = os.path.join(dst_bin, filename)
            if os.path.isfile(file_path):
                os.chmod(
                    file_path,
                    os.stat(file_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
                )

        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_dst
        os.environ["TRITON_PTXAS_PATH"] = ptxas_dst

        import triton.backends.nvidia as nv_backend  # type: ignore[import-not-found]

        nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")

    import triton.backends.nvidia.compiler as nv_compiler  # type: ignore[import-not-found]

    nv_compiler.get_ptxas_version = lambda arch: "12.0"
    print("Training environment fixes applied.")


def install_offline_packages() -> None:
    import torch

    def recursive_wheels(pattern: str) -> list[str]:
        return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))

    packages_dir = "/kaggle/input/datasets/mayukh18/nemotron-packages/packages"
    all_mamba = recursive_wheels("mamba_ssm-*.whl")
    all_causal = recursive_wheels("causal*conv1d*.whl")

    print("Found mamba wheels:", all_mamba)
    print("Found causal-conv1d wheels:", all_causal)
    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA:", torch.version.cuda)

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a Kaggle GPU runtime.")
    if not os.path.isdir(packages_dir):
        raise FileNotFoundError(f"Offline wheel directory not found: {packages_dir}")

    base_packages = [
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "tokenizers",
        "sentencepiece",
        "safetensors",
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-index",
            "--find-links",
            packages_dir,
            *base_packages,
        ],
        check=True,
    )

    def pick_last(wheels: list[str]) -> str | None:
        return wheels[-1] if wheels else None

    causal_wheel = pick_last(all_causal)
    mamba_wheel = pick_last(all_mamba)
    print("Selected causal wheel:", causal_wheel)
    print("Selected mamba wheel:", mamba_wheel)

    if causal_wheel:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", causal_wheel],
            check=True,
        )
    if mamba_wheel:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", mamba_wheel],
            check=True,
        )
    else:
        raise FileNotFoundError("Could not find a compatible mamba_ssm wheel.")

    print("Offline package installation finished.")


def resolve_model_path(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path

    import kagglehub

    model_path = kagglehub.model_download(
        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    )
    print(f"Model path: {model_path}")
    return model_path


def load_records_from_manifest(
    *,
    manifest_path: Path,
    category: str,
    max_seq_len: int,
    limit_examples: int | None,
) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["category"] != category:
                continue
            input_ids = json.loads(row["input_ids_json"])
            mask = json.loads(row["mask_json"])
            if len(input_ids) != len(mask):
                raise ValueError(f"Length mismatch in manifest row {row['problem_id']}")
            if len(input_ids) > max_seq_len:
                raise ValueError(
                    f"Manifest row {row['problem_id']} exceeds max length: {len(input_ids)} > {max_seq_len}"
                )
            labels = [token if m == 1 else -100 for token, m in zip(input_ids, mask)]
            records.append(
                {
                    "problem_id": row["problem_id"],
                    "source_problem_id": row["source_problem_id"],
                    "category": row["category"],
                    "segment": row.get("segment", "synthetic.jsonl"),
                    "num_loss_tokens": int(row["num_loss_tokens"]),
                    "completion_token_count": int(
                        row.get("completion_token_count") or row["num_loss_tokens"]
                    ),
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )
            if limit_examples is not None and len(records) >= limit_examples:
                break

    records.sort(key=lambda record: record["problem_id"])
    return records


def build_stratified_index_order(labels: list[str], batch_size: int, seed: int) -> list[int]:
    by_label: defaultdict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    rng = random.Random(seed)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)

    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)

    assigned = 0
    for label in sorted(by_label.keys()):
        for idx in by_label[label]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1

    order = [idx for batch in batches for idx in batch]
    if len(order) != len(labels):
        raise ValueError("Stratified order size mismatch")
    return order


def main() -> None:
    args = parse_args()

    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="strict")

    random.seed(args.seed)

    manifest_path = args.training_manifest_csv or find_default_manifest()
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(
            "Could not find a training manifest CSV. Pass --training-manifest-csv explicitly."
        )

    install_offline_triton()
    apply_kaggle_training_env_fixes()
    install_offline_packages()

    import torch
    import torch.nn.functional as F
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader, Sampler
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    class CrossEntropyLossConfig:
        name = "cross_entropy"
        class_name = "CrossEntropyLossConfig"

    class StepLinearDecayLRSchedule:
        class_name = "StepLinearDecayLRSchedule"

        def __init__(self, learning_rate: float = 2e-4):
            self.learning_rate = learning_rate

        def get_lr(self, step: int, total_steps: int, epoch: int, total_epochs: int) -> float:
            return self.learning_rate * (1 - step / total_steps)

    if args.effective_batch_size != args.per_device_batch_size * args.gradient_accumulation_steps:
        raise ValueError(
            "effective_batch_size must equal per_device_batch_size * gradient_accumulation_steps"
        )

    model_path = resolve_model_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs.update(
            {
                "torch_dtype": torch.bfloat16,
                "quantization_config": bnb_config,
                "device_map": "auto",
            }
        )
    else:
        load_kwargs.update(
            {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False
    print("Model loaded with standard Transformers.")

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj",
        "out_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    records = load_records_from_manifest(
        manifest_path=manifest_path,
        category=args.category,
        max_seq_len=args.max_seq_len,
        limit_examples=args.limit_examples,
    )
    if not records:
        raise ValueError(f"No manifest rows found for category={args.category}")

    category_counts = Counter(record["category"] for record in records)
    completion_lengths = [record["completion_token_count"] for record in records]
    total_tokens = sum(len(record["input_ids"]) for record in records)
    total_loss_tokens = sum(record["num_loss_tokens"] for record in records)
    max_completion_tokens = max(completion_lengths, default=0)
    over_budget = sum(length > args.reasoning_token_budget for length in completion_lengths)

    print(
        {
            "training_manifest_csv": str(manifest_path),
            "category": args.category,
            "manifest_examples": len(records),
            "manifest_total_tokens": total_tokens,
            "manifest_loss_tokens": total_loss_tokens,
            "max_completion_tokens": max_completion_tokens,
            "completion_over_budget": over_budget,
        }
    )
    print(f"Training categories: {category_counts}")

    train_dataset = HFDataset.from_list(records)
    record_categories = [record["category"] for record in records]

    class MaskedDataCollator:
        def __init__(self, pad_token_id: int):
            self.pad_token_id = pad_token_id

        def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
            max_len = max(len(feature["input_ids"]) for feature in features)
            input_ids = []
            attention_mask = []
            labels = []
            for feature in features:
                pad_len = max_len - len(feature["input_ids"])
                input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
                attention_mask.append(feature["attention_mask"] + [0] * pad_len)
                labels.append(feature["labels"] + [-100] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    class PrecomputedOrderSampler(Sampler[int]):
        def __init__(self, order: list[int]):
            self.order = list(order)

        def __iter__(self):
            return iter(self.order)

        def __len__(self) -> int:
            return len(self.order)

    class StratifiedTrainer(Trainer):
        def __init__(self, *args, stratified_order=None, loss_config=None, lr_schedule_config=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.stratified_order = stratified_order
            self.loss_config = loss_config or CrossEntropyLossConfig()
            self.lr_schedule_config = lr_schedule_config or StepLinearDecayLRSchedule(
                learning_rate=self.args.learning_rate
            )

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer requires a train_dataset.")
            if self.stratified_order is None:
                return super().get_train_dataloader()
            sampler = PrecomputedOrderSampler(self.stratified_order)
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=torch.cuda.is_available(),
            )

        def create_scheduler(self, num_training_steps: int, optimizer=None):
            if self.lr_scheduler is None:
                optimizer = optimizer or self.optimizer
                if optimizer is None:
                    raise ValueError("Trainer optimizer must be created before scheduler.")
                total_steps = max(num_training_steps, 1)
                base_lr = max(self.lr_schedule_config.learning_rate, 1e-12)

                def lr_lambda(current_step: int) -> float:
                    step = min(max(current_step, 0), total_steps)
                    lr = self.lr_schedule_config.get_lr(
                        step=step,
                        total_steps=total_steps,
                        epoch=0,
                        total_epochs=max(int(self.args.num_train_epochs), 1),
                    )
                    return max(lr, 0.0) / base_lr

                self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return self.lr_scheduler

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if self.loss_config.name != "cross_entropy":
                raise NotImplementedError("Only cross_entropy is implemented.")
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = inputs["labels"][:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            return (loss, outputs) if return_outputs else loss

    stratified_order = build_stratified_index_order(
        record_categories,
        args.effective_batch_size,
        args.epoch_batch_order_seed,
    )

    training_args = TrainingArguments(
        output_dir=str(args.training_root),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1e9,
    )

    trainer = StratifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MaskedDataCollator(tokenizer.pad_token_id),
        stratified_order=stratified_order,
        loss_config=CrossEntropyLossConfig(),
        lr_schedule_config=StepLinearDecayLRSchedule(learning_rate=args.learning_rate),
    )

    train_result = trainer.train()
    print(train_result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved adapter to {args.output_dir}")

    print(
        json.dumps(
            {
                "category": args.category,
                "training_manifest_csv": str(manifest_path),
                "examples": len(records),
                "total_tokens": total_tokens,
                "total_loss_tokens": total_loss_tokens,
                "output_dir": str(args.output_dir),
                "training_root": str(args.training_root),
                "base_model_name": args.base_model_name,
                "use_4bit": args.use_4bit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
