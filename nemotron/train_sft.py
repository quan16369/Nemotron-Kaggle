"""SFT training on nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 with per-epoch logprob saving.

Supports loss functions: cross_entropy, importance_sampling, ppo, cispo, dro.

(when training on modal)
uv run modal deploy trainer/server.py

uv run python3 -m train_sft
"""

from __future__ import annotations
from collections import Counter

import asyncio
import dataclasses
import json
import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import tinker
from tinker import types
from trainer.client import ServiceClient

from loss_config import (
    CrossEntropyLossConfig,
    CrossEntropyWithWeightingLossConfig,
    LossConfig,
)
from lr_schedule import LRSchedule, StepLinearDecayLRSchedule
from train_common import TrainingExample, load_corpus_entries

logger = logging.getLogger(__name__)


def _stratified_batches(
    examples: list[TrainingExample], batch_size: int, rng: random.Random
) -> list[list[int]]:
    """Return equal-sized batches with categories distributed evenly.

    Deals entries one at a time into a shuffled batch order, iterating
    through categories sequentially so each category's entries are spread
    across batches.
    """
    n = len(examples)
    n_batches = math.ceil(n / batch_size)

    # Group by category, shuffle within each
    by_cat: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        by_cat.setdefault(ex.category, []).append(i)
    for idx_list in by_cat.values():
        rng.shuffle(idx_list)

    # Each category gets its own shuffled batch order
    batches: list[list[int]] = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)
    assigned = 0
    for cat in sorted(by_cat.keys()):
        for idx in by_cat[cat]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1

    return batches


@dataclasses.dataclass
class AdamConfig:
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1e9

    def to_adam_params(self, learning_rate: float) -> types.AdamParams:
        return types.AdamParams(
            learning_rate=learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )


@dataclasses.dataclass
class Cfg:
    loss_config: LossConfig = dataclasses.field(default_factory=CrossEntropyLossConfig)
    lr_schedule: LRSchedule = dataclasses.field(
        default_factory=lambda: StepLinearDecayLRSchedule(learning_rate=2e-4)
    )
    log_path: str = dataclasses.field(
        default_factory=lambda: datetime.now().strftime("%m-%d-%H-%M")
    )
    model_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    batch_size: int = 64
    num_epochs: int = 1
    lora_rank: int = 32 # 32
    max_length: int = 8192
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    adam_config: AdamConfig = dataclasses.field(default_factory=AdamConfig)
    backend: Literal["tinker", "modal"] = "tinker"
    micro_batch_size: int | None = 16  # None for tinker (server decides), int for modal


@dataclasses.dataclass
class LogprobRecord:
    problem_id: str
    segment: str
    logprobs: list[float]

    def save(self, epoch_dir: Path) -> None:
        stem = self.segment.removesuffix(".jsonl")
        path = epoch_dir / self.problem_id / f"{stem}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"logprobs": self.logprobs}) + "\n")


@dataclasses.dataclass
class IndexRecord:
    epoch: int
    step: int
    problem_id: str
    segment: str
    category: str
    num_loss_tokens: int
    total_loss: float
    min_logprob: float


def build_datum(
    tokens: list[int],
    advantages: list[int],
    ref_logprobs: list[float] | None,
    prev_logprobs: list[float] | None,
    epoch: int,
    loss: LossConfig,
) -> tinker.Datum:
    """Build a Tinker Datum for training."""
    assert len(tokens) == len(advantages)
    assert ref_logprobs is None or len(ref_logprobs) == len(tokens) - 1
    assert prev_logprobs is None or len(prev_logprobs) == len(tokens) - 1

    model_input = tinker.ModelInput(
        chunks=[tinker.types.EncodedTextChunk(tokens=tokens[:-1])]
    )
    target_tokens = tokens[1:]

    loss_fn_inputs: dict[str, tinker.TensorData] = {
        "target_tokens": tinker.TensorData(
            data=target_tokens,
            dtype="int64",
            shape=[len(target_tokens)],
        ),
    }

    float_advantages = [float(a) for a in advantages[1:]]

    if isinstance(loss, CrossEntropyLossConfig):
        if isinstance(loss, CrossEntropyWithWeightingLossConfig):
            float_advantages = loss.apply_weights(
                float_advantages, prev_logprobs, ref_logprobs, epoch
            )
        loss_fn_inputs["weights"] = tinker.TensorData(
            data=float_advantages,
            dtype="float32",
            shape=[len(float_advantages)],
        )
    else:
        loss_fn_inputs["advantages"] = tinker.TensorData(
            data=float_advantages,
            dtype="float32",
            shape=[len(float_advantages)],
        )
        if ref_logprobs is not None:
            loss_fn_inputs["logprobs"] = tinker.TensorData(
                data=ref_logprobs,
                dtype="float32",
                shape=[len(ref_logprobs)],
            )

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs=loss_fn_inputs,
    )


def compute_epoch_metrics(
    loss_config: LossConfig,
    examples: list[TrainingExample],
    all_ref_logprobs: dict[str, list[float]],
    all_epoch_logprobs: dict[str, list[float]],
    epoch: int,
) -> list[list[dict[str, float]]]:
    """Compute aggregate metrics for one epoch, averaged across all examples."""
    all_metrics: dict[str, list[tuple[float, int]]] = {}
    all_nll_values: list[tuple[float, int]] = []
    all_token_diffs: list[float] = []
    all_token_epoch_lps: list[float] = []
    for example in examples:
        key = example.problem_id
        epoch_logprobs = all_epoch_logprobs[key]
        ref_logprobs = all_ref_logprobs[key]
        _, full_mask = example.load_tokens()
        target_mask = full_mask[1:]
        target_mask = target_mask[: len(epoch_logprobs)]
        weight = example.unmasked_token_count
        metrics = loss_config.compute_metrics(
            epoch_logprobs, ref_logprobs, target_mask, epoch=epoch
        )
        for metric_name, metric_value in metrics.items():
            all_metrics.setdefault(metric_name, []).append((metric_value, weight))
        unmasked_lps = []
        for lp, rp, m in zip(epoch_logprobs, ref_logprobs, target_mask):
            if m == 1:
                unmasked_lps.append(lp)
                all_token_diffs.append(lp - rp)
                all_token_epoch_lps.append(lp)
        if unmasked_lps:
            nll = -sum(unmasked_lps) / len(unmasked_lps)
            all_nll_values.append((nll, weight))

    # Token-weighted average for each metric
    avg = {}
    for k, metric_entries in all_metrics.items():
        total_weight = sum(w for _, w in metric_entries)
        if total_weight > 0:
            avg[k] = round(sum(v * w for v, w in metric_entries) / total_weight, 6)

    # General metrics: nll, perplexity, logprob diff percentiles
    general_metrics: list[list[dict]] = []
    if all_nll_values:
        total_w = sum(w for _, w in all_nll_values)
        avg_nll = sum(v * w for v, w in all_nll_values) / total_w
        general_metrics.append([{"nll_per_token": round(avg_nll, 6)}])
        general_metrics.append([{"perplexity": round(math.exp(min(avg_nll, 20)), 4)}])
    general_metrics.extend(
        loss_config.compute_global_metrics(all_token_diffs, all_token_epoch_lps)
    )

    # Structure into chart groups per chart_layout
    layout = loss_config.chart_layout()
    metrics_grouped: list[list[dict]] = []
    for group in layout:
        chart = [{k: avg[k]} for k in group if k in avg]
        if chart:
            metrics_grouped.append(chart)
    return metrics_grouped[:2] + general_metrics + metrics_grouped[2:]


def filter_training_examples(examples: list[TrainingExample]) -> list[TrainingExample]:
    # return [
    #     e
    #     for e in examples
    #     if e.category in ("numeral", "cryptarithm_deduce", "cryptarithm_guess")
    # ]
    return examples


async def main():
    cfg = Cfg()

    sft_dir = Path("./training/sft")
    log_path = sft_dir / cfg.log_path
    logprob_dir = log_path / "logprobs"

    # Load examples
    entries = load_corpus_entries()
    entries = [e for e in entries if e["included"]]
    examples = [TrainingExample.from_dict(e) for e in entries]
    examples = filter_training_examples(examples)
    total_masked = sum(e.masked_token_count for e in examples)
    total_unmasked = sum(e.unmasked_token_count for e in examples)
    logger.info(
        f"Loaded {len(examples)} examples, "
        f"{total_masked + total_unmasked:,} tokens "
        f"(unmasked={total_unmasked:,}, masked={total_masked:,})"
    )

    n_batches = math.ceil(len(examples) / cfg.batch_size)
    total_steps = n_batches * cfg.num_epochs
    logger.info(
        f"Training for {n_batches} batches x {cfg.num_epochs} epochs"
        f" = {total_steps} steps"
    )
    lr_params = dataclasses.asdict(cfg.lr_schedule)
    logger.info(f"LR schedule: {type(cfg.lr_schedule).__name__} {lr_params}")

    # Create training client
    service_client = ServiceClient(backend=cfg.backend)
    training_client = await service_client.create_lora_training_client_async(
        base_model=cfg.model_name,
        rank=cfg.lora_rank,
        train_mlp=cfg.train_mlp,
        train_attn=cfg.train_attn,
        train_unembed=cfg.train_unembed,
    )

    logprob_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = dataclasses.asdict(cfg)
    config["time"] = datetime.now().strftime("%m-%d-%H-%M")
    config["stats"] = {
        "num_examples": len(examples),
        "total_masked_tokens": total_masked,
        "total_unmasked_tokens": total_unmasked,
        "total_steps": total_steps,
    }
    with open(log_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Append to logpaths index for GitHub Pages
    logpaths_file = sft_dir / "logpaths.txt"
    existing = set(logpaths_file.read_text().splitlines()) if logpaths_file.exists() else set()
    if cfg.log_path not in existing:
        with open(logpaths_file, "a") as f:
            f.write(cfg.log_path + "\n")

    step = 0
    # Collected during epoch 0, used as ref logprobs in epochs 1+
    all_ref_logprobs: dict[str, list[float]] = {}
    # Logprobs from the latest epoch
    all_prev_logprobs: dict[str, list[float]] = {}

    metrics_path = log_path / "metrics.jsonl"
    loss_path = log_path / "loss.jsonl"
    with (
        open(logprob_dir / "index.jsonl", "w") as index_file,
        open(metrics_path, "w") as metrics_file,
        open(loss_path, "w") as loss_file,
    ):
        for epoch in range(cfg.num_epochs):
            loss_fn_config = cfg.loss_config.config(epoch)
            logger.info(
                f"Starting epoch {epoch} "
                f"(loss={cfg.loss_config.name} config={loss_fn_config})"
            )
            epoch_start = time.time()

            rng = random.Random(epoch)
            batches = _stratified_batches(examples, cfg.batch_size, rng)

            print(f"Batch sizes: {Counter(len(batch) for batch in batches)}")

            epoch_dir = logprob_dir / str(epoch)

            for batch_indices in batches:
                batch_examples = [examples[i] for i in batch_indices]

                data: list[tinker.Datum] = []
                valid_examples: list[TrainingExample] = []
                target_masks: list[list[int]] = []
                batch_tokens: list[list[int]] = []
                batch_masks: list[list[int]] = []
                for example in batch_examples:
                    tokens, advantages = example.load_tokens()
                    if len(tokens) > cfg.max_length:
                        tokens = tokens[: cfg.max_length]
                        advantages = advantages[: cfg.max_length]
                    key = example.problem_id
                    if epoch == 0:
                        ref_logprobs = None
                        prev_logprobs = None
                    else:
                        ref_logprobs = all_ref_logprobs[key][: len(tokens) - 1]
                        prev_logprobs = all_prev_logprobs[key][: len(tokens) - 1]
                    datum = build_datum(
                        tokens,
                        advantages,
                        ref_logprobs,
                        prev_logprobs,
                        epoch,
                        cfg.loss_config,
                    )
                    data.append(datum)
                    valid_examples.append(example)
                    target_masks.append(advantages[1:])
                    batch_tokens.append(tokens)
                    batch_masks.append(advantages)

                if not data:
                    continue

                batch_time = time.time()

                lr = cfg.lr_schedule.get_lr(step, total_steps, epoch, cfg.num_epochs)

                # Forward-backward + optimizer step
                fwd_bwd_future = await training_client.forward_backward_async(
                    data,
                    loss_fn=cfg.loss_config.name,
                    loss_fn_config=loss_fn_config,
                    micro_batch_size=cfg.micro_batch_size,
                )
                optim_future = await training_client.optim_step_async(
                    cfg.adam_config.to_adam_params(lr)
                )

                fwd_bwd_result = await fwd_bwd_future.result_async()
                optim_result = await optim_future.result_async()

                # Extract logprobs
                logprobs_list = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]

                elapsed = time.time() - batch_time
                logger.info(
                    f"epoch={epoch} step={step}/{total_steps} "
                    f"lr={lr:.2e} n={len(data)} "
                    f"loss={cfg.loss_config.name} t={elapsed:.1f}s"
                )

                # Log metrics
                metrics_record: dict = {
                    "epoch": epoch,
                    "step": step,
                    "lr": lr,
                    "n": len(data),
                    "elapsed": round(elapsed, 2),
                    "time": datetime.now().strftime("%m-%d-%H-%M"),
                }
                metrics_record.update(
                    {f"fwd/{k}": v for k, v in fwd_bwd_result.metrics.items()}
                )
                if optim_result.metrics:
                    metrics_record.update(
                        {f"optim/{k}": v for k, v in optim_result.metrics.items()}
                    )
                # Compute loss per token by category for this step
                cat_loss: dict[str, float] = {}
                cat_tokens: dict[str, int] = {}
                for i, example in enumerate(valid_examples):
                    cat = example.category
                    lp_data = logprobs_list[i].data
                    loss_val = sum(-v for v, m in zip(lp_data, target_masks[i]) if m)
                    cat_loss[cat] = cat_loss.get(cat, 0.0) + loss_val
                    cat_tokens[cat] = (
                        cat_tokens.get(cat, 0) + example.unmasked_token_count
                    )
                for cat in sorted(cat_loss):
                    if cat_tokens[cat] > 0:
                        metrics_record[f"_loss_per_token/{cat}"] = (
                            cat_loss[cat] / cat_tokens[cat]
                        )
                total_tokens = sum(cat_tokens.values())
                if total_tokens > 0:
                    metrics_record["_loss_per_token"] = (
                        sum(cat_loss.values()) / total_tokens
                    )

                # Compute min logprob by category for this step (unmasked tokens only)
                cat_min_lp: dict[str, float] = {}
                for i, example in enumerate(valid_examples):
                    cat = example.category
                    lp_data = logprobs_list[i].data
                    unmasked_lps = [v for v, m in zip(lp_data, target_masks[i]) if m]
                    cat_min = min(unmasked_lps) if unmasked_lps else 0.0
                    if cat not in cat_min_lp or cat_min < cat_min_lp[cat]:
                        cat_min_lp[cat] = cat_min
                for cat in sorted(cat_min_lp):
                    metrics_record[f"_min_logprob/{cat}"] = round(cat_min_lp[cat], 4)

                metrics_file.write(json.dumps(metrics_record) + "\n")
                metrics_file.flush()

                # Save per-example logprobs
                for i, example in enumerate(valid_examples):
                    lp_data = logprobs_list[i].data

                    LogprobRecord(
                        problem_id=example.problem_id,
                        segment=example.segment,
                        logprobs=[round(v, 4) for v in lp_data],
                    ).save(epoch_dir)

                    # Collect ref logprobs during epoch 0
                    key = example.problem_id
                    if epoch == 0:
                        all_ref_logprobs[key] = list(lp_data)
                        # Save token IDs once
                        tokens_dir = log_path / "tokens"
                        stem = example.segment.removesuffix(".jsonl")
                        tokens_path = tokens_dir / example.problem_id / f"{stem}.json"
                        tokens_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(tokens_path, "w") as tf:
                            json.dump(
                                {"tokens": batch_tokens[i], "mask": batch_masks[i]}, tf
                            )
                    all_prev_logprobs[key] = list(lp_data)

                    unmasked_lps_i = [v for v, m in zip(lp_data, target_masks[i]) if m]
                    record = IndexRecord(
                        epoch=epoch,
                        step=step,
                        problem_id=example.problem_id,
                        segment=example.segment,
                        category=example.category,
                        num_loss_tokens=example.unmasked_token_count,
                        total_loss=round(sum(-v for v in unmasked_lps_i), 4),
                        min_logprob=round(min(unmasked_lps_i), 4)
                        if unmasked_lps_i
                        else 0.0,
                    )
                    index_file.write(json.dumps(dataclasses.asdict(record)) + "\n")

                step += 1

            epoch_elapsed = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_elapsed:.1f}s")

            # Compute and save per-epoch loss metrics
            epoch_metrics = compute_epoch_metrics(
                loss_config=cfg.loss_config,
                examples=examples,
                all_ref_logprobs=all_ref_logprobs,
                all_epoch_logprobs=all_prev_logprobs,
                epoch=epoch,
            )
            loss_record = {"epoch": epoch, "metrics": epoch_metrics}
            loss_file.write(json.dumps(loss_record) + "\n")
            loss_file.flush()

    # Save final checkpoint
    await training_client.save_checkpoint_async(
        name="final",
        log_path=str(log_path),
    )

    logger.info(f"Training completed. Logprobs saved to {logprob_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
