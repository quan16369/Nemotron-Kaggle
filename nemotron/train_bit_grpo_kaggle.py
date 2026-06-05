"""Bit-manipulation-only GRPO tuning for Kaggle.

This is intentionally narrow: start from an SFT LoRA adapter, sample only
bit_manipulation prompts, reward exact 8-bit answers, and save a new LoRA.

Run from the repository root on Kaggle after installing the same offline
packages used by the SFT notebook.
"""

from __future__ import annotations

import argparse
import glob
import inspect
import math
import os
import random
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer


PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def extract_final_answer(text: str | None) -> str:
    if text is None:
        return "NOT_FOUND"
    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for i, match in enumerate(boxed_starts):
        start = match.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    matches = re.findall(r"[01]{8}", text)
    if matches:
        return matches[-1]
    matches = re.findall(r"[01]+", text)
    return matches[-1] if matches else "NOT_FOUND"


def find_train_csv(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True))
    valid: list[str] = []
    for path in candidates:
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            continue
        if {"id", "prompt", "answer"}.issubset(set(header)):
            valid.append(path)
    if not valid:
        raise FileNotFoundError("Could not find train.csv with id,prompt,answer")
    valid.sort(key=lambda p: ("nvidia" not in p.lower(), len(p)))
    return valid[0]


def is_bit_prompt(prompt: str) -> bool:
    p = str(prompt).lower()
    return "secret bit manipulation rule" in p and "8-bit binary" in p


def build_dataset(args: argparse.Namespace, tokenizer) -> HFDataset:
    train_csv = find_train_csv(args.train_csv)
    df = pd.read_csv(train_csv)
    df = df[df["prompt"].map(is_bit_prompt)].copy()
    df = df[df["answer"].astype(str).str.fullmatch(r"[01]{8}")].copy()
    df = df.sample(frac=1.0, random_state=args.sample_seed).reset_index(drop=True)
    if args.max_examples is not None:
        df = df.head(args.max_examples).copy()

    def format_prompt(prompt_text: str) -> str:
        user_content = str(prompt_text) + PROMPT_SUFFIX
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            return user_content + "\n"

    records = [
        {
            "id": str(row.id),
            "prompt": format_prompt(row.prompt),
            "answer": str(row.answer),
        }
        for row in df.itertuples(index=False)
    ]
    print(
        {
            "train_csv": train_csv,
            "bit_examples": len(records),
            "first_id": records[0]["id"] if records else None,
            "first_answer": records[0]["answer"] if records else None,
        }
    )
    return HFDataset.from_list(records)


def normalize_completion(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def make_reward_func(tokenizer, args: argparse.Namespace):
    def reward_func(prompts=None, completions=None, answer=None, **kwargs):
        rewards: list[float] = []
        answers = answer or kwargs.get("answers") or []
        for completion, gold in zip(completions, answers):
            text = normalize_completion(completion)
            extracted = extract_final_answer(text)
            gold = str(gold).strip()

            reward = 0.0
            if extracted == gold:
                reward += args.correct_reward
            elif re.fullmatch(r"[01]{8}", extracted):
                # valid binary but wrong: useful format, bad answer
                reward += args.valid_binary_wrong_reward
            else:
                reward -= args.invalid_answer_penalty

            if "\\boxed{" in text:
                reward += args.boxed_bonus
            else:
                reward -= args.missing_boxed_penalty

            if re.fullmatch(r"[01]{8}", extracted):
                reward += args.binary_format_bonus

            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            reward -= args.length_penalty * token_count
            if token_count > args.soft_length_budget:
                reward -= args.over_budget_penalty * (
                    token_count - args.soft_length_budget
                ) / 100.0
            rewards.append(float(reward))
        return rewards

    return reward_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--base-adapter-dir", required=True)
    parser.add_argument("--output-adapter-dir", default="/kaggle/working/bit_grpo_adapter")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--max-examples", type=int, default=800)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--correct-reward", type=float, default=2.0)
    parser.add_argument("--valid-binary-wrong-reward", type=float, default=-0.2)
    parser.add_argument("--invalid-answer-penalty", type=float, default=0.8)
    parser.add_argument("--boxed-bonus", type=float, default=0.1)
    parser.add_argument("--missing-boxed-penalty", type=float, default=0.2)
    parser.add_argument("--binary-format-bonus", type=float, default=0.1)
    parser.add_argument("--length-penalty", type=float, default=2e-5)
    parser.add_argument("--soft-length-budget", type=int, default=1800)
    parser.add_argument("--over-budget-penalty", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    try:
        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model_path,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False

    dataset = build_dataset(args, tokenizer)
    print("Loading trainable adapter:", args.base_adapter_dir)
    model = PeftModel.from_pretrained(model, args.base_adapter_dir, is_trainable=True)
    model.config.use_cache = False
    model.print_trainable_parameters()

    config_kwargs = dict(
        output_dir="/kaggle/working/bit_grpo_output",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        beta=args.beta,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )
    grpo_sig = inspect.signature(GRPOConfig)
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in grpo_sig.parameters}
    grpo_args = GRPOConfig(**config_kwargs)

    trainer_sig = inspect.signature(GRPOTrainer)
    trainer_kwargs = dict(
        model=model,
        args=grpo_args,
        reward_funcs=make_reward_func(tokenizer, args),
        train_dataset=dataset,
    )
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    if hasattr(trainer, "beta"):
        trainer.beta = float(args.beta)
    if hasattr(trainer.args, "beta"):
        trainer.args.beta = float(args.beta)
    if float(args.beta) == 0.0 and hasattr(trainer, "ref_model"):
        trainer.ref_model = None

    print(
        {
            "grpo_beta": getattr(trainer, "beta", None),
            "args_beta": getattr(trainer.args, "beta", None),
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
        }
    )
    result = trainer.train()
    print(result)

    output_dir = Path(args.output_adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved bit GRPO adapter to {output_dir}")


if __name__ == "__main__":
    main()
