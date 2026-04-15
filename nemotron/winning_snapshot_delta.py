"""Helpers for training from the winning snapshot plus safe current deltas."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase

from corpus import tokenize_prompt
from reasoning import GENERATORS, extract_answer
from reasoners.store_types import Problem

COMPETITION_CATEGORIES = {
    "bit_manipulation",
    "cipher",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "gravity",
    "numeral",
    "unit_conversion",
}


@dataclass
class DeltaStats:
    snapshot_examples: int
    snapshot_source_problems: int
    current_correct_records: int
    replaced_source_problems: int
    replaced_training_records: int
    added_source_problems: int
    added_training_records: int
    final_training_records: int


def _build_record(
    *,
    problem_id: str,
    source_problem_id: str,
    category: str,
    tokens: list[int],
    mask: list[int],
    max_seq_len: int,
) -> dict[str, Any]:
    if len(tokens) != len(mask):
        raise ValueError(
            f"Length mismatch for {problem_id}: {len(tokens)} tokens vs {len(mask)} mask"
        )
    if len(tokens) > max_seq_len:
        raise ValueError(
            f"Example {problem_id} exceeds max length: {len(tokens)} > {max_seq_len}"
        )
    labels = [token if m == 1 else -100 for token, m in zip(tokens, mask)]
    num_loss_tokens = sum(mask)
    if num_loss_tokens == 0:
        raise ValueError(f"Example {problem_id} has no unmasked tokens")
    return {
        "problem_id": problem_id,
        "source_problem_id": source_problem_id,
        "segment": "synthetic.jsonl",
        "category": category,
        "num_loss_tokens": num_loss_tokens,
        "input_ids": tokens,
        "attention_mask": [1] * len(tokens),
        "labels": labels,
        "completion_token_count": num_loss_tokens,
    }


def load_snapshot_records(
    snapshot_dir: Path,
    *,
    max_seq_len: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_path = snapshot_dir / "logprobs" / "index.jsonl"
    tokens_dir = snapshot_dir / "tokens"
    config_path = snapshot_dir / "config.json"

    if not index_path.exists() or not tokens_dir.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Snapshot is incomplete under {snapshot_dir}. "
            "Expected config.json, logprobs/index.jsonl, and tokens/."
        )

    records: list[dict[str, Any]] = []
    with open(index_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("epoch") != 0:
                continue
            problem_id = row["problem_id"]
            source_problem_id = problem_id.split("-", 1)[0]
            token_path = tokens_dir / problem_id / "synthetic.json"
            token_data = json.loads(token_path.read_text())
            records.append(
                _build_record(
                    problem_id=problem_id,
                    source_problem_id=source_problem_id,
                    category=row["category"],
                    tokens=token_data["tokens"],
                    mask=token_data["mask"],
                    max_seq_len=max_seq_len,
                )
            )

    records.sort(key=lambda record: record["problem_id"])
    return records, json.loads(config_path.read_text())


def _get_current_reasoning_text(
    problem_id: str,
    category: str,
    *,
    use_existing_reasoning_files: bool,
    reasoning_dir: Path,
) -> str:
    generator = GENERATORS.get(category)
    reasoning_text = ""
    if generator is not None:
        reasoning_text = (generator(Problem.load_from_json(problem_id)) or "").rstrip(
            "\n"
        )
    if not reasoning_text and use_existing_reasoning_files:
        path = reasoning_dir / f"{problem_id}.txt"
        if path.exists():
            reasoning_text = path.read_text().rstrip("\n")
    return reasoning_text


def build_current_correct_base_records(
    *,
    repo_dir: Path,
    chat_tokenizer: PreTrainedTokenizerBase,
    completion_tokenizer: Tokenizer,
    max_seq_len: int,
    use_existing_reasoning_files: bool,
) -> dict[str, dict[str, Any]]:
    train_csv_path = repo_dir / "train.csv"
    problems_index_path = repo_dir / "problems.jsonl"
    reasoning_dir = repo_dir / "reasoning"

    with open(problems_index_path) as f:
        problem_metadata = {
            row["id"]: row for row in (json.loads(line) for line in f if line.strip())
        }
    with open(train_csv_path, newline="") as f:
        prompt_rows = {row["id"]: row for row in csv.DictReader(f)}

    current_records: dict[str, dict[str, Any]] = {}
    for problem_id in sorted(problem_metadata):
        meta = problem_metadata[problem_id]
        category = meta["category"]
        if category not in COMPETITION_CATEGORIES:
            continue
        row = prompt_rows.get(problem_id)
        if row is None:
            continue

        reasoning_text = _get_current_reasoning_text(
            problem_id,
            category,
            use_existing_reasoning_files=use_existing_reasoning_files,
            reasoning_dir=reasoning_dir,
        )
        if not reasoning_text:
            continue

        answer = str(row["answer"])
        reasoning_answer = extract_answer(reasoning_text)
        if reasoning_answer != answer:
            continue

        prompt_ids = tokenize_prompt(row["prompt"], chat_tokenizer)
        completion_text = (
            f"{reasoning_text}\n</think>\n\\boxed{{{reasoning_answer}}}<|im_end|>"
        )
        completion_ids = completion_tokenizer.encode(
            completion_text, add_special_tokens=False
        ).ids
        tokens = prompt_ids + completion_ids
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

        current_records[problem_id] = _build_record(
            problem_id=problem_id,
            source_problem_id=problem_id,
            category=category,
            tokens=tokens,
            mask=mask,
            max_seq_len=max_seq_len,
        )

    return current_records


def merge_snapshot_with_current_delta(
    snapshot_records: list[dict[str, Any]],
    current_correct_base_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], DeltaStats]:
    snapshot_families: dict[str, list[dict[str, Any]]] = {}
    snapshot_base_records: dict[str, dict[str, Any]] = {}
    for record in snapshot_records:
        source_problem_id = record["source_problem_id"]
        snapshot_families.setdefault(source_problem_id, []).append(record)
        if record["problem_id"] == source_problem_id:
            snapshot_base_records[source_problem_id] = record

    replace_source_ids: set[str] = set()
    add_source_ids: set[str] = set()

    for source_problem_id, current_record in current_correct_base_records.items():
        snapshot_base = snapshot_base_records.get(source_problem_id)
        if snapshot_base is None:
            add_source_ids.add(source_problem_id)
            continue

        same_tokens = snapshot_base["input_ids"] == current_record["input_ids"]
        same_labels = snapshot_base["labels"] == current_record["labels"]
        same_category = snapshot_base["category"] == current_record["category"]
        if not (same_tokens and same_labels and same_category):
            replace_source_ids.add(source_problem_id)

    merged_records: list[dict[str, Any]] = [
        record
        for record in snapshot_records
        if record["source_problem_id"] not in replace_source_ids
    ]

    replaced_training_records = 0
    for source_problem_id in sorted(replace_source_ids):
        family = sorted(
            snapshot_families[source_problem_id], key=lambda record: record["problem_id"]
        )
        current_base = current_correct_base_records[source_problem_id]
        for old_record in family:
            cloned = dict(current_base)
            cloned["problem_id"] = old_record["problem_id"]
            merged_records.append(cloned)
        replaced_training_records += len(family)

    for source_problem_id in sorted(add_source_ids):
        merged_records.append(current_correct_base_records[source_problem_id])

    merged_records.sort(key=lambda record: record["problem_id"])
    stats = DeltaStats(
        snapshot_examples=len(snapshot_records),
        snapshot_source_problems=len(snapshot_families),
        current_correct_records=len(current_correct_base_records),
        replaced_source_problems=len(replace_source_ids),
        replaced_training_records=replaced_training_records,
        added_source_problems=len(add_source_ids),
        added_training_records=len(add_source_ids),
        final_training_records=len(merged_records),
    )
    return merged_records, stats


def summarize_categories(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(record["category"] for record in records)
