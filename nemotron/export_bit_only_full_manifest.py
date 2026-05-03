"""Export a bit-manipulation-only manifest covering all 1602 source problems.

This exporter is stricter than the old corpus path:
- one row per source problem (no duplicates)
- prefer richer existing reasoning when it is already correct
- fall back to the current whole-word solver
- then try a compact legacy trace
- finally emit a small gold-answer fallback trace so coverage stays at 1602/1602

The output format matches `winning_snapshot_delta_manifest.csv`, so the Kaggle
notebook can consume it directly.

Usage:
    uv run export_bit_only_full_manifest.py \
      --output /home/quan/nemotron-kaggle/bit_only_full_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from reasoning import extract_answer
from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.store_types import Problem

BASE_DIR = Path(__file__).parent
TRAIN_CSV = BASE_DIR / "train.csv"
PROBLEMS_INDEX = BASE_DIR / "problems.jsonl"
REASONING_DIR = BASE_DIR / "reasoning"
TOKENIZER_JSON = BASE_DIR / "tokenizer.json"
DEFAULT_OUTPUT = BASE_DIR.parent / "bit_only_full_manifest.csv"

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
PROMPT_PREFIX = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
PROMPT_MIDDLE = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
MAX_SEQ_LEN = 8192
MAX_COMPLETION_TOKENS = 7680

FALLBACK_TEMPLATE = (
    "I will return the final answer inside \\boxed{{}}.\n"
    "The output bits for the target input are {answer}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--exclude-fallback",
        action="store_true",
        help="Drop problem rows that cannot be solved by an existing/modern/legacy solver trace.",
    )
    return parser.parse_args()


def load_problem_rows() -> tuple[list[str], dict[str, dict[str, str]]]:
    with PROBLEMS_INDEX.open() as f:
        problem_meta = {
            row["id"]: row
            for row in (json.loads(line) for line in f if line.strip())
            if row["category"] == "bit_manipulation"
        }
    with TRAIN_CSV.open(newline="") as f:
        prompt_rows = {row["id"]: row for row in csv.DictReader(f) if row["id"] in problem_meta}
    bit_ids = sorted(problem_meta)
    return bit_ids, prompt_rows


def tokenize_prompt(prompt: str, tokenizer: Tokenizer) -> list[int]:
    prompt_text = PROMPT_PREFIX + prompt + PROMPT_SUFFIX + PROMPT_MIDDLE
    return tokenizer.encode(prompt_text, add_special_tokens=False).ids


def build_completion_tokens(reasoning_text: str, answer: str, tokenizer: Tokenizer) -> list[int]:
    completion_text = f"{reasoning_text}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
    return tokenizer.encode(completion_text, add_special_tokens=False).ids


def build_record(
    *,
    problem_id: str,
    prompt: str,
    answer: str,
    reasoning_text: str,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    prompt_ids = tokenize_prompt(prompt, tokenizer)
    completion_ids = build_completion_tokens(reasoning_text, answer, tokenizer)
    if len(completion_ids) > MAX_COMPLETION_TOKENS:
        raise ValueError(
            f"Example {problem_id} completion exceeds budget: "
            f"{len(completion_ids)} > {MAX_COMPLETION_TOKENS}"
        )
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > MAX_SEQ_LEN:
        raise ValueError(
            f"Example {problem_id} exceeds max length: {len(input_ids)} > {MAX_SEQ_LEN}"
        )
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return {
        "problem_id": problem_id,
        "source_problem_id": problem_id,
        "category": "bit_manipulation",
        "segment": "synthetic.jsonl",
        "num_loss_tokens": len(completion_ids),
        "completion_token_count": len(completion_ids),
        "token_count": len(input_ids),
        "input_ids_json": json.dumps(input_ids),
        "mask_json": json.dumps(mask),
    }


def get_existing_reasoning(problem_id: str, answer: str) -> str | None:
    path = REASONING_DIR / f"{problem_id}.txt"
    if not path.exists():
        return None
    reasoning_text = path.read_text().strip("\n")
    return reasoning_text if extract_answer(reasoning_text) == answer else None


def get_generated_reasoning(
    problem: Problem,
    answer: str,
    *,
    compact: bool,
    allow_whole_word: bool,
) -> str | None:
    reasoning_text = reasoning_bit_manipulation(
        problem,
        compact=compact,
        enable_three_bit_repair=False,
        allow_whole_word=allow_whole_word,
    )
    if not reasoning_text:
        return None
    return reasoning_text if extract_answer(reasoning_text) == answer else None


def get_fallback_reasoning(answer: str) -> str:
    return FALLBACK_TEMPLATE.format(answer=answer)


def select_reasoning(problem_id: str, answer: str) -> tuple[str, str]:
    problem = Problem.load_from_json(problem_id)

    existing = get_existing_reasoning(problem_id, answer)
    if existing is not None:
        return existing, "existing"

    modern = get_generated_reasoning(
        problem,
        answer,
        compact=False,
        allow_whole_word=True,
    )
    if modern is not None:
        return modern, "modern"

    legacy_compact = get_generated_reasoning(
        problem,
        answer,
        compact=True,
        allow_whole_word=False,
    )
    if legacy_compact is not None:
        return legacy_compact, "legacy_compact"

    return get_fallback_reasoning(answer), "fallback"


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    bit_ids, prompt_rows = load_problem_rows()

    fieldnames = [
        "problem_id",
        "source_problem_id",
        "category",
        "segment",
        "num_loss_tokens",
        "completion_token_count",
        "token_count",
        "input_ids_json",
        "mask_json",
    ]

    stats = Counter()
    fallback_problem_ids: list[str] = []
    max_token_count = 0
    max_completion_token_count = 0
    records: list[dict[str, Any]] = []

    for problem_id in bit_ids:
        row = prompt_rows[problem_id]
        answer = str(row["answer"])
        reasoning_text, source = select_reasoning(problem_id, answer)
        if args.exclude_fallback and source == "fallback":
            continue
        try:
            record = build_record(
                problem_id=problem_id,
                prompt=row["prompt"],
                answer=answer,
                reasoning_text=reasoning_text,
                tokenizer=tokenizer,
            )
        except ValueError:
            if source != "fallback":
                reasoning_text = get_fallback_reasoning(answer)
                if args.exclude_fallback:
                    continue
                record = build_record(
                    problem_id=problem_id,
                    prompt=row["prompt"],
                    answer=answer,
                    reasoning_text=reasoning_text,
                    tokenizer=tokenizer,
                )
                source = "fallback"
            else:
                raise

        stats[source] += 1
        if source == "fallback":
            fallback_problem_ids.append(problem_id)
        max_token_count = max(max_token_count, int(record["token_count"]))
        max_completion_token_count = max(
            max_completion_token_count,
            int(record["completion_token_count"]),
        )
        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "unique_source_problem_ids": len({r["source_problem_id"] for r in records}),
                "source_breakdown": dict(stats),
                "fallback_problem_ids": fallback_problem_ids,
                "max_token_count": max_token_count,
                "max_completion_token_count": max_completion_token_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
