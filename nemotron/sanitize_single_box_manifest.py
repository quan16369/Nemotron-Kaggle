"""Rewrite a tokenized training manifest so each completion has one boxed answer.

The manifest stores token IDs, so source-level reasoner fixes do not change an
already-exported CSV. This utility decodes the completion side of each row,
normalizes boxed markup in the reasoning before `</think>`, then re-encodes it.
The final `\boxed{answer}` after `</think>` is left intact.

Usage:
    uv run python sanitize_single_box_manifest.py \
      --input ../winning_snapshot_delta_manifest.csv \
      --output ../winning_snapshot_delta_manifest_single_box.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tokenizers import Tokenizer

from reasoning import normalize_reasoning_for_single_box
from reasoners.cipher import reasoning_cipher
from reasoners.cryptarithm import reasoning_cryptarithm
from reasoners.store_types import Problem

BASE_DIR = Path(__file__).parent
DEFAULT_INPUT = BASE_DIR.parent / "winning_snapshot_delta_manifest.csv"
DEFAULT_OUTPUT = BASE_DIR.parent / "winning_snapshot_delta_manifest_single_box.csv"
DEFAULT_TOKENIZER = BASE_DIR / "tokenizer.json"
PROBLEMS_DIR = BASE_DIR / "problems"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--compact-cipher",
        action="store_true",
        help="Replace cipher completions with the current compact cipher reasoner.",
    )
    parser.add_argument(
        "--compact-cryptarithm",
        action="store_true",
        help=(
            "Replace concat/reverse-concat cryptarithm completions with the "
            "current slot-copy cryptarithm reasoner."
        ),
    )
    return parser.parse_args()


def sanitize_completion_text(completion_text: str) -> str:
    if "</think>" not in completion_text:
        return normalize_reasoning_for_single_box(completion_text)

    reasoning_text, final_text = completion_text.split("</think>", 1)
    reasoning_text = normalize_reasoning_for_single_box(reasoning_text)
    return f"{reasoning_text}\n</think>{final_text}"


def load_problem(problem_id: str) -> Problem:
    with (PROBLEMS_DIR / f"{problem_id}.jsonl").open() as f:
        payload = json.loads(f.readline())
    return Problem.from_payload(payload)


def compact_cipher_completion(row: dict[str, str]) -> str:
    problem = load_problem(row["source_problem_id"])
    reasoning_text = reasoning_cipher(problem)
    if reasoning_text is None:
        raise ValueError(f"could not build compact cipher reasoning for {problem.id}")
    reasoning_text = normalize_reasoning_for_single_box(reasoning_text)
    return f"{reasoning_text}\n</think>\n\\boxed{{{problem.answer}}}<|im_end|>"


def compact_cryptarithm_completion(row: dict[str, str]) -> str | None:
    problem = load_problem(row["source_problem_id"])
    reasoning_text = reasoning_cryptarithm(problem)
    if reasoning_text is None:
        return None
    reasoning_text = normalize_reasoning_for_single_box(reasoning_text)
    return f"{reasoning_text}\n</think>\n\\boxed{{{problem.answer}}}<|im_end|>"


def rewrite_row(
    row: dict[str, str],
    tokenizer: Tokenizer,
    *,
    compact_cipher: bool,
    compact_cryptarithm: bool,
) -> tuple[dict[str, str], bool, int, int, str]:
    input_ids = json.loads(row["input_ids_json"])
    mask = json.loads(row["mask_json"])
    if len(input_ids) != len(mask):
        raise ValueError(f"length mismatch for {row['problem_id']}")

    try:
        completion_start = mask.index(1)
    except ValueError:
        return row, False, 0, 0

    if any(mask[:completion_start]) or any(m != 1 for m in mask[completion_start:]):
        raise ValueError(f"non-contiguous completion mask for {row['problem_id']}")

    prompt_ids = input_ids[:completion_start]
    completion_ids = input_ids[completion_start:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
    old_box_count = completion_text.count("\\boxed{")

    compact_kind = ""
    if compact_cipher and row["category"] == "cipher":
        sanitized_text = compact_cipher_completion(row)
        compact_kind = "cipher"
    elif compact_cryptarithm and row["category"].startswith("cryptarithm_"):
        compact_completion = compact_cryptarithm_completion(row)
        if compact_completion is None:
            sanitized_text = sanitize_completion_text(completion_text)
        else:
            sanitized_text = compact_completion
            compact_kind = "cryptarithm"
    else:
        sanitized_text = sanitize_completion_text(completion_text)
    new_box_count = sanitized_text.count("\\boxed{")
    if sanitized_text == completion_text:
        return row, False, old_box_count, new_box_count, compact_kind

    new_completion_ids = tokenizer.encode(
        sanitized_text,
        add_special_tokens=False,
    ).ids
    new_input_ids = prompt_ids + new_completion_ids
    new_mask = [0] * len(prompt_ids) + [1] * len(new_completion_ids)

    updated = dict(row)
    updated["input_ids_json"] = json.dumps(new_input_ids, separators=(",", ":"))
    updated["mask_json"] = json.dumps(new_mask, separators=(",", ":"))
    updated["num_loss_tokens"] = str(len(new_completion_ids))
    updated["completion_token_count"] = str(len(new_completion_ids))
    updated["token_count"] = str(len(new_input_ids))
    return updated, True, old_box_count, new_box_count, compact_kind


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    rows = 0
    changed = 0
    rows_not_one_box = 0
    total_boxes_before = 0
    total_boxes_after = 0
    compact_cipher_rows = 0
    compact_cryptarithm_rows = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="") as src, args.output.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError("input manifest has no header")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            rows += 1
            updated, row_changed, old_boxes, new_boxes, compact_kind = rewrite_row(
                row,
                tokenizer,
                compact_cipher=args.compact_cipher,
                compact_cryptarithm=args.compact_cryptarithm,
            )
            changed += int(row_changed)
            compact_cipher_rows += int(compact_kind == "cipher")
            compact_cryptarithm_rows += int(compact_kind == "cryptarithm")
            total_boxes_before += old_boxes
            total_boxes_after += new_boxes
            if new_boxes != 1:
                rows_not_one_box += 1
            writer.writerow(updated)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": rows,
                "changed_rows": changed,
                "compact_cipher_rows": compact_cipher_rows,
                "compact_cryptarithm_rows": compact_cryptarithm_rows,
                "total_boxes_before": total_boxes_before,
                "total_boxes_after": total_boxes_after,
                "rows_not_one_box_after": rows_not_one_box,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
