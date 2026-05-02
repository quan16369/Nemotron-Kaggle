"""Export the effective SFT training dataset to a single CSV file.

This follows the same inclusion logic used by the notebook / corpus pipeline:
- `choose_entry_to_include(status, category)`
- deterministic downsampling with `_keep_by_hash`
- priority duplicates (`-p0`)
- category duplicates (`-d{n}`)
- augmentations/*.txt

The output is row-oriented and suitable for inspection or for CSV-based
training scripts. Each row contains:
- the plain prompt
- the generated chain-of-thought (`generated_cot`)
- the exact completion text used for training
- metadata about duplication / augmentation / category
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from tokenizers import Tokenizer  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional local dependency
    Tokenizer = None  # type: ignore[assignment]

from reasoning import (
    GENERATORS,
    compare_answer,
    extract_answer,
    normalize_reasoning_for_single_box,
)
from reasoners.store_types import Problem

BASE_DIR = Path(__file__).parent
TRAIN_CSV = BASE_DIR / "train.csv"
PROBLEMS_INDEX = BASE_DIR / "problems.jsonl"
PROBLEMS_DIR = BASE_DIR / "problems"
REASONING_DIR = BASE_DIR / "reasoning"
AUGMENTATIONS_DIR = BASE_DIR / "augmentations"
TOKENIZER_JSON_PATH = BASE_DIR / "tokenizer.json"
PRIORITY_IDS_PATH = BASE_DIR / "investigators" / "priority_problem_ids.txt"
REASONING_TOKEN_BUDGET = 7680

# Keep these definitions local so the export works even when optional training
# dependencies are unavailable in the current environment.
DOWNSAMPLE_RATES: dict[str, float] = {
    "numeral": 0.4,
    "gravity": 0.6,
    "unit_conversion": 0.6,
}

DUPLICATE_COUNTS: dict[str, int] = {
    "cryptarithm_deduce": 1,
}


def _keep_by_hash(problem_id: str, rate: float) -> bool:
    h = int(hashlib.sha256((problem_id + str(rate)).encode()).hexdigest(), 16) + int(
        10000 * rate
    )
    return (h % 10000) < int(rate * 10000)


def choose_entry_to_include(problem_status: str, category: str) -> bool:
    if category.endswith("_guess"):
        return True
    return problem_status == "rule_found"


def load_problem_metadata() -> dict[str, dict]:
    problems: dict[str, dict] = {}
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            problems[entry["id"]] = entry
    return problems


def load_problem_detail(problem_id: str) -> dict:
    path = PROBLEMS_DIR / f"{problem_id}.jsonl"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.loads(f.readline())


def load_reasoning_from_file(problem_id: str) -> str:
    path = REASONING_DIR / f"{problem_id}.txt"
    if not path.exists():
        return ""
    return path.read_text().rstrip("\n")


def get_reasoning_text(problem_id: str, category: str, use_existing_files: bool) -> tuple[str, str]:
    if use_existing_files:
        text = load_reasoning_from_file(problem_id)
        if text:
            return text, "file"

    generator = GENERATORS.get(category)
    if generator is None:
        return "", "missing"

    text = generator(Problem.load_from_json(problem_id)) or ""
    return text.rstrip("\n"), "generated"


def load_priority_ids() -> set[str]:
    if not PRIORITY_IDS_PATH.exists():
        return set()
    return {
        line.strip()
        for line in PRIORITY_IDS_PATH.read_text().splitlines()
        if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "effective_training_dataset.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--regenerate-reasoning",
        action="store_true",
        help="Regenerate reasoning with current solver code instead of preferring reasoning/*.txt",
    )
    args = parser.parse_args()

    completion_tokenizer = (
        Tokenizer.from_file(str(TOKENIZER_JSON_PATH)) if Tokenizer is not None else None
    )
    problem_metadata = load_problem_metadata()
    priority_ids = load_priority_ids()

    def encode_completion_ids(text: str) -> list[int]:
        if completion_tokenizer is None:
            return []
        return completion_tokenizer.encode(text, add_special_tokens=False).ids

    fieldnames = [
        "id",
        "source_problem_id",
        "sample_kind",
        "duplicate_index",
        "is_augmentation",
        "category",
        "status",
        "submission",
        "prompt",
        "answer",
        "reasoning_answer",
        "reasoning_is_correct",
        "generated_cot",
        "completion_text",
        "completion_token_count",
        "completion_over_7680",
        "reasoning_source",
        "question",
        "num_examples",
        "examples_json",
    ]

    with TRAIN_CSV.open(newline="") as f:
        prompt_rows = {row["id"]: row for row in csv.DictReader(f)}

    rows: list[dict] = []

    problem_ids = sorted(
        problem_id
        for problem_id in problem_metadata
        if problem_id in prompt_rows
        and (
            args.regenerate_reasoning
            or (REASONING_DIR / f"{problem_id}.txt").exists()
        )
    )

    for problem_id in problem_ids:
        row = prompt_rows[problem_id]
        prompt_text = row["prompt"]
        answer = str(row["answer"])
        meta = problem_metadata.get(problem_id, {})
        category = meta.get("category", "")
        status = meta.get("status", "")
        submission = meta.get("submission", "")
        detail = load_problem_detail(problem_id)

        if not choose_entry_to_include(status, category):
            continue

        rate = DOWNSAMPLE_RATES.get(category, 1.0)
        if rate < 1.0 and problem_id not in priority_ids and not _keep_by_hash(problem_id, rate):
            continue

        reasoning_text, reasoning_source = get_reasoning_text(
            problem_id,
            category,
            use_existing_files=not args.regenerate_reasoning,
        )
        if not reasoning_text:
            continue

        reasoning_answer = extract_answer(reasoning_text) or answer
        normalized_reasoning_text = normalize_reasoning_for_single_box(reasoning_text)
        completion_text = (
            f"{normalized_reasoning_text}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
        )
        completion_ids = encode_completion_ids(completion_text)
        examples = detail.get("examples", [])

        rows.append(
            {
                "id": problem_id,
                "source_problem_id": problem_id,
                "sample_kind": "base",
                "duplicate_index": "",
                "is_augmentation": False,
                "category": category,
                "status": status,
                "submission": submission,
                "prompt": prompt_text,
                "answer": answer,
                "reasoning_answer": reasoning_answer,
                "reasoning_is_correct": compare_answer(answer, reasoning_answer),
                "generated_cot": normalized_reasoning_text,
                "completion_text": completion_text,
                "completion_token_count": len(completion_ids) if completion_ids else "",
                "completion_over_7680": (
                    len(completion_ids) > REASONING_TOKEN_BUDGET
                    if completion_ids
                    else ""
                ),
                "reasoning_source": reasoning_source,
                "question": detail.get("question", ""),
                "num_examples": len(examples),
                "examples_json": json.dumps(examples, ensure_ascii=True),
            }
        )

    if AUGMENTATIONS_DIR.exists():
        for aug_path in sorted(AUGMENTATIONS_DIR.glob("*.txt")):
            text = aug_path.read_text()
            category = text.split("[category]\n", 1)[1].split("\n[prompt]\n", 1)[0]
            prompt_text = text.split("[prompt]\n", 1)[1].split("\n[completion]\n", 1)[0]
            completion = text.split("\n[completion]\n", 1)[1].rstrip("\n")
            completion_text = f"{completion}\n</think><|im_end|>"
            completion_ids = encode_completion_ids(completion_text)

            rows.append(
                {
                    "id": aug_path.stem,
                    "source_problem_id": aug_path.stem,
                    "sample_kind": "augmentation",
                    "duplicate_index": "",
                    "is_augmentation": True,
                    "category": category,
                    "status": "",
                    "submission": "",
                    "prompt": prompt_text,
                    "answer": completion,
                    "reasoning_answer": "",
                    "reasoning_is_correct": "",
                    "generated_cot": completion,
                    "completion_text": completion_text,
                    "completion_token_count": len(completion_ids) if completion_ids else "",
                    "completion_over_7680": (
                        len(completion_ids) > REASONING_TOKEN_BUDGET
                        if completion_ids
                        else ""
                    ),
                    "reasoning_source": "augmentation",
                    "question": "",
                    "num_examples": 0,
                    "examples_json": "[]",
                }
            )

    def duplicate_row(row: dict, suffix: str, sample_kind: str, duplicate_index: int | str) -> dict:
        copied = dict(row)
        copied["id"] = f"{row['id']}{suffix}"
        copied["sample_kind"] = sample_kind
        copied["duplicate_index"] = duplicate_index
        return copied

    priority_dups = [
        duplicate_row(row, "-p0", "priority_dup", 0)
        for row in rows
        if row["id"] in priority_ids
    ]
    if priority_dups:
        rows.extend(priority_dups)

    category_dups = [
        duplicate_row(row, f"-d{dup_idx}", "category_dup", dup_idx)
        for row in rows
        if "-p0" not in row["id"]
        for dup_idx in range(DUPLICATE_COUNTS.get(row["category"], 0))
    ]
    if category_dups:
        rows.extend(category_dups)

    rows.sort(key=lambda row: row["id"])

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    base_count = sum(row["sample_kind"] == "base" for row in rows)
    priority_count = sum(row["sample_kind"] == "priority_dup" for row in rows)
    dup_count = sum(row["sample_kind"] == "category_dup" for row in rows)
    aug_count = sum(row["sample_kind"] == "augmentation" for row in rows)
    over_budget = sum(bool(row["completion_over_7680"]) for row in rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    if completion_tokenizer is None:
        print("Tokenizer library not installed locally. Token-count columns were left blank.")
    print(
        {
            "base": base_count,
            "priority_dup": priority_count,
            "category_dup": dup_count,
            "augmentation": aug_count,
            "completion_over_7680": over_budget,
        }
    )


if __name__ == "__main__":
    main()
