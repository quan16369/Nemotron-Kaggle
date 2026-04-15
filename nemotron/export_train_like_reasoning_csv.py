"""Export a train-like CSV with generated reasoning.

Output schema keeps the original training columns first:
    id,prompt,answer,generated_cot,...

By default this regenerates reasoning with the current solver code so the CSV
reflects the current state of `reasoners/*.py`, instead of relying on possibly
stale `reasoning/*.txt` files.

Usage:
    python export_train_like_reasoning_csv.py
    python export_train_like_reasoning_csv.py --output train_with_reasoning.csv
    python export_train_like_reasoning_csv.py --use-existing-files
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reasoners.store_types import Problem
from reasoning import GENERATORS, compare_answer, extract_answer

BASE_DIR = Path(__file__).parent
TRAIN_CSV = BASE_DIR / "train.csv"
PROBLEMS_INDEX = BASE_DIR / "problems.jsonl"
REASONING_DIR = BASE_DIR / "reasoning"


def load_problem_metadata() -> dict[str, dict]:
    problems: dict[str, dict] = {}
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            problems[entry["id"]] = entry
    return problems


def load_reasoning_from_file(problem_id: str) -> str:
    path = REASONING_DIR / f"{problem_id}.txt"
    if not path.exists():
        return ""
    return path.read_text()


def generate_reasoning(problem_id: str, category: str, use_existing_files: bool) -> str:
    if use_existing_files:
        return load_reasoning_from_file(problem_id)

    generator = GENERATORS.get(category)
    if generator is None:
        return ""

    problem = Problem.load_from_json(problem_id)
    reasoning_text = generator(problem)
    return reasoning_text or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "train_with_reasoning.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--use-existing-files",
        action="store_true",
        help="Read reasoning/<id>.txt instead of regenerating with current code",
    )
    args = parser.parse_args()

    problem_metadata = load_problem_metadata()
    fieldnames = [
        "id",
        "prompt",
        "answer",
        "generated_cot",
        "category",
        "status",
        "submission",
        "reasoning_answer",
        "reasoning_is_correct",
    ]

    rows_written = 0
    with TRAIN_CSV.open() as src, args.output.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            problem_id = row["id"]
            meta = problem_metadata.get(problem_id, {})
            category = meta.get("category", "")
            reasoning_text = generate_reasoning(
                problem_id,
                category,
                use_existing_files=args.use_existing_files,
            )
            reasoning_answer = extract_answer(reasoning_text) if reasoning_text else ""
            writer.writerow(
                {
                    "id": problem_id,
                    "prompt": row["prompt"],
                    "answer": row["answer"],
                    "generated_cot": reasoning_text,
                    "category": category,
                    "status": meta.get("status", ""),
                    "submission": meta.get("submission", ""),
                    "reasoning_answer": reasoning_answer,
                    "reasoning_is_correct": (
                        compare_answer(row["answer"], reasoning_answer)
                        if reasoning_answer
                        else False
                    ),
                }
            )
            rows_written += 1
            if rows_written % 500 == 0:
                print(f"Processed {rows_written} rows...")

    print(f"Wrote {rows_written} rows to {args.output}")


if __name__ == "__main__":
    main()
