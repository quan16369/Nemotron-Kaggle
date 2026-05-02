"""Export the full training dataset with generated reasoning into a CSV file.

Usage:
    python export_reasoning_csv.py
    python export_reasoning_csv.py --output reasoning_dataset.csv
    python export_reasoning_csv.py --compact

The export is keyed by `train.csv` order and merges in:
- `problems.jsonl` metadata (`category`, `status`, `submission`)
- `problems/<id>.jsonl` details (`question`, `examples`)
- `reasoning/<id>.txt` generated traces (or regenerated with --compact)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reasoning import extract_answer, normalize_reasoning_for_single_box
from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.store_types import Problem

BASE_DIR = Path(__file__).parent
TRAIN_CSV = BASE_DIR / "train.csv"
PROBLEMS_INDEX = BASE_DIR / "problems.jsonl"
PROBLEMS_DIR = BASE_DIR / "problems"
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


def load_problem_detail(problem_id: str) -> dict:
    path = PROBLEMS_DIR / f"{problem_id}.jsonl"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.loads(f.readline())


def load_reasoning(problem_id: str, category: str = "", *, compact: bool = False) -> str:
    # For bit_manipulation with compact flag, regenerate instead of loading from file
    if compact and category == "bit_manipulation":
        try:
            problem = Problem.load_from_json(problem_id)
            reasoning_text = reasoning_bit_manipulation(problem, compact=True)
            return reasoning_text or ""
        except Exception:
            # Fallback to file-based loading on error
            pass
    
    path = REASONING_DIR / f"{problem_id}.txt"
    if not path.exists():
        return ""
    return path.read_text()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "reasoning_dataset.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Regenerate bit_manipulation reasoning in compact format",
    )
    args = parser.parse_args()

    problems = load_problem_metadata()
    output_path = args.output

    fieldnames = [
        "id",
        "category",
        "status",
        "submission",
        "prompt",
        "answer",
        "question",
        "num_examples",
        "examples_json",
        "has_reasoning",
        "reasoning_answer",
        "reasoning",
    ]

    rows_written = 0
    with TRAIN_CSV.open() as src, output_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            problem_id = row["id"]
            metadata = problems.get(problem_id, {})
            detail = load_problem_detail(problem_id)
            category = metadata.get("category", detail.get("category", ""))
            reasoning_text = load_reasoning(problem_id, category, compact=args.compact)
            reasoning_answer = extract_answer(reasoning_text)
            reasoning_text = normalize_reasoning_for_single_box(reasoning_text)
            examples = detail.get("examples", [])

            writer.writerow(
                {
                    "id": problem_id,
                    "category": category,
                    "status": metadata.get("status", ""),
                    "submission": metadata.get("submission", ""),
                    "prompt": row.get("prompt", detail.get("prompt", "")),
                    "answer": row.get("answer", detail.get("answer", "")),
                    "question": detail.get("question", ""),
                    "num_examples": len(examples),
                    "examples_json": json.dumps(examples, ensure_ascii=True),
                    "has_reasoning": bool(reasoning_text),
                    "reasoning_answer": reasoning_answer,
                    "reasoning": reasoning_text,
                }
            )
            rows_written += 1

    print(f"Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
