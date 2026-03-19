from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
FINAL_PATTERNS = [
    re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+)$", re.IGNORECASE),
    re.compile(r"(?:thus|therefore)\s*,?\s*(.+)$", re.IGNORECASE),
]


def read_records(path: str) -> List[Dict[str, str]]:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".csv":
        with open(path_obj, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    records: List[Dict[str, str]] = []
    with open(path_obj, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_boxed(text: str) -> Optional[str]:
    marker = r"\boxed{"
    starts = [idx for idx in range(len(text)) if text.startswith(marker, idx)]
    if not starts:
        return None

    start = starts[-1] + len(marker)
    depth = 1
    chars: List[str] = []

    for ch in text[start:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(ch)
    return None


def normalize_answer(text: str) -> str:
    text = text.strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text).strip().rstrip(".")


def try_parse_number(text: str) -> Optional[float]:
    cleaned = normalize_answer(text).replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_prediction(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed:
        return normalize_answer(boxed)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        for pattern in FINAL_PATTERNS:
            match = pattern.search(line)
            if match:
                return normalize_answer(match.group(1))

    numbers = NUMERIC_RE.findall(text)
    if numbers:
        return normalize_answer(numbers[-1])

    if lines:
        return normalize_answer(lines[-1])
    return ""


def answers_match(
    prediction: str,
    truth: str,
    rel_tol: float = 1e-4,
    abs_tol: float = 0.0,
) -> bool:
    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(truth)

    if pred_norm == truth_norm:
        return True

    pred_num = try_parse_number(pred_norm)
    truth_num = try_parse_number(truth_norm)
    if pred_num is None or truth_num is None:
        return False

    return math.isclose(pred_num, truth_num, rel_tol=rel_tol, abs_tol=abs_tol)


def build_truth_map(records: Iterable[Dict[str, str]], id_column: str, answer_column: str) -> Dict[str, str]:
    return {str(record[id_column]): str(record[answer_column]) for record in records}


def score_predictions(
    truth_records: Iterable[Dict[str, str]],
    pred_records: Iterable[Dict[str, str]],
    id_column: str,
    answer_column: str,
    prediction_column: str,
    rel_tol: float,
    abs_tol: float,
) -> Tuple[float, List[Dict[str, str]]]:
    truth_map = build_truth_map(truth_records, id_column=id_column, answer_column=answer_column)
    pred_map = {str(record[id_column]): str(record[prediction_column]) for record in pred_records}

    mismatches: List[Dict[str, str]] = []
    correct = 0

    for sample_id, truth in truth_map.items():
        raw_prediction = pred_map.get(sample_id, "")
        extracted_prediction = extract_prediction(raw_prediction)
        matched = answers_match(
            extracted_prediction,
            truth,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        if matched:
            correct += 1
            continue

        mismatches.append(
            {
                "id": sample_id,
                "truth": truth,
                "raw_prediction": raw_prediction,
                "extracted_prediction": extracted_prediction,
            }
        )

    total = len(truth_map)
    accuracy = correct / total if total else 0.0
    return accuracy, mismatches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    score = sub.add_parser("score")
    score.add_argument("--truth", required=True)
    score.add_argument("--pred", required=True)
    score.add_argument("--id-column", default="id")
    score.add_argument("--answer-column", default="answer")
    score.add_argument("--prediction-column", default="prediction")
    score.add_argument("--rel-tol", type=float, default=1e-4)
    score.add_argument("--abs-tol", type=float, default=0.0)
    score.add_argument("--show-mismatches", type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd != "score":
        raise ValueError(f"Unsupported command: {args.cmd}")

    truth_records = read_records(args.truth)
    pred_records = read_records(args.pred)
    accuracy, mismatches = score_predictions(
        truth_records=truth_records,
        pred_records=pred_records,
        id_column=args.id_column,
        answer_column=args.answer_column,
        prediction_column=args.prediction_column,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )

    print(json.dumps(
        {
            "accuracy": accuracy,
            "total": len(truth_records),
            "wrong": len(mismatches),
            "rel_tol": args.rel_tol,
            "abs_tol": args.abs_tol,
        },
        indent=2,
    ))

    for mismatch in mismatches[: args.show_mismatches]:
        print(json.dumps(mismatch, ensure_ascii=False))


if __name__ == "__main__":
    main()
