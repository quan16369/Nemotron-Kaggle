from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, TypedDict


Category = Literal[
    "bit_manipulation",
    "cipher",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "gravity",
    "numeral",
    "unit_conversion",
]


class Example(TypedDict):
    input_value: str
    output_value: str


class ProblemLike(TypedDict, total=False):
    id: str
    prompt: str
    question: str
    answer: str
    examples: list[Example]


_BINARY_8_RE = re.compile(r"^[01]{8}$")
_LOWER_TEXT_RE = re.compile(r"^[a-z ]+$")
_INT_RE = re.compile(r"^\d+$")
_DECIMAL_RE = re.compile(r"^\d+\.\d+$")
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$")
_EQUATION_NUMERIC_RE = re.compile(r"^\d+\D\d+$")


def _is_bit_manipulation(examples: list[Example], question: str) -> bool:
    return bool(examples) and all(
        _BINARY_8_RE.fullmatch(ex["input_value"])
        and _BINARY_8_RE.fullmatch(ex["output_value"])
        for ex in examples
    ) and bool(_BINARY_8_RE.fullmatch(question))


def _is_cipher(examples: list[Example], question: str) -> bool:
    return bool(examples) and all(
        _LOWER_TEXT_RE.fullmatch(ex["input_value"])
        and _LOWER_TEXT_RE.fullmatch(ex["output_value"])
        for ex in examples
    ) and bool(_LOWER_TEXT_RE.fullmatch(question))


def _is_numeral(examples: list[Example], question: str) -> bool:
    return bool(examples) and all(
        _INT_RE.fullmatch(ex["input_value"])
        and _ROMAN_RE.fullmatch(ex["output_value"])
        for ex in examples
    ) and bool(_INT_RE.fullmatch(question))


def _is_decimal_mapping(examples: list[Example], question: str) -> bool:
    return bool(examples) and all(
        _DECIMAL_RE.fullmatch(ex["input_value"])
        and _DECIMAL_RE.fullmatch(ex["output_value"])
        for ex in examples
    ) and bool(_DECIMAL_RE.fullmatch(question))


def _is_equation_numeric(examples: list[Example], question: str) -> bool:
    return bool(examples) and all(
        _EQUATION_NUMERIC_RE.fullmatch(ex["input_value"]) for ex in examples
    ) and bool(_EQUATION_NUMERIC_RE.fullmatch(question))


def _is_cryptarithm(examples: list[Example], question: str) -> bool:
    return bool(examples) and all(len(ex["input_value"]) == 5 for ex in examples) and (
        len(question) == 5
    )


def _question_operator_present(examples: list[Example], question: str) -> bool:
    qop = question[2]
    exops = {ex["input_value"][2] for ex in examples if len(ex["input_value"]) >= 3}
    return qop in exops


def detect_category(problem: ProblemLike) -> Category:
    prompt = str(problem.get("prompt", ""))
    question = str(problem["question"])
    examples = problem["examples"]

    if _is_bit_manipulation(examples, question):
        return "bit_manipulation"

    if _is_cipher(examples, question):
        return "cipher"

    if _is_numeral(examples, question):
        return "numeral"

    if _is_decimal_mapping(examples, question):
        prompt_lower = prompt.lower()
        if "gravitational constant" in prompt_lower or "d = 0.5*g*t^2" in prompt_lower:
            return "gravity"
        return "unit_conversion"

    if _is_equation_numeric(examples, question):
        if _question_operator_present(examples, question):
            return "equation_numeric_deduce"
        return "equation_numeric_guess"

    if _is_cryptarithm(examples, question):
        if _question_operator_present(examples, question):
            return "cryptarithm_deduce"
        return "cryptarithm_guess"

    problem_id = problem.get("id", "<unknown>")
    raise ValueError(f"Could not detect category for problem {problem_id}")


def _iter_problem_files(base_dir: Path) -> list[Path]:
    return sorted((base_dir / "problems").glob("*.jsonl"))


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    problem_files = _iter_problem_files(base_dir)
    mismatches = 0

    for path in problem_files:
        with path.open() as f:
            problem = json.loads(f.readline())
        detected = detect_category(problem)
        actual = problem.get("category")
        if detected != actual:
            mismatches += 1
            print(f"{path.stem}: actual={actual} detected={detected}")

    if mismatches == 0:
        print(f"All {len(problem_files)} problems matched detected categories.")
    else:
        print(f"{mismatches} mismatches out of {len(problem_files)} problems.")


if __name__ == "__main__":
    main()
