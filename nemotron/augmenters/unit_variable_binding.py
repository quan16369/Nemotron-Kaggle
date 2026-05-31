"""Unit conversion variable-binding augmenter.

Teaches assigning amount/rate variables and returning the converted_value
without mixing in unrelated numbers.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import random

LINES_PER_PROBLEM = 50
N_PROBLEMS = 300
DEMO_LINES = 3
UNITS = [
    ("zint", "morb"),
    ("glorp", "tav"),
    ("nix", "quol"),
    ("drim", "sarn"),
    ("vep", "luma"),
]


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    decimal = float(value)
    return f"{decimal:.6f}".rstrip("0").rstrip(".")


def _case(rng: random.Random) -> tuple[str, str, Fraction, Fraction, Fraction]:
    source_unit, target_unit = rng.choice(UNITS)
    amount = Fraction(rng.randint(1, 80), rng.choice([1, 2, 4, 5, 10]))
    rate = Fraction(rng.randint(2, 50), rng.choice([1, 2, 4, 5, 10]))
    converted = amount * rate
    return source_unit, target_unit, amount, rate, converted


def _line(index: int, source: str, target: str, amount: Fraction, rate: Fraction) -> str:
    return (
        f"{index:02d} amount={_format_fraction(amount)} {source}; "
        f"rate={_format_fraction(rate)} {target}/{source}"
    )


def _answer_line(
    index: int,
    source: str,
    target: str,
    amount: Fraction,
    rate: Fraction,
    converted: Fraction,
) -> str:
    return (
        f"{index:02d} amount={_format_fraction(amount)} {source}; "
        f"rate={_format_fraction(rate)} {target}/{source} -> "
        f"converted_value=amount*rate="
        f"{_format_fraction(amount)}*{_format_fraction(rate)}="
        f"{_format_fraction(converted)}; "
        f"answer=converted_value={_format_fraction(converted)} {target}"
    )


def generate() -> list[dict[str, str]]:
    rng = random.Random(20260533)
    problems: list[dict[str, str]] = []

    for i in range(N_PROBLEMS):
        demo = [_case(rng) for _ in range(DEMO_LINES)]
        sample_input_lines = [
            _line(j, source, target, amount, rate)
            for j, (source, target, amount, rate, _) in enumerate(demo)
        ]
        sample_output_lines = [
            _answer_line(j, source, target, amount, rate, converted)
            for j, (source, target, amount, rate, converted) in enumerate(demo)
        ]

        test_inputs: list[str] = []
        test_answers: list[str] = []
        for row_num in range(LINES_PER_PROBLEM):
            source, target, amount, rate, converted = _case(rng)
            test_inputs.append(_line(row_num, source, target, amount, rate))
            test_answers.append(
                _answer_line(row_num, source, target, amount, rate, converted)
            )

        prompt = (
            "In Alice's Wonderland, solve each unit conversion row by binding "
            "amount, rate, converted_value, and answer. Do not use any unrelated "
            "numbers.\n\n"
            "This is a sample input.\n"
            + "\n".join(sample_input_lines)
            + "\n\n"
            + "This is a sample output.\n"
            + "\n".join(sample_output_lines)
            + "\n\n"
            + "This is your input.\n"
            + "\n".join(test_inputs)
        )
        completion = "\n".join(test_answers)
        pid = hashlib.sha256(f"unit_variable_binding_{i}".encode()).hexdigest()[:8]
        problems.append(
            {
                "id": pid,
                "prompt": prompt,
                "completion": completion,
                "category": "unit_variable_binding",
            }
        )

    print(f"[unit_variable_binding] Generated {len(problems)} problems")
    return problems
