"""Gravity arithmetic scratchpad augmenter.

Teaches binding intermediate arithmetic values to names and taking the final
answer from the final variable only.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import random

LINES_PER_PROBLEM = 50
N_PROBLEMS = 300
DEMO_LINES = 3


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    decimal = float(value)
    text = f"{decimal:.6f}".rstrip("0").rstrip(".")
    return text


def _case(rng: random.Random) -> tuple[Fraction, int, Fraction]:
    g = Fraction(rng.randint(2, 40), rng.choice([1, 2, 3, 4, 5, 8, 10]))
    t = rng.randint(1, 20)
    distance = Fraction(1, 2) * g * t * t
    return g, t, distance


def _line(index: int, g: Fraction, t: int) -> str:
    return f"{index:02d} g={_format_fraction(g)} t={t}"


def _answer_line(index: int, g: Fraction, t: int, distance: Fraction) -> str:
    t_squared = t * t
    half_g = g / 2
    return (
        f"{index:02d} g={_format_fraction(g)} t={t} -> "
        f"t_squared={t_squared}; "
        f"half_g={_format_fraction(half_g)}; "
        f"distance=half_g*t_squared={_format_fraction(distance)}; "
        f"answer=distance={_format_fraction(distance)}"
    )


def generate() -> list[dict[str, str]]:
    rng = random.Random(20260532)
    problems: list[dict[str, str]] = []

    for i in range(N_PROBLEMS):
        demo = [_case(rng) for _ in range(DEMO_LINES)]
        sample_input_lines = [_line(j, g, t) for j, (g, t, _) in enumerate(demo)]
        sample_output_lines = [
            _answer_line(j, g, t, distance)
            for j, (g, t, distance) in enumerate(demo)
        ]

        test_inputs: list[str] = []
        test_answers: list[str] = []
        for row_num in range(LINES_PER_PROBLEM):
            g, t, distance = _case(rng)
            test_inputs.append(_line(row_num, g, t))
            test_answers.append(_answer_line(row_num, g, t, distance))

        prompt = (
            "In Alice's Wonderland, solve each gravity arithmetic row by binding "
            "intermediate values to variable names. The final answer must be exactly "
            "the value bound to answer.\n\n"
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
        pid = hashlib.sha256(f"gravity_arithmetic_{i}".encode()).hexdigest()[:8]
        problems.append(
            {
                "id": pid,
                "prompt": prompt,
                "completion": completion,
                "category": "gravity_arithmetic",
            }
        )

    print(f"[gravity_arithmetic] Generated {len(problems)} problems")
    return problems
