"""Leading-zero binary augmenter.

Teaches exact fixed-width binary formatting so answers like 00001011 keep
their leading zeros instead of collapsing to 1011.
"""

from __future__ import annotations

import hashlib
import random

LINES_PER_PROBLEM = 80
N_PROBLEMS = 300
DEMO_LINES = 4
WIDTHS = (4, 6, 8, 10, 12, 16)


def _case(rng: random.Random) -> tuple[int, int, str]:
    width = rng.choice(WIDTHS)
    value = rng.randrange(0, 2**width)
    return width, value, format(value, f"0{width}b")


def _line(index: int, width: int, value: int) -> str:
    return f"{index:02d} width={width} value={value}"


def _answer_line(index: int, width: int, value: int, answer: str) -> str:
    return f"{index:02d} width={width} value={value} -> {answer}"


def generate() -> list[dict[str, str]]:
    rng = random.Random(20260531)
    problems: list[dict[str, str]] = []

    for i in range(N_PROBLEMS):
        demo = [_case(rng) for _ in range(DEMO_LINES)]
        sample_input_lines = [_line(j, width, value) for j, (width, value, _) in enumerate(demo)]
        sample_output_lines = [
            _answer_line(j, width, value, answer)
            for j, (width, value, answer) in enumerate(demo)
        ]

        test_inputs: list[str] = []
        test_answers: list[str] = []
        for row_num in range(LINES_PER_PROBLEM):
            width, value, answer = _case(rng)
            test_inputs.append(_line(row_num, width, value))
            test_answers.append(_answer_line(row_num, width, value, answer))

        prompt = (
            "In Alice's Wonderland, binary answers must preserve their exact fixed width.\n\n"
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
        pid = hashlib.sha256(f"leading_zero_binary_{i}".encode()).hexdigest()[:8]
        problems.append(
            {
                "id": pid,
                "prompt": prompt,
                "completion": completion,
                "category": "leading_zero_binary",
            }
        )

    print(f"[leading_zero_binary] Generated {len(problems)} problems")
    return problems
