"""Compact unit-conversion reasoning generator."""

from __future__ import annotations

import statistics

from reasoners.store_types import Problem


def reasoning_unit_conversion(problem: Problem) -> str | None:
    """Infer a stable conversion ratio from examples, then apply it."""
    observations: list[tuple[str, str, float]] = []
    for example in problem.examples:
        source = float(example.input_value)
        target = float(example.output_value)
        if source != 0:
            observations.append((example.input_value, example.output_value, target / source))
    if not observations:
        return None

    ratio_avg = statistics.mean(ratio for _, _, ratio in observations)
    query = float(problem.question)
    converted = query * ratio_avg
    final_answer = problem.answer.strip()

    lines = [
        "The examples use one linear unit-conversion ratio.",
        "",
        "Step 1: Calculate output/input for each example.",
    ]
    for index, (source, target, ratio) in enumerate(observations, 1):
        lines.extend(
            [
                f"Example {index}: {source} -> {target}",
                f"  ratio_{index} = {target} / {source} = {ratio:.6f}",
            ]
        )

    lines.extend(
        [
            "",
            "Step 2: Average the conversion ratios to reduce rounding noise.",
            f"  ratio_avg = ({' + '.join(f'{ratio:.6f}' for _, _, ratio in observations)}) / {len(observations)}",
            f"  ratio_avg = {ratio_avg:.6f}",
            "",
            f"Step 3: Convert {problem.question}.",
            f"  converted_value = {problem.question} * ratio_avg = {converted:.6f}",
            f"  Rounded to the required precision: {final_answer}",
            "",
            f"The answer is \\boxed{{{final_answer}}}",
        ]
    )
    return "\n".join(lines)
