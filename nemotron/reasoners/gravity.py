"""Gravity: d = k * t^2 reasoning generator."""

from __future__ import annotations

from reasoners.store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    truncate_3dp,
)


def reasoning_gravity(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append("We determine the falling distance using d = k*t^2.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    k_strs: list[str] = []
    for ex in problem.examples:
        t = float(ex.input_value)
        if t > 0:
            t_squared = round(t * t, 4)
            t_sq_full = str(t_squared)
            t_sq_str = truncate_3dp(t_sq_full)
            d_str = truncate_3dp(ex.output_value)

            d_cast, tsq_cast, _, _ = cast_dp_pair(d_str, t_sq_str)
            _, k_str = long_division_lines(d_cast, tsq_cast)
            k_strs.append(k_str)
            lines.append(
                f"t = {ex.input_value}, d = {ex.output_value}: "
                f"t^2 = {t_sq_full}, so k ≈ {d_cast} / {tsq_cast} = {k_str}"
            )
            lines.append("")

    if not k_strs:
        return None

    k_values = [float(s) for s in k_strs]

    # List k values and pick median (for even count, use the smaller middle value)
    k_list_str = ", ".join(k_strs)
    lines.append(f"k values: {k_list_str}")
    paired = sorted(zip(k_values, k_strs))
    if len(paired) % 2 == 0 and len(paired) >= 2:
        _, k_fit_str = paired[len(paired) // 2 - 1]
    else:
        mid = len(paired) // 2
        _, k_fit_str = paired[mid]
    lines.append(f"We use the median k = {k_fit_str}.")

    lines.append("")
    lines.append(f"For t = {problem.question}:")
    _, t_sq_str = long_multiplication_lines(problem.question, problem.question)
    lines.append(f"t^2 = {problem.question} * {problem.question} = {t_sq_str}")
    k_display = k_fit_str.rstrip("0").rstrip(".")
    _, mult_result = long_multiplication_lines(k_display, t_sq_str)
    boxed_answer = truncate_3dp(mult_result)
    lines.append(f"d = {k_display} * {t_sq_str} = {mult_result}")
    lines.append(f"So the distance is {boxed_answer}.")

    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{boxed_answer}}}")
    return "\n".join(lines)
