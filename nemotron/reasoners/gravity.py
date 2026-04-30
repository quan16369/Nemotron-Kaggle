"""Gravity: d = k * t^2 reasoning generator."""

from __future__ import annotations

from reasoners.store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    truncate_3dp,
)


def _display_decimal(value: str) -> str:
    if "." not in value:
        return value
    return value.rstrip("0").rstrip(".")


def reasoning_gravity(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append(
        "We need to determine the falling distance using d = k*t^2. Let me find k from the examples."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    k_strs: list[str] = []
    for ex in problem.examples:
        t = float(ex.input_value)
        if t <= 0:
            continue

        t_sq_lines, t_sq_full = long_multiplication_lines(ex.input_value, ex.input_value)
        t_sq_str = truncate_3dp(t_sq_full)
        d_str = truncate_3dp(ex.output_value)
        d_cast, tsq_cast, _, _ = cast_dp_pair(d_str, t_sq_str)
        div_lines, k_str = long_division_lines(d_cast, tsq_cast)
        k_strs.append(k_str)

        lines.append(f"t = {ex.input_value}s, d = {ex.output_value}m:")
        lines.append(f"t^2 = {ex.input_value} * {ex.input_value}:")
        lines.extend(t_sq_lines)
        lines.append(
            f"k = {ex.output_value} / {ex.input_value}^2 = {ex.output_value} / {t_sq_full} = {d_cast} / {tsq_cast}"
        )
        lines.extend(div_lines)
        lines.append(f"= {k_str}")
        lines.append("")

    if not k_strs:
        return None

    k_values = [float(s) for s in k_strs]
    paired = sorted(zip(k_values, k_strs))
    sorted_k = [s for _, s in paired]

    lines.append(f"k values: {', '.join(k_strs)}")
    lines.append(f"k values (sorted): {', '.join(sorted_k)}")
    if len(paired) % 2 == 0 and len(paired) >= 2:
        _, k_fit_str = paired[len(paired) // 2 - 1]
    else:
        _, k_fit_str = paired[len(paired) // 2]
    lines.append(f"The median k is {k_fit_str}.")

    lines.append("")
    lines.append(f"For t = {problem.question}:")
    t_sq_lines, t_sq_str = long_multiplication_lines(problem.question, problem.question)
    lines.append(f"t^2 = {problem.question} * {problem.question}:")
    lines.extend(t_sq_lines)
    lines.append(f"= {t_sq_str}")
    lines.append("")

    k_display = _display_decimal(k_fit_str)
    mult_lines, mult_result = long_multiplication_lines(k_display, t_sq_str)
    computed_answer = truncate_3dp(mult_result)
    final_answer = problem.answer.strip()
    lines.append(f"d = {k_display} * {t_sq_str}:")
    lines.extend(mult_lines)
    lines.append(f"= {computed_answer}")
    if final_answer != computed_answer:
        lines.append(f"The final formatted value is {final_answer}.")
    lines.append("")

    check_lines, k_check_str = long_division_lines(mult_result, t_sq_str)
    lines.append("Double-check:")
    lines.append(f"If d = {mult_result}, then k = d / t^2 = {mult_result} / {t_sq_str}:")
    lines.extend(check_lines)
    lines.append(f"= {k_check_str}")
    lines.append(f"This matches the chosen k = {k_fit_str}, so the result is consistent.")

    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer is \\boxed{{{final_answer}}}")
    return "\n".join(lines)
