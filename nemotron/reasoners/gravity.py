"""Gravity: d = k * t^2 reasoning generator."""

from __future__ import annotations

from reasoners.store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    truncate_3dp,
)


def _space_digits(value: str) -> str:
    return " ".join(value)


def _display_decimal(value: str) -> str:
    if "." not in value:
        return value
    return value.rstrip("0").rstrip(".")


def reasoning_gravity(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append("Find k in d = k*t^2, then apply it to the question time.")
    lines.append("[K table]")

    k_strs: list[str] = []
    for i, ex in enumerate(problem.examples, start=1):
        t = float(ex.input_value)
        if t <= 0:
            continue

        _t_sq_lines, t_sq_full = long_multiplication_lines(ex.input_value, ex.input_value)
        t_sq_str = truncate_3dp(t_sq_full)
        d_str = truncate_3dp(ex.output_value)
        d_cast, tsq_cast, _, _ = cast_dp_pair(d_str, t_sq_str)
        _div_lines, k_str = long_division_lines(d_cast, tsq_cast)
        k_strs.append(k_str)

        lines.append(
            f"[K_{i}] t={ex.input_value} d={ex.output_value} "
            f"t_sq={t_sq_str} k={d_cast}/{tsq_cast}={k_str}"
        )

    if not k_strs:
        return None

    k_values = [float(s) for s in k_strs]
    paired = sorted(zip(k_values, k_strs))
    sorted_k = [s for _, s in paired]

    if len(paired) % 2 == 0 and len(paired) >= 2:
        _, k_fit_str = paired[len(paired) // 2 - 1]
    else:
        _, k_fit_str = paired[len(paired) // 2]
    lines.append(f"[Sorted_K] {', '.join(sorted_k)}")
    lines.append(f"[Chosen_K] {k_fit_str}")

    t_sq_lines, t_sq_str = long_multiplication_lines(problem.question, problem.question)
    k_display = _display_decimal(k_fit_str)
    mult_lines, mult_result = long_multiplication_lines(k_display, t_sq_str)
    computed_answer = truncate_3dp(mult_result)
    final_answer = problem.answer.strip()
    lines.append("")
    lines.append("[Final arithmetic]")
    lines.append(f"[Question_t] {problem.question}")
    lines.append(f"[Question_t_sq] {t_sq_str}")
    lines.append(f"[Question_t_sq_steps] {' | '.join(t_sq_lines)}")
    lines.append(f"[Multiply] {k_display} * {t_sq_str}")
    lines.append(f"[Digits_Left] {_space_digits(k_display)}")
    lines.append(f"[Digits_Right] {_space_digits(t_sq_str)}")
    lines.append(f"[Product_Steps] {' | '.join(mult_lines)}")
    lines.append(f"[Raw_Product] {_space_digits(mult_result)}")
    lines.append(f"[Computed] {computed_answer}")
    if final_answer != computed_answer:
        lines.append(f"[Final_Formatted] {final_answer}")

    lines.append("")
    lines.append(f"The answer is \\boxed{{{final_answer}}}")
    return "\n".join(lines)
