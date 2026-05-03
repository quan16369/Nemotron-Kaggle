"""Unit conversion: output = factor * input reasoning generator."""

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


def reasoning_unit_conversion(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append("Find the linear conversion factor.")
    lines.append("[Factor table]")
    factor_strs: list[str] = []
    for i, ex in enumerate(problem.examples, start=1):
        inp = float(ex.input_value)
        if inp != 0:
            out_str = truncate_3dp(ex.output_value)
            inp_str = truncate_3dp(ex.input_value)
            inp_cast, out_cast, inp_dp, out_dp = cast_dp_pair(inp_str, out_str)
            _div_lines, factor_str = long_division_lines(out_cast, inp_cast)
            lines.append(
                f"[Factor_{i}] input={inp_cast} output={out_cast} "
                f"factor={out_cast}/{inp_cast}={factor_str}"
            )
            factor_strs.append(factor_str)

    if not factor_strs:
        return None

    factors = [float(s) for s in factor_strs]

    # List factor values and pick median (for even count, use the smaller middle value)
    f_list_str = ", ".join(factor_strs)
    lines.append(f"factor values: {f_list_str}")
    paired = sorted(zip(factors, factor_strs))
    if len(paired) % 2 == 0 and len(paired) >= 2:
        _, med_factor_str = paired[len(paired) // 2 - 1]
    else:
        mid = len(paired) // 2
        _, med_factor_str = paired[mid]
    sorted_str = ", ".join(s for _, s in paired)
    lines.append(f"[Sorted_Factors] {sorted_str}")
    lines.append(f"[Chosen_Factor] {med_factor_str}")

    q_str = problem.question
    med_display = med_factor_str.rstrip("0").rstrip(".")
    mult_lines, mult_result = long_multiplication_lines(q_str, med_display)
    computed_answer = truncate_3dp(mult_result)
    final_answer = problem.answer.strip()
    lines.append("")
    lines.append("[Final arithmetic]")
    lines.append(f"[Question] {q_str}")
    lines.append(f"[Multiply] {q_str} * {med_display}")
    lines.append(f"[Digits_Left] {_space_digits(q_str)}")
    lines.append(f"[Digits_Right] {_space_digits(med_display)}")
    lines.append(f"[Product_Steps] {' | '.join(mult_lines)}")
    lines.append(f"[Raw_Product] {_space_digits(mult_result)}")
    lines.append(f"[Computed] {computed_answer}")
    if final_answer != computed_answer:
        lines.append(f"[Final_Formatted] {final_answer}")

    lines.append("")
    lines.append(f"The answer is \\boxed{{{final_answer}}}")
    return "\n".join(lines)
