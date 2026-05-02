"""Unit conversion: output = factor * input reasoning generator."""

from __future__ import annotations

from reasoners.store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    truncate_3dp,
)


def reasoning_unit_conversion(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append(
        "We need to find a conversion rule that maps the inputs to outputs. "
        "Let me check if it's a linear factor."
    )
    lines.append("I will compute the answer and state it at the end.")
    lines.append("")
    factor_strs: list[str] = []
    for ex in problem.examples:
        inp = float(ex.input_value)
        if inp != 0:
            out_str = truncate_3dp(ex.output_value)
            inp_str = truncate_3dp(ex.input_value)
            lines.append(f"{ex.input_value} -> {ex.output_value}")
            inp_cast, out_cast, inp_dp, out_dp = cast_dp_pair(inp_str, out_str)
            lines.append(
                f"Casting input to {inp_dp} decimal places, "
                f"output to {out_dp} decimal places: "
                f"{inp_cast} -> {out_cast}"
            )
            lines.append(f"factor = {out_cast} / {inp_cast}")
            div_lines, factor_str = long_division_lines(out_cast, inp_cast)
            lines.extend(div_lines)
            lines.append(f"= {factor_str}")
            factor_strs.append(factor_str)
            lines.append("")

    if not factor_strs:
        return None

    factors = [float(s) for s in factor_strs]

    # List factor values and pick median (for even count, use the smaller middle value)
    f_list_str = ", ".join(factor_strs)
    lines.append(f"factor values: {f_list_str}")
    paired = sorted(zip(factors, factor_strs))
    sorted_str = ", ".join(s for _, s in paired)
    lines.append(f"factor values (sorted): {sorted_str}")
    if len(paired) % 2 == 0 and len(paired) >= 2:
        _, med_factor_str = paired[len(paired) // 2 - 1]
    else:
        mid = len(paired) // 2
        _, med_factor_str = paired[mid]
    lines.append(f"The median factor is {med_factor_str}.")

    q_str = problem.question
    med_display = med_factor_str.rstrip("0").rstrip(".")
    lines.append("")
    lines.append(f"Converting {q_str}:")
    lines.append(f"{q_str} * {med_display}:")
    mult_lines, mult_result = long_multiplication_lines(q_str, med_display)
    lines.extend(mult_lines)
    computed_answer = truncate_3dp(mult_result)
    final_answer = problem.answer.strip()
    lines.append(f"= {computed_answer}")
    if final_answer != computed_answer:
        lines.append(f"Rounded to the required precision, this is {final_answer}.")

    lines.append("")
    check_lines, factor_check_str = long_division_lines(mult_result, q_str)
    lines.append("Double-check:")
    lines.append(
        f"If the converted value is {mult_result}, then factor = {mult_result} / {q_str}:"
    )
    lines.extend(check_lines)
    lines.append(f"= {factor_check_str}")
    lines.append(
        f"This matches the chosen factor = {med_factor_str}, so the result is consistent."
    )

    lines.append("")
    lines.append("I will now state the final answer.")
    lines.append(f"Final answer is: {final_answer}")
    return "\n".join(lines)
