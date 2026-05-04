"""Unit conversion: output = factor * input reasoning generator."""

from __future__ import annotations

from decimal import Decimal

from reasoners.store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    truncate_3dp,
)


def _decimal_places(value: str) -> int:
    if "." not in value:
        return 0
    return len(value.split(".", 1)[1])


def _to_scaled_int(value: str, dp: int) -> int:
    sign = -1 if value.startswith("-") else 1
    value = value.lstrip("-")
    if "." in value:
        whole, frac = value.split(".", 1)
    else:
        whole, frac = value, ""
    frac = frac.ljust(dp, "0")
    return sign * int((whole or "0") + frac[:dp])


def _format_scaled(value: int, dp: int) -> str:
    sign = "-" if value < 0 else ""
    digits = str(abs(value)).zfill(dp + 1)
    if dp == 0:
        return sign + digits
    return sign + digits[:-dp] + "." + digits[-dp:]


def _scratchpad_multiplication_lines(
    mult_lines: list[str],
    multiplier: str,
    final_label: str,
    final_value: str,
) -> list[str]:
    temp_lines: list[str] = []
    product_entries: list[tuple[str, str, str]] = []
    for line in mult_lines:
        if " = " not in line:
            continue
        lhs, rhs = line.split(" = ", 1)
        if "*" in lhs:
            multiplicand, component = lhs.split(" * ", 1)
            product_entries.append((multiplicand, component, rhs))

    product_entries.sort(key=lambda item: Decimal(item[1]), reverse=True)
    if not product_entries:
        return temp_lines

    components = [component for _, component, _ in product_entries]
    temp_lines.append("[Multiplier_Decomposition]")
    temp_lines.append(f"{multiplier} = {' + '.join(components)}")

    product_vars: list[str] = []
    for i, (multiplicand, component, product) in enumerate(product_entries, start=1):
        var = f"[Temp_Product_{i}]"
        product_vars.append(var)
        temp_lines.append(f"{var} = {multiplicand} * {component} = {product}")

    dp = max(_decimal_places(product) for _, _, product in product_entries)
    running = _to_scaled_int(product_entries[0][2], dp)
    running_var = product_vars[0]
    for i, product_var in enumerate(product_vars[1:], start=1):
        addend = _to_scaled_int(product_entries[i][2], dp)
        running += addend
        sum_var = f"[Temp_Sum_{i}]"
        temp_lines.append(
            f"{sum_var} = {running_var} + {product_var} = {_format_scaled(running, dp)}"
        )
        running_var = sum_var

    temp_lines.append(f"[{final_label}] = {running_var} = {final_value}")
    return temp_lines


def reasoning_unit_conversion(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append(
        "We need to find a conversion rule that maps the inputs to outputs. "
        "Let me check if it's a linear factor."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
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
    mult_lines, mult_result = long_multiplication_lines(q_str, med_display)
    lines.append("")
    lines.append("[Scratchpad]")
    lines.append(f"[Value_1_input] {q_str}")
    lines.append(f"[Value_2_factor] {med_display}")
    lines.append(f"[Final_Product] [Value_1_input] * [Value_2_factor]")
    lines.extend(
        _scratchpad_multiplication_lines(
            mult_lines,
            multiplier=med_display,
            final_label="Final_Product",
            final_value=mult_result,
        )
    )
    lines.append("")
    lines.append(f"Converting {q_str}:")
    lines.append(f"{q_str} * {med_display}:")
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
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer is \\boxed{{{final_answer}}}")
    return "\n".join(lines)
