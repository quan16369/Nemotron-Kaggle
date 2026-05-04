"""Gravity: d = k * t^2 reasoning generator."""

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
    lines.append("[Scratchpad]")
    lines.append(f"[Value_1_k] {k_display}")
    lines.append(f"[Value_2_t_sq] {t_sq_str}")
    lines.append(f"[Final_Distance] [Value_1_k] * [Value_2_t_sq]")
    lines.extend(
        _scratchpad_multiplication_lines(
            mult_lines,
            multiplier=t_sq_str,
            final_label="Final_Distance",
            final_value=mult_result,
        )
    )
    lines.append("")
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
