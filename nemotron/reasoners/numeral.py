"""Numeral: Arabic to Roman reasoning generator."""

from __future__ import annotations

from reasoners.store_types import Problem

ROMAN_VALUES: list[tuple[int, str]] = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def _to_roman(n: int) -> str:
    parts: list[str] = []
    remaining = n
    for val, sym in ROMAN_VALUES:
        while remaining >= val:
            parts.append(sym)
            remaining -= val
    return "".join(parts)


def reasoning_numeral(problem: Problem) -> str:
    lines: list[str] = []
    lines.append("We need to determine the conversion rule from the examples:")
    lines.append("I will put my final answer inside \\boxed{}.")
    for ex in problem.examples:
        lines.append(f"  {ex.input_value} -> {ex.output_value}")

    lines.append("")
    lines.append("This is Arabic to Roman numeral conversion.")
    lines.append("")
    lines.append("Reference table (1-100):")
    for i in range(1, 101):
        lines.append(f"  {i} = {_to_roman(i)}")

    lines.append("")
    n = int(problem.question)
    computed = _to_roman(n)
    lines.append(f"Converting {n}:")

    remaining = n
    parts: list[str] = []
    for val, sym in ROMAN_VALUES:
        while remaining >= val:
            lines.append(
                f"  {remaining} >= {val} -> {sym}, remainder {remaining - val}"
            )
            parts.append(sym)
            remaining -= val

    lines.append("")
    spaced = " ".join(parts)
    lines.append(f"Result: {spaced} -> {computed}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{computed}}}")
    return "\n".join(lines)
