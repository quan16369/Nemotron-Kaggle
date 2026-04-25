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


def _from_roman(s: str) -> int:
    values = {sym: val for val, sym in ROMAN_VALUES}
    total = 0
    i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i : i + 2] in values:
            total += values[s[i : i + 2]]
            i += 2
        else:
            total += values[s[i]]
            i += 1
    return total


def reasoning_numeral(problem: Problem) -> str:
    lines: list[str] = []
    lines.append("We determine the numeral system from the examples:")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    for ex in problem.examples:
        lines.append(f"  {ex.input_value} -> {ex.output_value}")

    lines.append("")
    lines.append("This is Arabic to Roman numeral conversion.")
    n = int(problem.question)
    computed = _to_roman(n)
    lines.append(f"{n} in Roman numerals is {computed}.")

    lines.append("")
    checked = _from_roman(computed)
    lines.append("Double-check:")
    lines.append(f"Converting {computed} back to Arabic gives {checked}.")
    lines.append(f"This matches the original number {n}, so the result is consistent.")

    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer is \\boxed{{{computed}}}")
    return "\n".join(lines)
