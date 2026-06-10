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
    lines = [
        "This is standard Roman numeral conversion.",
        "",
        "Step 1: Identify the numeral system from the examples.",
    ]
    for ex in problem.examples:
        lines.append(f"  {ex.input_value} -> {ex.output_value}")

    lines.extend(
        [
            "",
            "Roman values: I=1, V=5, X=10, L=50, C=100, D=500, M=1000.",
            "Subtractive forms: IV=4, IX=9, XL=40, XC=90, CD=400, CM=900.",
            "",
        ]
    )
    n = int(problem.question)
    remaining = n
    parts: list[str] = []
    lines.append(f"Step 2: Convert {n} using largest values first.")
    for value, symbol in ROMAN_VALUES:
        while remaining >= value:
            parts.append(symbol)
            remaining -= value
            lines.append(f"  Take {symbol} ({value}); remainder = {remaining}")

    computed = "".join(parts)
    final_answer = problem.answer.strip()
    lines.extend(
        [
            f"  Combining parts: {computed}",
            "",
            "Step 3: Verify the examples use the same Roman convention.",
        ]
    )
    for ex in problem.examples[:3]:
        check = _to_roman(int(ex.input_value))
        status = "matches" if check == ex.output_value else "does not match"
        lines.append(f"  {ex.input_value} -> {check} ({status})")
    lines.extend(["", f"The answer is \\boxed{{{final_answer}}}"])
    return "\n".join(lines)
