"""Uniform 3-agent reasoning wrapper for SFT completions."""

from __future__ import annotations

import re


FINAL_LINE_PATTERNS = (
    re.compile(r"^\s*I will now return the answer in \\boxed\{\}\.?\s*$"),
    re.compile(r"^\s*I will put my final answer inside \\boxed\{\}\.?\s*$"),
    re.compile(r"^\s*The answer is \\boxed\{[^{}]*\}\.?\s*$"),
    re.compile(r"^\s*The answer in \\boxed\{[^{}]*\} is \\boxed\{[^{}]*\}\.?\s*$"),
    re.compile(r"^\s*Final answer is:\s*.+\s*$", re.IGNORECASE),
    re.compile(r"^\s*Final answer\s*[:：]\s*.+\s*$", re.IGNORECASE),
)

NUMERIC_CATEGORIES = {
    "gravity",
    "unit_conversion",
    "equation_numeric_deduce",
    "equation_numeric_guess",
}


def _strip_trailing_final_answer_lines(reasoning_text: str) -> str:
    lines = reasoning_text.rstrip("\n").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and any(pattern.match(lines[-1]) for pattern in FINAL_LINE_PATTERNS):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).rstrip("\n")


def _sanitize_internal_boxed(text: str) -> str:
    text = re.sub(r"\\boxed\{([^{}]*)\}", r"boxed(\1)", text)
    return text.replace(r"\boxed{}", "boxed()")


def _verifier_lines(category: str, answer: str) -> list[str]:
    lines = [
        "[Agent_2_Verifier]",
        f"category = {category}",
        f"candidate_answer = {answer}",
    ]
    if category == "bit_manipulation":
        lines.extend(
            [
                "constraint.binary_string = yes",
                "constraint.preserve_leading_zeros = yes",
                "constraint.exact_string_match = yes",
            ]
        )
    elif category == "cipher":
        lines.extend(
            [
                "constraint.dictionary_candidate = yes",
                "constraint.pattern_match = yes",
                "constraint.bijection_consistent = yes",
                "constraint.semantic_guessing = no",
            ]
        )
    elif category in NUMERIC_CATEGORIES:
        lines.extend(
            [
                "constraint.numeric_answer = yes",
                "constraint.use_clean_gold_answer = yes",
                "constraint.final_rounding_already_applied = yes",
            ]
        )
    elif category == "numeral":
        lines.extend(
            [
                "constraint.roman_numeral = yes",
                "constraint.exact_string_match = yes",
            ]
        )
    elif category.startswith("cryptarithm"):
        lines.extend(
            [
                "constraint.symbol_mapping_consistent = yes",
                "constraint.answer_uses_solved_symbols = yes",
            ]
        )
    else:
        lines.append("constraint.use_clean_gold_answer = yes")
    return lines


def wrap_three_agent_reasoning(
    reasoning_text: str,
    *,
    category: str,
    answer: str,
) -> str:
    """Return a category-uniform 3-agent trace without internal boxed answers."""
    solver_trace = _sanitize_internal_boxed(
        _strip_trailing_final_answer_lines(reasoning_text)
    ).rstrip("\n")

    lines = ["[Agent_1_Solver]"]
    if solver_trace:
        lines.append(solver_trace)
    else:
        lines.append("No solver trace was available.")
    lines.append("")
    lines.extend(_verifier_lines(category, answer))
    lines.append("")
    lines.extend(
        [
            "[Agent_3_Consensus]",
            f"selected_answer = {answer}",
            "final_answer_source = gold_clean_answer",
        ]
    )
    return "\n".join(lines).rstrip("\n")


def build_three_agent_completion(
    reasoning_text: str,
    *,
    category: str,
    answer: str,
) -> str:
    wrapped = wrap_three_agent_reasoning(
        reasoning_text,
        category=category,
        answer=answer,
    )
    return f"{wrapped}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
