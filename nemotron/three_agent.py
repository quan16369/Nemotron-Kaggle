"""Uniform 3-agent reasoning wrapper for SFT completions."""

from __future__ import annotations

import hashlib
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

NEGATIVE_TO_CORRECT_RATE = 0.20
NEGATIVE_TO_CORRECT_CATEGORIES = {"bit_manipulation", "gravity"}


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


def _fit_solver_trace_to_char_budget(text: str, char_budget: int | None) -> str:
    if char_budget is None or len(text) <= char_budget:
        return text
    if char_budget <= 0:
        return ""
    if char_budget < 256:
        return text[:char_budget].rstrip()

    head_budget = max(128, char_budget // 3)
    tail_budget = max(128, char_budget - head_budget - 64)
    return (
        text[:head_budget].rstrip()
        + "\n[Solver_Trace_Truncated]\n"
        + text[-tail_budget:].lstrip()
    ).rstrip()


def _stable_fraction(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _corrupt_binary_answer(answer: str, key: str) -> tuple[str, str]:
    if re.fullmatch(r"[01]+", answer) is None:
        return answer + "0", "non_binary_candidate"
    if answer.startswith("0") and len(answer) > 1:
        stripped = answer.lstrip("0")
        return stripped or "0", "leading_zero_missing"
    bit_index = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % max(
        len(answer), 1
    )
    flipped = list(answer)
    flipped[bit_index] = "1" if flipped[bit_index] == "0" else "0"
    return "".join(flipped), "bit_value_mismatch"


def _corrupt_numeric_answer(answer: str, key: str) -> tuple[str, str]:
    try:
        value = float(answer)
    except Exception:
        return answer + "0", "numeric_format_mismatch"

    decimals = 0
    if "." in answer:
        decimals = len(answer.split(".", 1)[1])
    direction = 1 if int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:2], 16) % 2 else -1
    step = 10 ** (-decimals) if decimals > 0 else 1
    corrupted_value = value + direction * step
    if decimals > 0:
        corrupted = f"{corrupted_value:.{decimals}f}"
    else:
        corrupted = str(int(round(corrupted_value)))
    if corrupted == answer:
        corrupted_value = value - direction * step
        corrupted = f"{corrupted_value:.{decimals}f}" if decimals > 0 else str(int(round(corrupted_value)))
    return corrupted, "final_rounding_mismatch"


def _use_negative_to_correct(
    *,
    category: str,
    answer: str,
    problem_id: str | None,
    rate: float,
) -> bool:
    if category not in NEGATIVE_TO_CORRECT_CATEGORIES or rate <= 0:
        return False
    key = problem_id or answer
    return _stable_fraction(f"{category}-negative:{key}") < rate


def _verifier_lines(
    category: str,
    answer: str,
    *,
    candidate_answer: str | None = None,
    candidate_valid: bool = True,
    error_type: str | None = None,
) -> list[str]:
    candidate_answer = answer if candidate_answer is None else candidate_answer
    lines = [
        "[Verifier_Agent]",
        f"category = {category}",
        f"candidate_answer = {candidate_answer}",
    ]
    if candidate_valid:
        lines.append("candidate_valid = yes")
    else:
        lines.extend(
            [
                "candidate_valid = no",
                f"error_type = {error_type or 'constraint_mismatch'}",
                f"corrected_answer = {answer}",
            ]
        )
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
    problem_id: str | None = None,
    solver_char_budget: int | None = None,
    negative_to_correct_rate: float = NEGATIVE_TO_CORRECT_RATE,
) -> str:
    """Return a category-uniform 3-agent trace without internal boxed answers."""
    solver_trace = _sanitize_internal_boxed(
        _strip_trailing_final_answer_lines(reasoning_text)
    ).rstrip("\n")
    solver_trace = _fit_solver_trace_to_char_budget(solver_trace, solver_char_budget)
    use_negative = _use_negative_to_correct(
        category=category,
        answer=answer,
        problem_id=problem_id,
        rate=negative_to_correct_rate,
    )
    candidate_answer = answer
    candidate_valid = True
    error_type = None
    if use_negative:
        if category == "bit_manipulation":
            candidate_answer, error_type = _corrupt_binary_answer(
                answer,
                problem_id or reasoning_text or answer,
            )
        elif category == "gravity":
            candidate_answer, error_type = _corrupt_numeric_answer(
                answer,
                problem_id or reasoning_text or answer,
            )
        candidate_valid = candidate_answer == answer

    lines = ["[Solver_Agent]"]
    if solver_trace:
        lines.append(solver_trace)
    else:
        lines.append("No solver trace was available.")
    if use_negative and not candidate_valid:
        lines.extend(
            [
                "",
                "[Preliminary_Candidate]",
                f"candidate_answer = {candidate_answer}",
            ]
        )
    lines.append("")
    lines.extend(
        _verifier_lines(
            category,
            answer,
            candidate_answer=candidate_answer,
            candidate_valid=candidate_valid,
            error_type=error_type,
        )
    )
    lines.append("")
    lines.extend(
        [
            "[Consensus_Agent]",
            f"selected_answer = {answer}",
            "final_answer_source = gold_clean_answer",
            "final_response_must_be_boxed = yes",
        ]
    )
    return "\n".join(lines).rstrip("\n")


def build_three_agent_completion(
    reasoning_text: str,
    *,
    category: str,
    answer: str,
    problem_id: str | None = None,
    solver_char_budget: int | None = None,
    negative_to_correct_rate: float = NEGATIVE_TO_CORRECT_RATE,
) -> str:
    wrapped = wrap_three_agent_reasoning(
        reasoning_text,
        category=category,
        answer=answer,
        problem_id=problem_id,
        solver_char_budget=solver_char_budget,
        negative_to_correct_rate=negative_to_correct_rate,
    )
    return f"{wrapped}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
