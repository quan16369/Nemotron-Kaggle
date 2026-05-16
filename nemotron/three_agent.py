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


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        return 0
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def _select_variant(key: str, variants: list[str]) -> str:
    return variants[_stable_index(key, len(variants))]


def _extract_bit_output_from_trace(
    reasoning_text: str,
) -> tuple[str | None, dict[int, str], dict[int, str]]:
    lines = reasoning_text.splitlines()

    for start in range(len(lines) - 1, -1, -1):
        if lines[start].strip() != "Output":
            continue

        bit_map: dict[int, str] = {}
        rule_map: dict[int, str] = {}
        for raw_line in lines[start + 1 :]:
            line = raw_line.strip()
            if not line:
                if bit_map:
                    break
                continue
            lower = line.lower()
            if (
                line.startswith("[")
                or lower.startswith("i will")
                or lower.startswith("the answer")
                or lower.startswith("final answer")
            ):
                break

            match = re.match(r"^(\d+)\s+(?:.*=\s*)?([01])\s*$", line)
            if match:
                position = int(match.group(1))
                bit_map[position] = match.group(2)
                rule_map[position] = line
                continue
            if bit_map:
                break

        if bit_map:
            positions = sorted(bit_map)
            reconstructed = "".join(bit_map[position] for position in positions)
            return reconstructed, bit_map, rule_map

    for raw_line in reversed(lines):
        line = raw_line.strip()
        match = re.search(r"\bOutput\s*=.*=\s*([01]{2,})\s*$", line)
        if not match:
            match = re.search(r"\bBOX\s+\\boxed\{([01]{2,})\}", line)
        if match:
            reconstructed = match.group(1)
            bit_map = {idx: bit for idx, bit in enumerate(reconstructed)}
            rule_map = {idx: line for idx in bit_map}
            return reconstructed, bit_map, rule_map

    return None, {}, {}


def _corrupt_binary_answer(
    answer: str,
    key: str,
    reasoning_text: str = "",
    forced_failed_constraint: str | None = None,
) -> tuple[str, str, dict[str, str]]:
    reconstructed, bit_map, rule_map = _extract_bit_output_from_trace(reasoning_text)
    corrected = (
        reconstructed
        if reconstructed is not None
        and re.fullmatch(r"[01]+", reconstructed)
        and len(reconstructed) == len(answer)
        else answer
    )
    verification_source = (
        "solver_output_block" if reconstructed is not None else "clean_gold_answer"
    )

    variants = ["binary_string", "exact_string_match"]
    if corrected.startswith("0") and len(corrected) > 1:
        variants.append("preserve_leading_zeros")
    failed_constraint = (
        forced_failed_constraint
        if forced_failed_constraint in variants
        else _select_variant(f"bit:{key}", variants)
    )

    if failed_constraint == "binary_string" or re.fullmatch(r"[01]+", answer) is None:
        candidate = (corrected[:-1] + "2") if corrected else "2"
        evidence = {
            "verification_source": verification_source,
            "corrected_answer_source": verification_source,
            "failed_constraint": "binary_string",
            "format_check": "failed",
            "candidate_error": "contains_non_binary_or_extra_character",
            "required_format": "binary_string",
            "candidate_length": str(len(candidate)),
            "correct_length": str(len(corrected)),
        }
        if reconstructed is not None:
            evidence["reconstructed_answer_from_trace"] = reconstructed
        return candidate, "non_binary_candidate", evidence

    if failed_constraint == "preserve_leading_zeros":
        stripped = corrected.lstrip("0")
        candidate = stripped or "0"
        missing = len(corrected) - len(candidate)
        evidence = {
            "verification_source": verification_source,
            "corrected_answer_source": verification_source,
            "failed_constraint": "preserve_leading_zeros",
            "length_check": "failed",
            "leading_zero_check": "failed",
            "candidate_error": "dropped_leading_zero",
            "required_length": str(len(corrected)),
            "candidate_length": str(len(candidate)),
            "missing_leading_zeros": str(missing),
        }
        if reconstructed is not None:
            evidence["reconstructed_answer_from_trace"] = reconstructed
        return candidate, "leading_zero_missing", evidence

    bit_index = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % max(
        len(corrected), 1
    )
    flipped = list(corrected)
    flipped[bit_index] = "1" if flipped[bit_index] == "0" else "0"
    candidate = "".join(flipped)
    expected_bit = bit_map.get(bit_index, corrected[bit_index])
    evidence = {
        "verification_source": verification_source,
        "corrected_answer_source": verification_source,
        "failed_constraint": "exact_string_match",
        "reconstructed_answer_from_trace": reconstructed or corrected,
        "length_check": "passed",
        "candidate_error": "single_bit_flip",
        "mismatch_position": str(bit_index),
        "candidate_bit_at_position": candidate[bit_index],
        "expected_bit_from_trace": expected_bit,
        "trace_output_line": rule_map.get(bit_index, ""),
        "required_length": str(len(corrected)),
        "candidate_length": str(len(candidate)),
    }
    return candidate, "bit_value_mismatch", evidence


def _format_numeric_like(value: float, decimals: int) -> str:
    if decimals > 0:
        return f"{value:.{decimals}f}"
    return str(int(round(value)))


def _extract_gravity_computed_value(reasoning_text: str, answer: str) -> str | None:
    decimals = len(answer.split(".", 1)[1]) if "." in answer else 0
    final_distance_candidates: list[float] = []
    for line in reasoning_text.splitlines():
        if "[Final_Distance]" in line:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
            for number in numbers:
                try:
                    final_distance_candidates.append(float(number))
                except Exception:
                    pass
    if final_distance_candidates:
        return str(final_distance_candidates[-1])

    candidates: list[float] = []
    if not candidates:
        for line in reversed(reasoning_text.splitlines()):
            stripped = line.strip()
            if stripped.startswith("="):
                numbers = re.findall(r"-?\d+(?:\.\d+)?", stripped)
                for number in numbers:
                    try:
                        candidates.append(float(number))
                    except Exception:
                        pass
                if candidates:
                    break
    for value in reversed(candidates):
        if _format_numeric_like(value, decimals) == answer:
            return str(value)
    return str(candidates[-1]) if candidates else None


def _corrupt_numeric_answer(
    answer: str,
    key: str,
    reasoning_text: str = "",
    forced_failed_constraint: str | None = None,
) -> tuple[str, str, dict[str, str]]:
    try:
        value = float(answer)
    except Exception:
        candidate = answer + "0"
        return (
            candidate,
            "numeric_format_mismatch",
            {
                "verification_source": "numeric_format_check",
                "corrected_answer_source": "verifier_numeric_answer",
                "failed_constraint": "numeric_answer",
                "required_numeric_answer": "yes",
                "numeric_parse_check": "failed",
                "candidate_error": "not_parseable_as_clean_number",
            },
        )

    decimals = 0
    if "." in answer:
        decimals = len(answer.split(".", 1)[1])
    computed_value = _extract_gravity_computed_value(reasoning_text, answer)
    variants = ["numeric_answer", "final_rounding_already_applied"]
    if computed_value is not None:
        try:
            computed_float = float(computed_value)
            if _format_numeric_like(computed_float, decimals) == answer and computed_value != answer:
                variants.append("use_clean_final_answer")
        except Exception:
            pass
    failed_constraint = (
        forced_failed_constraint
        if forced_failed_constraint in variants
        else _select_variant(f"gravity:{key}", variants)
    )

    if failed_constraint == "numeric_answer":
        candidate = f"{answer} m"
        return (
            candidate,
            "numeric_format_mismatch",
            {
                "verification_source": "numeric_format_check",
                "corrected_answer_source": "verifier_numeric_answer",
                "failed_constraint": "numeric_answer",
                "numeric_parse_check": "failed",
                "candidate_error": "answer_contains_unit_text",
            },
        )

    if failed_constraint == "use_clean_final_answer" and computed_value is not None:
        return (
            computed_value,
            "raw_unrounded_value_leaked",
            {
                "verification_source": "computed_value_rounding",
                "corrected_answer_source": "computed_value_rounding",
                "failed_constraint": "use_clean_final_answer",
                "rounding_rule": f"nearest_{decimals}_decimal_places",
                "rounding_decimals": str(decimals),
                "computed_value": computed_value,
                "candidate_error": "raw_unrounded_value_used_as_final",
                "correct_rounded_value": answer,
            },
        )

    direction = 1 if int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:2], 16) % 2 else -1
    step = 10 ** (-decimals) if decimals > 0 else 1
    corrupted_value = value + direction * step
    corrupted = _format_numeric_like(corrupted_value, decimals)
    if corrupted == answer:
        corrupted_value = value - direction * step
        corrupted = _format_numeric_like(corrupted_value, decimals)
    evidence = {
        "verification_source": "computed_value_rounding",
        "corrected_answer_source": "computed_value_rounding",
        "failed_constraint": "final_rounding_already_applied",
        "rounding_rule": f"nearest_{decimals}_decimal_places",
        "rounding_decimals": str(decimals),
        "candidate_rounded_value": corrupted,
        "correct_rounded_value": answer,
        "candidate_error": "one_unit_in_last_decimal_place",
    }
    if computed_value is not None:
        evidence["computed_value"] = computed_value
    return corrupted, "final_rounding_mismatch", evidence


def available_negative_constraints(
    *,
    category: str,
    answer: str,
    reasoning_text: str,
) -> list[str]:
    if category == "bit_manipulation":
        reconstructed, _, _ = _extract_bit_output_from_trace(reasoning_text)
        corrected = (
            reconstructed
            if reconstructed is not None
            and re.fullmatch(r"[01]+", reconstructed)
            and len(reconstructed) == len(answer)
            else answer
        )
        constraints = ["binary_string", "exact_string_match"]
        if corrected.startswith("0") and len(corrected) > 1:
            constraints.append("preserve_leading_zeros")
        return constraints

    if category == "gravity":
        constraints = ["numeric_answer", "final_rounding_already_applied"]
        computed_value = _extract_gravity_computed_value(reasoning_text, answer)
        if computed_value is not None:
            try:
                decimals = len(answer.split(".", 1)[1]) if "." in answer else 0
                computed_float = float(computed_value)
                if (
                    _format_numeric_like(computed_float, decimals) == answer
                    and computed_value != answer
                ):
                    constraints.append("use_clean_final_answer")
            except Exception:
                pass
        return constraints

    return []


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


def _criterion_lines(
    *,
    category: str,
    candidate_valid: bool,
    evidence: dict[str, str] | None,
) -> list[str]:
    failed_constraint = (evidence or {}).get("failed_constraint")
    if category == "bit_manipulation":
        criteria = {
            "binary_string": "1",
            "preserve_leading_zeros": "1",
            "exact_string_match": "1",
        }
        if failed_constraint == "binary_string":
            criteria["binary_string"] = "0"
            criteria["exact_string_match"] = "0"
        elif failed_constraint == "preserve_leading_zeros":
            criteria["preserve_leading_zeros"] = "0"
            criteria["exact_string_match"] = "0"
        elif failed_constraint == "exact_string_match":
            criteria["exact_string_match"] = "0"
        return [
            f"criterion.binary_string = {criteria['binary_string']}",
            f"criterion.preserve_leading_zeros = {criteria['preserve_leading_zeros']}",
            f"criterion.exact_string_match = {criteria['exact_string_match']}",
            f"criterion.all_pass = {'1' if candidate_valid else '0'}",
        ]

    if category == "gravity":
        criteria = {
            "numeric_answer": "1",
            "use_clean_final_answer": "1",
            "final_rounding_already_applied": "1",
        }
        if failed_constraint == "numeric_answer":
            criteria["numeric_answer"] = "0"
            criteria["use_clean_final_answer"] = "0"
            criteria["final_rounding_already_applied"] = "0"
        elif failed_constraint == "use_clean_final_answer":
            criteria["use_clean_final_answer"] = "0"
            criteria["final_rounding_already_applied"] = "0"
        elif failed_constraint == "final_rounding_already_applied":
            criteria["final_rounding_already_applied"] = "0"
        return [
            f"criterion.numeric_answer = {criteria['numeric_answer']}",
            f"criterion.use_clean_final_answer = {criteria['use_clean_final_answer']}",
            (
                "criterion.final_rounding_already_applied = "
                f"{criteria['final_rounding_already_applied']}"
            ),
            f"criterion.all_pass = {'1' if candidate_valid else '0'}",
        ]

    return []


def _verifier_lines(
    category: str,
    answer: str,
    *,
    candidate_answer: str | None = None,
    candidate_valid: bool = True,
    error_type: str | None = None,
    evidence: dict[str, str] | None = None,
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
                *(
                    f"{key} = {value}"
                    for key, value in (evidence or {}).items()
                ),
                f"corrected_answer = {answer}",
            ]
        )
    lines.extend(
        _criterion_lines(
            category=category,
            candidate_valid=candidate_valid,
            evidence=evidence,
        )
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
                "constraint.use_clean_final_answer = yes",
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
        lines.append("constraint.use_clean_final_answer = yes")
    return lines


def build_legacy_double_check_completion(
    completion_text: str,
    *,
    category: str,
    answer: str,
    problem_id: str | None = None,
    forced_failed_constraint: str | None = None,
) -> str:
    key = problem_id or completion_text or answer
    if category == "bit_manipulation":
        candidate_answer, error_type, evidence = _corrupt_binary_answer(
            answer,
            key,
            reasoning_text=completion_text,
            forced_failed_constraint=forced_failed_constraint,
        )
    elif category == "gravity":
        candidate_answer, error_type, evidence = _corrupt_numeric_answer(
            answer,
            key,
            reasoning_text=completion_text,
            forced_failed_constraint=forced_failed_constraint,
        )
    else:
        return completion_text

    lines = [
        "[Double_Check]",
        f"candidate_answer = {candidate_answer}",
        "candidate_valid = no",
        f"error_type = {error_type}",
        *(f"{key} = {value}" for key, value in evidence.items()),
        f"corrected_answer = {answer}",
    ]
    lines.extend(
        _criterion_lines(
            category=category,
            candidate_valid=False,
            evidence=evidence,
        )
    )
    lines.append("final_answer_source = double_check_corrected_answer")
    block = "\n".join(lines)

    if "</think>" in completion_text:
        head, tail = completion_text.rsplit("</think>", 1)
        return f"{head.rstrip()}\n\n{block}\n</think>{tail}"
    if "<|im_end|>" in completion_text:
        head, tail = completion_text.rsplit("<|im_end|>", 1)
        return f"{head.rstrip()}\n\n{block}<|im_end|>{tail}"
    return f"{completion_text.rstrip()}\n\n{block}"


def wrap_three_agent_reasoning(
    reasoning_text: str,
    *,
    category: str,
    answer: str,
    problem_id: str | None = None,
    solver_char_budget: int | None = None,
    negative_to_correct_rate: float = NEGATIVE_TO_CORRECT_RATE,
    forced_failed_constraint: str | None = None,
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
    evidence: dict[str, str] = {}
    if use_negative:
        if category == "bit_manipulation":
            candidate_answer, error_type, evidence = _corrupt_binary_answer(
                answer,
                problem_id or reasoning_text or answer,
                reasoning_text=reasoning_text,
                forced_failed_constraint=forced_failed_constraint,
            )
        elif category == "gravity":
            candidate_answer, error_type, evidence = _corrupt_numeric_answer(
                answer,
                problem_id or reasoning_text or answer,
                reasoning_text=reasoning_text,
                forced_failed_constraint=forced_failed_constraint,
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
    final_answer_source = (
        "verifier_corrected_answer"
        if use_negative and not candidate_valid
        else "verifier_candidate_answer"
    )
    lines.append("")
    lines.extend(
        _verifier_lines(
            category,
            answer,
            candidate_answer=candidate_answer,
            candidate_valid=candidate_valid,
            error_type=error_type,
            evidence=evidence,
        )
    )
    lines.append("")
    lines.extend(
        [
            "[Consensus_Agent]",
            f"selected_answer = {answer}",
            f"final_answer_source = {final_answer_source}",
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
    forced_failed_constraint: str | None = None,
) -> str:
    wrapped = wrap_three_agent_reasoning(
        reasoning_text,
        category=category,
        answer=answer,
        problem_id=problem_id,
        solver_char_budget=solver_char_budget,
        negative_to_correct_rate=negative_to_correct_rate,
        forced_failed_constraint=forced_failed_constraint,
    )
    return f"{wrapped}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
