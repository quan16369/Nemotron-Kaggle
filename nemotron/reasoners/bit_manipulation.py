"""Reasoning generator for 8-bit bit-manipulation tasks.

The output follows the legacy trace style used by the existing reasoning files,
with a strict-validity filter for candidate assignment vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

from reasoners.store_types import Problem

N_BITS = 8

SYM_FAMILIES = ("XOR", "OR", "AND")
ASYM_FAMILIES = ("AND-NOT", "XOR-NOT", "OR-NOT")
PAIR_FAMILIES = SYM_FAMILIES + ASYM_FAMILIES
UNARY_FAMILIES = ("I", "NOT")
CONSTANT_FAMILIES = ("0", "1")
DEFAULT_FAMILY: RuleFamily = "DEFAULT"
STRIDES: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)
SECTION_ORDER = (
    "Identity",
    "NOT",
    "Constant",
    "AND",
    "OR",
    "XOR",
    "AND-NOT",
    "OR-NOT",
    "XOR-NOT",
)

# Map section names to their constituent family codes.
_SECTION_TO_FAMILIES = {
    "Identity": ("I",),
    "NOT": ("NOT",),
    "Constant": ("0", "1"),
}

# Reverse map: family code → section name.
_FAMILY_TO_SECTION: dict[str, str] = {}
for _section in SECTION_ORDER:
    for _fam in _SECTION_TO_FAMILIES.get(_section, (_section,)):
        _FAMILY_TO_SECTION[_fam] = _section

WHOLE_WORD_GENERIC3_TRUTH = "11010110"
WHOLE_WORD_GENERIC3_ALIAS = "if c then NAND(a,b) else XNOR(a,b)"


def _rotate_left_word(value: int, shift: int) -> int:
    shift %= N_BITS
    return ((value << shift) | (value >> (N_BITS - shift))) & 0xFF


def _shift_left_word(value: int, shift: int) -> int:
    return (value << shift) & 0xFF


def _shift_right_word(value: int, shift: int) -> int:
    return (value >> shift) & 0xFF


def _build_whole_word_transforms() -> tuple[tuple[str, Callable[[int], int]], ...]:
    transforms: list[tuple[str, Callable[[int], int]]] = [
        ("I", lambda value: value),
        ("NOT", lambda value: value ^ 0xFF),
    ]
    for shift in range(1, N_BITS):
        transforms.append(
            (f"ROT({shift})", lambda value, shift=shift: _rotate_left_word(value, shift))
        )
    for shift in range(1, N_BITS):
        transforms.append(
            (f"SHL({shift})", lambda value, shift=shift: _shift_left_word(value, shift))
        )
    for shift in range(1, N_BITS):
        transforms.append(
            (f"SHR({shift})", lambda value, shift=shift: _shift_right_word(value, shift))
        )
    for shift in range(1, N_BITS):
        transforms.append(
            (
                f"NOT ROT({shift})",
                lambda value, shift=shift: _rotate_left_word(value, shift) ^ 0xFF,
            )
        )
    for shift in range(1, N_BITS):
        transforms.append(
            (
                f"NOT SHL({shift})",
                lambda value, shift=shift: _shift_left_word(value, shift) ^ 0xFF,
            )
        )
    for shift in range(1, N_BITS):
        transforms.append(
            (
                f"NOT SHR({shift})",
                lambda value, shift=shift: _shift_right_word(value, shift) ^ 0xFF,
            )
        )
    return tuple(transforms)


WHOLE_WORD_TRANSFORMS = _build_whole_word_transforms()
WHOLE_WORD_TRANSFORM_MAP = {name: fn for name, fn in WHOLE_WORD_TRANSFORMS}
WHOLE_WORD_COMBINERS: tuple[tuple[str, Callable[[int, int], int]], ...] = (
    ("XOR", lambda a, b: a ^ b),
    ("AND", lambda a, b: a & b),
    ("OR", lambda a, b: a | b),
)
WHOLE_WORD_MIXED_TRIPLES: tuple[
    tuple[str, str, Callable[[int, int, int], int]],
    ...
] = (
    ("(A XOR B) AND C", "({} XOR {}) AND {}", lambda a, b, c: (a ^ b) & c),
    ("(A AND B) XOR C", "({} AND {}) XOR {}", lambda a, b, c: (a & b) ^ c),
    ("(A OR B) XOR C", "({} OR {}) XOR {}", lambda a, b, c: (a | b) ^ c),
    ("(A XOR B) OR C", "({} XOR {}) OR {}", lambda a, b, c: (a ^ b) | c),
    ("(A AND B) OR C", "({} AND {}) OR {}", lambda a, b, c: (a & b) | c),
    ("(A OR B) AND C", "({} OR {}) AND {}", lambda a, b, c: (a | b) & c),
)


RuleFamily = Literal[
    "I",
    "NOT",
    "0",
    "1",
    "XOR",
    "OR",
    "AND",
    "AND-NOT",
    "XOR-NOT",
    "OR-NOT",
    "DEFAULT",
]


@dataclass(frozen=True)
class RuleCandidate:
    family: RuleFamily
    primary: Optional[int]
    secondary: Optional[int]
    expr: str
    primary_stride: Optional[int] = None
    secondary_stride: Optional[int] = None
    primary_offset: Optional[int] = (
        None  # primary at bit 0: primary = (offset + bit * stride) % 8
    )
    secondary_offset: Optional[int] = (
        None  # secondary at bit 0: secondary = (offset + bit * stride) % 8
    )

    @property
    def is_default(self) -> bool:
        return self.family == DEFAULT_FAMILY


@dataclass(frozen=True)
class Record:
    label: str
    col: str
    hash_: str
    matches: Tuple[int, ...]


@dataclass(frozen=True)
class WholeWordMatch:
    transform_names: Tuple[str, ...]
    formula_template: str
    rule_text: str
    answer_bits: str
    note: str = ""


def _normalize_bits(value: str) -> str:
    bits = "".join(ch for ch in str(value) if ch in {"0", "1"})
    if len(bits) != N_BITS:
        return ""
    return bits


def _format_byte(value: int) -> str:
    return format(value & 0xFF, "08b")


def _append_bit_examples(lines: List[str], label: str, values: Sequence[str]) -> None:
    for idx, bits in enumerate(values):
        lines.append(f"{label} {idx}: {bits}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {bits[bit]}")
        lines.append("")


def _append_bit_columns(lines: List[str], title: str, values: Sequence[str]) -> None:
    lines.append(title)
    for bit in range(N_BITS):
        column = _column_bits(values, bit)
        lines.append(f"{bit} {column} {_column_hash(column, len(values))}")


def _append_transform_audit(
    lines: List[str],
    label: str,
    transform_name: str,
    values: Sequence[str],
) -> None:
    _append_bit_examples(lines, f"{label} ({transform_name})", values)
    _append_bit_columns(
        lines,
        f"{label} ({transform_name}) bit columns (with bitsum as hash)",
        values,
    )


def _apply_majority_word(a: int, b: int, c: int) -> int:
    return (a & b) | (a & c) | (b & c)


def _apply_choice_word(a: int, b: int, c: int) -> int:
    return (a & b) | ((~a & 0xFF) & c)


def _apply_generic_truth_word(a: int, b: int, c: int, truth: str) -> int:
    out = 0
    for bit in range(N_BITS):
        combo = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
        if truth[combo] == "1":
            out |= 1 << bit
    return out


def _render_whole_word_reasoning(
    question_bits: str,
    inputs: Sequence[int],
    outputs: Sequence[int],
    match: WholeWordMatch,
    *,
    compact: bool,
) -> str:
    n_examples = len(inputs)
    query_value = int(question_bits, 2)
    labels = ("A", "B", "C")
    transformed_values = [
        (label, name, _format_byte(WHOLE_WORD_TRANSFORM_MAP[name](query_value)))
        for label, name in zip(labels, match.transform_names)
    ]

    if compact:
        lines = [
            "We need to deduce the transformation by testing whole-word transforms on the full 8-bit input.",
            f"RULE {match.rule_text}",
        ]
        if match.note:
            lines.append(f"NOTE {match.note}")
        lines.append(f"MATCH {n_examples}")
        lines.append(f"Q {question_bits}")
        for label, name, value in transformed_values:
            lines.append(f"{label} {name} {value}")
        lines.append(f"OUT {match.formula_template} {match.answer_bits}")
        lines.append(f"Final answer is: {match.answer_bits}")
        return "\n".join(lines)

    lines = [
        "We need to deduce the transformation by testing whole-word transforms on the full 8-bit input.",
        "I will compute the answer and state it at the end.",
        "",
    ]
    output_bits_list = [_format_byte(value) for value in outputs]
    input_bits_list = [_format_byte(value) for value in inputs]
    _append_bit_examples(lines, "Output", output_bits_list)
    _append_bit_columns(lines, "Output bit columns (with bitsum as hash)", output_bits_list)
    lines.append("")
    _append_bit_examples(lines, "Input", input_bits_list)
    transform_example_bits = []
    for label, transform_name in zip(labels, match.transform_names):
        values = [
            _format_byte(WHOLE_WORD_TRANSFORM_MAP[transform_name](input_value))
            for input_value in inputs
        ]
        transform_example_bits.append((label, transform_name, values))
    for label, transform_name, values in transform_example_bits:
        lines.append("")
        _append_transform_audit(lines, label, transform_name, values)
    lines.extend(
        [
            "",
            "Selected whole-word rule",
            match.rule_text,
        ]
    )
    if match.note:
        lines.append(match.note)
    lines.extend(
        [
            f"This rule matches all {n_examples} examples exactly.",
            "",
            "Verification on examples",
        ]
    )
    for example_idx, (input_value, output_value) in enumerate(zip(inputs, outputs), start=1):
        input_bits = _format_byte(input_value)
        output_bits = _format_byte(output_value)
        lines.append(f"EX{example_idx}: IN = {input_bits}")
        for label, transform_name in zip(labels, match.transform_names):
            transform_bits = _format_byte(WHOLE_WORD_TRANSFORM_MAP[transform_name](input_value))
            lines.append(f"EX{example_idx}: {label} = {transform_name} = {transform_bits}")
        lines.append(
            f"EX{example_idx}: OUT = {match.formula_template} = {output_bits}"
        )
        lines.append("")
    lines.extend(
        [
            f"Applying to {question_bits}",
            "Input",
        ]
    )
    for bit in range(N_BITS):
        lines.append(f"{bit} {question_bits[bit]}")
    lines.append("")
    lines.append("Transforms")
    for label, name, value in transformed_values:
        lines.append(f"{label} = {name} = {value}")
        for bit in range(N_BITS):
            lines.append(f"{label}{bit} {value[bit]}")
    lines.append("")
    lines.append(f"Output = {match.formula_template} = {match.answer_bits}")
    lines.append("Output")
    for bit in range(N_BITS):
        lines.append(f"{bit} {match.answer_bits[bit]}")
    lines.extend(
        [
            "",
            "I will now state the final answer.",
            f"Final answer is: {match.answer_bits}",
        ]
    )
    return "\n".join(lines)


def _solve_whole_word_rule(
    inputs: Sequence[int],
    outputs: Sequence[int],
    query: int,
) -> Optional[WholeWordMatch]:
    n_examples = len(inputs)
    n_transforms = len(WHOLE_WORD_TRANSFORMS)
    transformed_examples = [
        [fn(inp) for inp in inputs] for _, fn in WHOLE_WORD_TRANSFORMS
    ]

    # Existing whole-word search space from investigators/bit_manipulation.py.
    for idx in range(n_transforms):
        name, fn = WHOLE_WORD_TRANSFORMS[idx]
        if all(transformed_examples[idx][i] == outputs[i] for i in range(n_examples)):
            return WholeWordMatch(
                transform_names=(name,),
                formula_template="A",
                rule_text=name,
                answer_bits=_format_byte(fn(query)),
            )

    for op_name, op_fn in WHOLE_WORD_COMBINERS:
        for left in range(n_transforms):
            for right in range(left + 1, n_transforms):
                if not all(
                    op_fn(transformed_examples[left][i], transformed_examples[right][i])
                    == outputs[i]
                    for i in range(n_examples)
                ):
                    continue
                left_name, left_fn = WHOLE_WORD_TRANSFORMS[left]
                right_name, right_fn = WHOLE_WORD_TRANSFORMS[right]
                return WholeWordMatch(
                    transform_names=(left_name, right_name),
                    formula_template=f"A {op_name} B",
                    rule_text=f"{left_name} {op_name} {right_name}",
                    answer_bits=_format_byte(op_fn(left_fn(query), right_fn(query))),
                )

    for op_name, op_fn in WHOLE_WORD_COMBINERS:
        for first in range(n_transforms):
            for second in range(first + 1, n_transforms):
                pair = [
                    op_fn(transformed_examples[first][i], transformed_examples[second][i])
                    for i in range(n_examples)
                ]
                for third in range(second + 1, n_transforms):
                    if not all(op_fn(pair[i], transformed_examples[third][i]) == outputs[i] for i in range(n_examples)):
                        continue
                    first_name, first_fn = WHOLE_WORD_TRANSFORMS[first]
                    second_name, second_fn = WHOLE_WORD_TRANSFORMS[second]
                    third_name, third_fn = WHOLE_WORD_TRANSFORMS[third]
                    pair_query = op_fn(first_fn(query), second_fn(query))
                    return WholeWordMatch(
                        transform_names=(first_name, second_name, third_name),
                        formula_template=f"A {op_name} B {op_name} C",
                        rule_text=f"{first_name} {op_name} {second_name} {op_name} {third_name}",
                        answer_bits=_format_byte(op_fn(pair_query, third_fn(query))),
                    )

    for formula_template, rule_template, op_fn in WHOLE_WORD_MIXED_TRIPLES:
        for first in range(n_transforms):
            for second in range(first + 1, n_transforms):
                for third in range(n_transforms):
                    if third == first or third == second:
                        continue
                    if not all(
                        op_fn(
                            transformed_examples[first][i],
                            transformed_examples[second][i],
                            transformed_examples[third][i],
                        )
                        == outputs[i]
                        for i in range(n_examples)
                    ):
                        continue
                    first_name, first_fn = WHOLE_WORD_TRANSFORMS[first]
                    second_name, second_fn = WHOLE_WORD_TRANSFORMS[second]
                    third_name, third_fn = WHOLE_WORD_TRANSFORMS[third]
                    return WholeWordMatch(
                        transform_names=(first_name, second_name, third_name),
                        formula_template=formula_template,
                        rule_text=rule_template.format(first_name, second_name, third_name),
                        answer_bits=_format_byte(
                            op_fn(first_fn(query), second_fn(query), third_fn(query))
                        ),
                    )

    # Additional byte-level boolean families.
    for first in range(n_transforms):
        for second in range(n_transforms):
            if second == first:
                continue
            for third in range(n_transforms):
                if third == first or third == second:
                    continue
                if all(
                    _apply_choice_word(
                        transformed_examples[first][i],
                        transformed_examples[second][i],
                        transformed_examples[third][i],
                    )
                    == outputs[i]
                    for i in range(n_examples)
                ):
                    first_name, first_fn = WHOLE_WORD_TRANSFORMS[first]
                    second_name, second_fn = WHOLE_WORD_TRANSFORMS[second]
                    third_name, third_fn = WHOLE_WORD_TRANSFORMS[third]
                    return WholeWordMatch(
                        transform_names=(first_name, second_name, third_name),
                        formula_template="CH(A, B, C)",
                        rule_text=f"CH({first_name}, {second_name}, {third_name})",
                        answer_bits=_format_byte(
                            _apply_choice_word(
                                first_fn(query), second_fn(query), third_fn(query)
                            )
                        ),
                        note="CH(A, B, C) = (A & B) | (~A & C)",
                    )

    for first in range(n_transforms):
        for second in range(first + 1, n_transforms):
            for third in range(second + 1, n_transforms):
                if all(
                    _apply_majority_word(
                        transformed_examples[first][i],
                        transformed_examples[second][i],
                        transformed_examples[third][i],
                    )
                    == outputs[i]
                    for i in range(n_examples)
                ):
                    first_name, first_fn = WHOLE_WORD_TRANSFORMS[first]
                    second_name, second_fn = WHOLE_WORD_TRANSFORMS[second]
                    third_name, third_fn = WHOLE_WORD_TRANSFORMS[third]
                    return WholeWordMatch(
                        transform_names=(first_name, second_name, third_name),
                        formula_template="MAJ(A, B, C)",
                        rule_text=f"MAJ({first_name}, {second_name}, {third_name})",
                        answer_bits=_format_byte(
                            _apply_majority_word(
                                first_fn(query), second_fn(query), third_fn(query)
                            )
                        ),
                        note="MAJ(A, B, C) = (A & B) | (A & C) | (B & C)",
                    )

    for first in range(n_transforms):
        for second in range(n_transforms):
            if second == first:
                continue
            for third in range(n_transforms):
                if third == first or third == second:
                    continue
                if all(
                    _apply_generic_truth_word(
                        transformed_examples[first][i],
                        transformed_examples[second][i],
                        transformed_examples[third][i],
                        WHOLE_WORD_GENERIC3_TRUTH,
                    )
                    == outputs[i]
                    for i in range(n_examples)
                ):
                    first_name, first_fn = WHOLE_WORD_TRANSFORMS[first]
                    second_name, second_fn = WHOLE_WORD_TRANSFORMS[second]
                    third_name, third_fn = WHOLE_WORD_TRANSFORMS[third]
                    return WholeWordMatch(
                        transform_names=(first_name, second_name, third_name),
                        formula_template=f"GEN3[{WHOLE_WORD_GENERIC3_TRUTH}](A, B, C)",
                        rule_text=(
                            f"GEN3[{WHOLE_WORD_GENERIC3_TRUTH}]"
                            f"({first_name}, {second_name}, {third_name})"
                        ),
                        answer_bits=_format_byte(
                            _apply_generic_truth_word(
                                first_fn(query),
                                second_fn(query),
                                third_fn(query),
                                WHOLE_WORD_GENERIC3_TRUTH,
                            )
                        ),
                        note=(
                            f"GEN3[{WHOLE_WORD_GENERIC3_TRUTH}] = "
                            f"{WHOLE_WORD_GENERIC3_ALIAS}"
                        ),
                    )

    return None


def _column_bits(values: Sequence[str], bit: int) -> str:
    return "".join(v[bit] for v in values)


def _bit_not(bit: str) -> str:
    return "1" if bit == "0" else "0"


def _invert(bits: str) -> str:
    return "".join(_bit_not(b) for b in bits)


def _column_hash(bits: str, total_examples: int) -> str:
    ones = bits.count("1")
    if ones == 0 or ones == total_examples:
        return "a"
    return format(ones, "x")


def _evaluate_binary(a: str, b: str, family: str) -> str:
    if family in ("AND", "AND-NOT"):
        return "1" if a == "1" and b == "1" else "0"
    if family in ("OR", "OR-NOT"):
        return "1" if a == "1" or b == "1" else "0"
    if family in ("XOR", "XOR-NOT"):
        return "1" if a != b else "0"
    raise ValueError(f"Unsupported family {family}")


def _apply_family(
    a_bits: str, b_bits: str, family: str, invert_second: bool = False
) -> str:
    b_eff = _invert(b_bits) if invert_second else b_bits
    out = []
    for x, y in zip(a_bits, b_eff):
        out.append(_evaluate_binary(x, y, family))
    return "".join(out)


def _find_match(
    candidates: List[RuleCandidate], fam: str, ep: Optional[int], es: Optional[int]
) -> Optional[RuleCandidate]:
    """Find candidate matching (fam, ep, es) by direct lookup."""
    for c in candidates:
        if c.family != fam:
            continue
        if c.primary == ep and (fam not in PAIR_FAMILIES or c.secondary == es):
            return c
    return None


def _with_stride_metadata(
    candidate: RuleCandidate, p_step: int, s_step: int
) -> RuleCandidate:
    """Attach stride metadata so extrapolation can follow the chosen run."""
    return replace(
        candidate,
        primary_stride=p_step if candidate.primary is not None else None,
        secondary_stride=s_step if candidate.secondary is not None else None,
    )


def _exists_anywhere(
    all_matches: List[List[RuleCandidate]],
    fam: str,
    ep: Optional[int],
    es: Optional[int],
) -> bool:
    """Check if operand pair (ep, es) exists in any bit position for this family."""
    for bit_cands in all_matches:
        if _find_match(bit_cands, fam, ep, es) is not None:
            return True
    return False


def _fail_suffix(
    all_matches: List[List[RuleCandidate]],
    fam: str,
    ep: Optional[int],
    es: Optional[int],
) -> str:
    """Return 'y' if operand exists somewhere (wrong position), 'x' if nowhere."""
    if _exists_anywhere(all_matches, fam, ep, es):
        return "y"
    return "x"


def _find_all_left_runs(
    all_matches: List[List[RuleCandidate]],
) -> List[Tuple[List[RuleCandidate], Optional[str]]]:
    """All stride-consistent runs from bit 0, all stride combos per starter.

    Returns list of (chain, failed_next_expr) tuples.
    """
    if not all_matches or not all_matches[0]:
        return []
    runs: List[Tuple[List[RuleCandidate], Optional[str]]] = []
    for start_cand in all_matches[0]:
        fam = start_cand.family
        for p_step, s_step in STRIDES:
            chain = [_with_stride_metadata(start_cand, p_step, s_step)]
            # Track expected position independently (don't use found candidate's operands)
            cur_p = start_cand.primary
            cur_s = start_cand.secondary
            failed_next: Optional[str] = None
            for b in range(1, len(all_matches)):
                ep = (cur_p + p_step) % N_BITS if cur_p is not None else None
                es = (cur_s + s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, ep, es)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, ep, es)
                    if ep is not None and es is not None:
                        failed_next = f"{ep}{es}{suffix}"
                    elif ep is not None:
                        failed_next = f"{ep}{suffix}"
                    break
                chain.append(_with_stride_metadata(found, p_step, s_step))
                cur_p, cur_s = ep, es
            runs.append((chain, failed_next))
    return runs


def _find_all_right_runs(
    all_matches: List[List[RuleCandidate]],
) -> List[Tuple[List[RuleCandidate], Optional[str]]]:
    """All stride-consistent runs ending at last bit, all stride combos per ender.

    Returns list of (chain, failed_next_expr) tuples.
    """
    n = len(all_matches)
    if not all_matches or not all_matches[-1]:
        return []
    runs: List[Tuple[List[RuleCandidate], Optional[str]]] = []
    for end_cand in all_matches[-1]:
        fam = end_cand.family
        for p_step, s_step in STRIDES:
            chain = [_with_stride_metadata(end_cand, p_step, s_step)]
            # Track expected position independently
            cur_p = end_cand.primary
            cur_s = end_cand.secondary
            failed_next: Optional[str] = None
            for k in range(1, n):
                b = n - 1 - k
                pp = (cur_p - p_step) % N_BITS if cur_p is not None else None
                ps = (cur_s - s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, pp, ps)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, pp, ps)
                    if pp is not None and ps is not None:
                        failed_next = f"{pp}{ps}{suffix}"
                    elif pp is not None:
                        failed_next = f"{pp}{suffix}"
                    break
                chain.insert(0, _with_stride_metadata(found, p_step, s_step))
                cur_p, cur_s = pp, ps
            runs.append((chain, failed_next))
    return runs


def _lr_from_matches(
    all_matches: List[List[RuleCandidate]],
) -> Tuple[List[str], str, List[str], str]:
    """Compute Left/Right from full per-bit match lists.

    Returns (left_all_lines, left_best, right_all_lines, right_best).
    """
    all_left_runs = _find_all_left_runs(all_matches)
    all_right_runs = _find_all_right_runs(all_matches)
    left_run = max(all_left_runs, key=lambda t: len(t[0])) if all_left_runs else ([], None)
    right_run = max(all_right_runs, key=lambda t: len(t[0])) if all_right_runs else ([], None)

    left_lines = (
        [_format_list(chain, failed=failed) for chain, failed in all_left_runs]
        if all_left_runs
        else ["none"]
    )
    left_best = _format_list(left_run[0], with_count=True)
    right_lines = (
        [
            _format_list(list(reversed(chain)), failed=failed)
            for chain, failed in all_right_runs
        ]
        if all_right_runs
        else ["none"]
    )
    right_best = _format_list(list(reversed(right_run[0])), with_count=True)

    return left_lines, left_best, right_lines, right_best


def _format_list(
    cands: List[RuleCandidate],
    with_count: bool = False,
    failed: Optional[str] = None,
) -> str:
    if not cands:
        return "none"
    if with_count:
        parts = []
        for i, c in enumerate(cands):
            if i == 0:
                parts.append(c.expr)
            else:
                parts.append(_compact_rule(c))
        return " ".join(parts) + f": {len(cands)}"
    parts = [_compact_rule(c) for c in cands]
    if failed:
        parts.append(failed)
    return " ".join(parts)


def _compact_rule(c: RuleCandidate) -> str:
    """Compact display: just the operand indices without family prefix."""
    if c.primary is not None and c.secondary is not None:
        return f"{c.primary}{c.secondary}"
    if c.primary is not None:
        return str(c.primary)
    return c.family


_T3_TO_CODE = {
    "AND": "A",
    "OR": "O",
    "XOR": "X",
}
_T3_FROM_CODE = {code: name for name, code in _T3_TO_CODE.items()}


def _make_t3_expr(
    i: int,
    j: int,
    k: int,
    op1: str,
    op2: str,
    neg_mask: int,
    assoc: int,
) -> str:
    """Encode a 3-bit repair rule with a token-friendlier compact string."""
    assoc_code = "L" if assoc == 0 else "R"
    return f"T3{assoc_code}{i}{j}{k}{_T3_TO_CODE[op1]}{_T3_TO_CODE[op2]}{neg_mask}"


def _parse_t3_expr(expr: str) -> tuple[int, int, int, str, str, int, int]:
    """Decode a compact T3 expression created by _make_t3_expr."""
    if len(expr) != 9 or not expr.startswith("T3"):
        raise ValueError(f"Invalid T3 expr {expr}")
    assoc = 0 if expr[2] == "L" else 1
    return (
        int(expr[3]),
        int(expr[4]),
        int(expr[5]),
        _T3_FROM_CODE[expr[6]],
        _T3_FROM_CODE[expr[7]],
        int(expr[8]),
        assoc,
    )


def _rule_display(rule: RuleCandidate) -> str:
    """Stable display for legacy and compact renderers."""
    return rule.expr


def _evaluate_rule(bits: str, rule: RuleCandidate) -> str:
    if rule.family == "DEFAULT":
        return "1"
    if rule.family == "0":
        return "0"
    if rule.family == "1":
        return "1"
    if rule.family == "I":
        assert rule.primary is not None
        return bits[rule.primary]
    if rule.family == "NOT":
        assert rule.primary is not None
        return _bit_not(bits[rule.primary])
    if rule.family in PAIR_FAMILIES:
        assert rule.primary is not None and rule.secondary is not None
        a = bits[rule.primary]
        b = bits[rule.secondary]
        if "-NOT" in rule.family:
            b = _bit_not(b)
        return _evaluate_binary(a, b, rule.family)
    raise ValueError(f"Unknown family {rule.family}")


def _eval_three_bit_expr(
    a: str,
    b: str,
    c: str,
    op1: str,
    op2: str,
    neg_mask: int,
    assoc: int,
) -> str:
    xa = _bit_not(a) if (neg_mask & 1) else a
    xb = _bit_not(b) if (neg_mask & 2) else b
    xc = _bit_not(c) if (neg_mask & 4) else c
    if assoc == 0:
        left = _evaluate_binary(xa, xb, op1)
        return _evaluate_binary(left, xc, op2)
    right = _evaluate_binary(xb, xc, op2)
    return _evaluate_binary(xa, right, op1)


def _evaluate_extended_rule(bits: str, rule: RuleCandidate) -> str:
    """Evaluate a normal rule or a compact 3-bit repair rule."""
    if rule.expr.startswith("T3"):
        i, j, k, op1, op2, neg_mask, assoc = _parse_t3_expr(rule.expr)
        return _eval_three_bit_expr(
            bits[i], bits[j], bits[k], op1, op2, neg_mask, assoc
        )
    return _evaluate_rule(bits, rule)


def _emit_apply(
    lines: List[str], question_bits: str, vector: List[RuleCandidate]
) -> None:
    lines.append(f"Applying to {question_bits}")
    lines.append("Input")
    for i, bit in enumerate(question_bits):
        lines.append(f"{i} {bit}")
    lines.append("Output")

    answer_bits: List[str] = []
    for i, rule in enumerate(vector):
        result = _evaluate_extended_rule(question_bits, rule)
        if rule.family == "DEFAULT":
            lines.append(f"{i} default 1 = 1")
            answer_bits.append(result)
            continue
        if rule.family in CONSTANT_FAMILIES:
            lines.append(f"{i} {rule.expr} = {rule.family}")
            answer_bits.append(result)
            continue
        if rule.family == "I":
            lines.append(f"{i} {rule.expr} = {result}")
            answer_bits.append(result)
            continue
        if rule.family == "NOT":
            assert rule.primary is not None
            val = question_bits[rule.primary]
            lines.append(f"{i} {rule.expr} = NOT({val}) = {result}")
            answer_bits.append(result)
            continue
        if rule.expr.startswith("T3"):
            lines.append(f"{i} {rule.expr} = {result}")
            answer_bits.append(result)
            continue

        assert rule.primary is not None and rule.secondary is not None
        a = question_bits[rule.primary]
        b = question_bits[rule.secondary]
        if rule.family in SYM_FAMILIES:
            lines.append(f"{i} {rule.expr} = {rule.family}({a},{b}) = {result}")
            answer_bits.append(result)
            continue

        base = rule.family.split("-")[0]
        lines.append(f"{i} {rule.expr} = {base}({a},NOT({b})) = {result}")
        answer_bits.append(result)

    lines.append("")
    lines.append("I will now state the final answer.")
    lines.append(f"Final answer is: {''.join(answer_bits)}")


def _emit_apply_compact(
    lines: List[str], question_bits: str, vector: List[RuleCandidate]
) -> None:
    """Compact apply block used only when compact=True."""
    answer_bits: List[str] = []
    lines.append(f"Q {question_bits}")
    for i, rule in enumerate(vector):
        result = _evaluate_extended_rule(question_bits, rule)
        lines.append(f"B{i} {_rule_display(rule)} {result}")
        answer_bits.append(result)
    answer = "".join(answer_bits)
    lines.append(f"Final answer is: {answer}")


def _circular_distance(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, N_BITS - diff)


def _neighbor_hint_indices(vector: Sequence[RuleCandidate], bit_idx: int) -> tuple[int, ...]:
    """Collect nearby operand indices to bias local repair toward the current pattern."""
    hints: list[int] = []
    for direction in (-1, 1):
        pos = bit_idx + direction
        while 0 <= pos < len(vector):
            rule = vector[pos]
            if not rule.is_default:
                if rule.primary is not None:
                    hints.append(rule.primary)
                if rule.secondary is not None:
                    hints.append(rule.secondary)
                break
            pos += direction
    return tuple(hints)


def _build_three_bit_column(
    inputs: Sequence[str],
    i: int,
    j: int,
    k: int,
    op1: str,
    op2: str,
    neg_mask: int,
    assoc: int,
) -> str:
    out: list[str] = []
    for bits in inputs:
        out.append(
            _eval_three_bit_expr(
                bits[i], bits[j], bits[k], op1, op2, neg_mask, assoc
            )
        )
    return "".join(out)


def _score_three_bit_candidate(
    bit_idx: int,
    i: int,
    j: int,
    k: int,
    op1: str,
    op2: str,
    neg_mask: int,
    assoc: int,
    hint_indices: Sequence[int],
) -> tuple[int, int, int, int, int, int, int, int, int]:
    """Prefer simpler, less arbitrary local repairs."""
    neg_count = (neg_mask & 1) + ((neg_mask >> 1) & 1) + ((neg_mask >> 2) & 1)
    mixed_penalty = 0 if op1 == op2 else 1
    span = max(
        _circular_distance(i, j),
        _circular_distance(i, k),
        _circular_distance(j, k),
    )
    if hint_indices:
        hint_penalty = sum(
            min(_circular_distance(idx, hint) for hint in hint_indices)
            for idx in (i, j, k)
        )
    else:
        hint_penalty = (
            _circular_distance(bit_idx, i)
            + _circular_distance(bit_idx, j)
            + _circular_distance(bit_idx, k)
        )
    return (
        neg_count,
        mixed_penalty,
        hint_penalty,
        span,
        assoc,
        i,
        j,
        k,
        neg_mask,
    )


def _repair_with_three_bit_search(
    inputs: Sequence[str],
    output_columns: Sequence[str],
    vector: List[RuleCandidate],
) -> List[RuleCandidate]:
    """Repair unresolved bits with a guarded local 3-bit search.

    This stays conservative: only DEFAULT slots are considered, and we only
    apply a repair when there is a unique best-scoring candidate.
    """
    repaired = list(vector)
    for bit_idx, rule in enumerate(repaired):
        if not rule.is_default:
            continue

        target = output_columns[bit_idx]
        hint_indices = _neighbor_hint_indices(repaired, bit_idx)
        matches: list[tuple[tuple[int, int, int, int, int, int, int, int, int], RuleCandidate]] = []

        for i in range(N_BITS):
            for j in range(N_BITS):
                if j == i:
                    continue
                for k in range(N_BITS):
                    if k == i or k == j:
                        continue
                    for op1 in SYM_FAMILIES:
                        for op2 in SYM_FAMILIES:
                            for neg_mask in range(8):
                                for assoc in (0, 1):
                                    col = _build_three_bit_column(
                                        inputs, i, j, k, op1, op2, neg_mask, assoc
                                    )
                                    if col != target:
                                        continue
                                    score = _score_three_bit_candidate(
                                        bit_idx,
                                        i,
                                        j,
                                        k,
                                        op1,
                                        op2,
                                        neg_mask,
                                        assoc,
                                        hint_indices,
                                    )
                                    expr = _make_t3_expr(
                                        i, j, k, op1, op2, neg_mask, assoc
                                    )
                                    matches.append(
                                        (
                                            score,
                                            RuleCandidate(
                                                "XOR",
                                                i,
                                                j,
                                                expr,
                                            ),
                                        )
                                    )

        if not matches:
            continue

        matches.sort(key=lambda item: (item[0], item[1].expr))
        best_score, best_rule = matches[0]
        tied = [rule for score, rule in matches if score == best_score]
        if len(tied) == 1:
            repaired[bit_idx] = best_rule

    return repaired


def _render_reasoning_compact(
    question_bits: str,
    output_columns: Sequence[str],
    n_examples: int,
    all_matches: Dict[str, List[List[RuleCandidate]]],
    left_winner_text: str,
    right_winner_text: str,
    best: List[RuleCandidate],
) -> str:
    """Shorter trace format for optional compact SFT generation."""
    lines: List[str] = []
    lines.append("We need to deduce the transformation by matching the example outputs.")
    hashes = " ".join(
        f"{bit}:{_column_hash(output_columns[bit], n_examples)}" for bit in range(N_BITS)
    )
    lines.append(f"COL {hashes}")

    section_codes = {
        "Identity": "I",
        "NOT": "N",
        "Constant": "C",
        "AND": "A",
        "OR": "O",
        "XOR": "X",
        "AND-NOT": "D",
        "OR-NOT": "R",
        "XOR-NOT": "Z",
    }
    for name in SECTION_ORDER:
        items: list[str] = []
        for bit, cands in enumerate(all_matches[name]):
            if not cands:
                continue
            uniq = list(dict.fromkeys(_compact_rule(c) for c in cands))
            items.append(f"{bit}:{'|'.join(uniq)}")
        if items:
            lines.append(f"M {section_codes[name]} " + " ".join(items))

    lines.append(f"L {left_winner_text}")
    lines.append(f"R {right_winner_text}")
    lines.append("SEL " + " ".join(f"{i}={_rule_display(rule)}" for i, rule in enumerate(best)))
    _emit_apply_compact(lines, question_bits, best)
    return "\n".join(lines)


def _iter_unique_candidates_for_bit(
    all_matches: Dict[str, List[List[RuleCandidate]]],
    bit_idx: int,
) -> List[RuleCandidate]:
    seen: set[str] = set()
    ordered: list[RuleCandidate] = []
    for section_name in SECTION_ORDER:
        for candidate in all_matches[section_name][bit_idx]:
            if candidate.expr in seen:
                continue
            seen.add(candidate.expr)
            ordered.append(candidate)
    return ordered


def _candidate_priority(rule: RuleCandidate) -> tuple[int, int, int, int, str]:
    if rule.expr.startswith("T3"):
        section_rank = len(SECTION_ORDER)
    else:
        section_rank = SECTION_ORDER.index(_FAMILY_TO_SECTION.get(rule.family, "XOR"))
    arity = 0
    if rule.primary is not None:
        arity += 1
    if rule.secondary is not None:
        arity += 1
    return (
        section_rank,
        arity,
        1 if rule.expr.startswith("T3") else 0,
        len(rule.expr),
        rule.expr,
    )


def _build_whole_word_guided_vector(
    question_bits: str,
    answer_bits: str,
    inputs: Sequence[str],
    output_columns: Sequence[str],
    all_matches: Dict[str, List[List[RuleCandidate]]],
) -> Optional[List[RuleCandidate]]:
    default_cand = RuleCandidate(DEFAULT_FAMILY, None, None, "default 1")
    repaired = _repair_with_three_bit_search(
        inputs, output_columns, [default_cand] * N_BITS
    )
    guided: list[RuleCandidate] = []
    for bit_idx in range(N_BITS):
        candidates = _iter_unique_candidates_for_bit(all_matches, bit_idx)
        repaired_rule = repaired[bit_idx]
        if not repaired_rule.is_default and all(
            repaired_rule.expr != candidate.expr for candidate in candidates
        ):
            candidates.append(repaired_rule)
        filtered = [
            candidate
            for candidate in candidates
            if _evaluate_extended_rule(question_bits, candidate) == answer_bits[bit_idx]
        ]
        if not filtered:
            return None
        filtered.sort(key=_candidate_priority)
        guided.append(filtered[0])
    return guided


def _render_reasoning_legacy_guided(
    question_bits: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    output_columns: Sequence[str],
    all_records: Dict[str, List[Record]],
    all_matches: Dict[str, List[List[RuleCandidate]]],
    best: List[RuleCandidate],
    *,
    whole_word_rule: str,
    whole_word_note: str = "",
) -> str:
    n_examples = len(outputs)
    lines: List[str] = []
    lines.append(
        "We need to deduce the transformation by matching the example outputs."
    )
    lines.append("I will compute the answer and state it at the end.")
    lines.append("")

    for i, out in enumerate(outputs):
        lines.append(f"Output {i}: {out}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {out[bit]}")
        lines.append("")

    lines.append("Output bit columns (with bitsum as hash)")
    for bit in range(N_BITS):
        lines.append(
            f"{bit} {output_columns[bit]} {_column_hash(output_columns[bit], n_examples)}"
        )

    lines.append("")
    for i, inp in enumerate(inputs):
        lines.append(f"Input {i}: {inp}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {inp[bit]}")
        lines.append("")

    lines.append("When matching output")
    lines.append("x: not in operator")
    lines.append("y: wrong position")
    lines.append("")

    def _add_section(name: str) -> None:
        records = all_records[name]
        per_bit = all_matches[name]
        lines.append(name)
        prev_diff = None
        for rec in records:
            if (
                len(rec.label) >= 2
                and rec.label[0].isdigit()
                and rec.label[1].isdigit()
            ):
                diff = (int(rec.label[1]) - int(rec.label[0])) % N_BITS
                if prev_diff is not None and diff != prev_diff:
                    lines.append("")
                prev_diff = diff
            line = f"{rec.label} {rec.col} {rec.hash_}"
            if rec.matches:
                line += " match " + " ".join(str(i) for i in rec.matches)
            lines.append(line)
        lines.append("")
        lines.append("Matching output")
        for i in range(N_BITS):
            cands = per_bit[i]
            if cands:
                lines.append(f"{i} " + " ".join(_compact_rule(c) for c in cands))
            else:
                lines.append(f"{i} absent")
        lines.append("")
        left_lines, left_best, right_lines, right_best = _lr_from_matches(per_bit)
        lines.append("Left")
        for ll in left_lines:
            lines.append(ll)
        lines.append(f"Best: {left_best}")
        lines.append("")
        lines.append("Right")
        for rl in right_lines:
            lines.append(rl)
        lines.append(f"Best: {right_best}")
        lines.append("")

    for name in all_records:
        _add_section(name)

    lines.append("Selecting")
    lines.append("")
    lines.append("Whole-word guide")
    lines.append(whole_word_rule)
    if whole_word_note:
        lines.append(whole_word_note)
    lines.append("")
    lines.append("Selected")
    for i, rule in enumerate(best):
        lines.append(f"{i} {_rule_display(rule)}")
    lines.append("")
    _emit_apply(lines, question_bits, best)
    return "\n".join(lines)


def reasoning_bit_manipulation(
    problem: Problem,
    *,
    compact: bool = False,
    enable_three_bit_repair: bool = False,
    allow_whole_word: bool = True,
) -> Optional[str]:
    examples = problem.examples
    if not examples:
        return None

    outputs = [_normalize_bits(ex.output_value) for ex in examples]
    inputs = [_normalize_bits(ex.input_value) for ex in examples]
    question_bits = _normalize_bits(problem.question)

    if any(not bits for bits in outputs + inputs) or not question_bits:
        return None

    if len(outputs[0]) != N_BITS or len(inputs[0]) != N_BITS:
        return None

    if len(outputs) != len(inputs):
        return None

    n_examples = len(outputs)
    input_values = [int(bits, 2) for bits in inputs]
    output_values = [int(bits, 2) for bits in outputs]
    query_value = int(question_bits, 2)

    whole_word_match = None
    if allow_whole_word:
        whole_word_match = _solve_whole_word_rule(
            input_values, output_values, query_value
        )

    # 1) Example columns.
    output_columns = [_column_bits(outputs, i) for i in range(N_BITS)]
    input_columns = [_column_bits(inputs, i) for i in range(N_BITS)]
    input_inverted = [_invert(col) for col in input_columns]

    all_records: Dict[str, List[Record]] = {name: [] for name in SECTION_ORDER}
    all_matches: Dict[str, List[List[RuleCandidate]]] = {
        name: [[] for _ in range(N_BITS)] for name in SECTION_ORDER
    }

    # Build unary records and matches.
    for out_idx, out_col in enumerate(output_columns):
        for i_col, in_col in enumerate(input_columns):
            if in_col == out_col:
                all_matches["Identity"][out_idx].append(
                    RuleCandidate("I", i_col, None, f"I{i_col}")
                )
            if input_inverted[i_col] == out_col:
                all_matches["NOT"][out_idx].append(
                    RuleCandidate("NOT", i_col, None, f"NOT{i_col}")
                )
        if out_col.count("1") == 0:
            all_matches["Constant"][out_idx].append(
                RuleCandidate("0", None, None, "C0")
            )
        if out_col.count("1") == n_examples:
            all_matches["Constant"][out_idx].append(
                RuleCandidate("1", None, None, "C1")
            )

    # Build unary raw records.
    for label, col in zip([str(i) for i in range(N_BITS)], input_columns):
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["Identity"].append(
            Record(
                label=label,
                col=col,
                hash_=_column_hash(col, n_examples),
                matches=matches,
            )
        )
    for label, col in zip([str(i) for i in range(N_BITS)], input_inverted):
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["NOT"].append(
            Record(
                label=label,
                col=col,
                hash_=_column_hash(col, n_examples),
                matches=matches,
            )
        )
    for val in ("0", "1"):
        col = val * n_examples
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["Constant"].append(
            Record(
                label=val, col=col, hash_=_column_hash(col, n_examples), matches=matches
            )
        )

    # Build pair records (ordered by circular difference for symmetric ops).
    fam: RuleFamily
    for fam in ("XOR", "OR", "AND"):
        for circ_diff in range(1, N_BITS // 2 + 1):
            # For circ_diff == N_BITS/2, only half the circle to avoid duplicates
            n_pairs = N_BITS // 2 if circ_diff == N_BITS // 2 else N_BITS
            for a in range(n_pairs):
                b = (a + circ_diff) % N_BITS
                # Canonical pair for the operation: smaller index first
                lo, hi = min(a, b), max(a, b)
                col = _apply_family(input_columns[lo], input_columns[hi], fam)
                matches = tuple(
                    i for i, out_col in enumerate(output_columns) if col == out_col
                )
                all_records[fam].append(
                    Record(
                        label=f"{a}{b} {b}{a}",
                        col=col,
                        hash_=_column_hash(col, n_examples),
                        matches=matches,
                    )
                )
                for out_idx in matches:
                    all_matches[fam][out_idx].append(
                        RuleCandidate(fam, a, b, f"{fam}{a}{b}")
                    )
                    all_matches[fam][out_idx].append(
                        RuleCandidate(fam, b, a, f"{fam}{b}{a}")
                    )

    for fam in ("AND-NOT", "XOR-NOT", "OR-NOT"):
        for diff in range(1, N_BITS):
            for a in range(N_BITS):
                b = (a + diff) % N_BITS
                col = _apply_family(
                    input_columns[a], input_columns[b], fam, invert_second=True
                )
                matches = tuple(
                    i for i, out_col in enumerate(output_columns) if col == out_col
                )
                all_records[fam].append(
                    Record(
                        label=f"{a}{b}",
                        col=col,
                        hash_=_column_hash(col, n_examples),
                        matches=matches,
                    )
                )
                for out_idx in matches:
                    all_matches[fam][out_idx].append(
                        RuleCandidate(fam, a, b, f"{fam}{a}{b}")
                    )

    # Deterministic order for unary/constant records (pair records already ordered by diff).
    for name in ("Identity", "NOT", "Constant"):
        all_records[name].sort(key=lambda r: r.label)

    lines: List[str] = []

    # 1) header
    lines.append(
        "We need to deduce the transformation by matching the example outputs."
    )
    lines.append("I will compute the answer and state it at the end.")
    lines.append("")

    # 2) output examples
    for i, out in enumerate(outputs):
        lines.append(f"Output {i}: {out}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {out[bit]}")
        lines.append("")

    # 3) output bit columns
    lines.append("Output bit columns (with bitsum as hash)")
    for bit in range(N_BITS):
        lines.append(
            f"{bit} {output_columns[bit]} {_column_hash(output_columns[bit], n_examples)}"
        )

    # 4) input examples
    lines.append("")
    for i, inp in enumerate(inputs):
        lines.append(f"Input {i}: {inp}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {inp[bit]}")
        lines.append("")

    # 5) Operation sections (raw data + matching + LRM)
    lines.append("When matching output")
    lines.append("x: not in operator")
    lines.append("y: wrong position")
    lines.append("")
    section_lefts: list[tuple[str, str]] = []  # (name, left_best)
    section_rights: list[tuple[str, str]] = []  # (name, right_best)

    def _add_section(name: str) -> None:
        records = all_records[name]
        per_bit = all_matches[name]
        # Raw data
        lines.append(name)
        prev_diff = None
        for rec in records:
            # Insert blank line between diff groups for pair operations
            if (
                len(rec.label) >= 2
                and rec.label[0].isdigit()
                and rec.label[1].isdigit()
            ):
                diff = (int(rec.label[1]) - int(rec.label[0])) % N_BITS
                if prev_diff is not None and diff != prev_diff:
                    lines.append("")
                prev_diff = diff
            line = f"{rec.label} {rec.col} {rec.hash_}"
            if rec.matches:
                line += " match " + " ".join(str(i) for i in rec.matches)
            lines.append(line)
        lines.append("")
        # Matching: per output bit, which candidates match
        lines.append("Matching output")
        for i in range(N_BITS):
            cands = per_bit[i]
            if cands:

                def _compact(c: RuleCandidate) -> str:
                    if c.primary is not None and c.secondary is not None:
                        return f"{c.primary}{c.secondary}"
                    if c.primary is not None:
                        return str(c.primary)
                    return c.expr

                lines.append(f"{i} " + " ".join(_compact(c) for c in cands))
            else:
                lines.append(f"{i} absent")
        lines.append("")
        left_lines, left_best, right_lines, right_best = _lr_from_matches(per_bit)
        section_lefts.append((name, left_best))
        section_rights.append((name, right_best))
        lines.append("Left")
        for ll in left_lines:
            lines.append(ll)
        lines.append(f"Best: {left_best}")
        lines.append("")
        lines.append("Right")
        for rl in right_lines:
            lines.append(rl)
        lines.append(f"Best: {right_best}")
        lines.append("")

    for name in all_records:
        _add_section(name)

    if whole_word_match is not None and not compact:
        guided_vector = _build_whole_word_guided_vector(
            question_bits=question_bits,
            answer_bits=whole_word_match.answer_bits,
            inputs=inputs,
            output_columns=output_columns,
            all_matches=all_matches,
        )
        if guided_vector is not None:
            return _render_reasoning_legacy_guided(
                question_bits=question_bits,
                inputs=inputs,
                outputs=outputs,
                output_columns=output_columns,
                all_records=all_records,
                all_matches=all_matches,
                best=guided_vector,
                whole_word_rule=whole_word_match.rule_text,
                whole_word_note=whole_word_match.note,
            )
        return _render_whole_word_reasoning(
            question_bits,
            input_values,
            output_values,
            whole_word_match,
            compact=compact,
        )

    # 7) Selecting rule block.
    lines.append("Selecting")
    lines.append("")

    # Pick winners from per-section analysis
    def _parse_count(val: str) -> int:
        if val == "none":
            return 0
        try:
            return int(val.rsplit(": ", 1)[-1])
        except ValueError:
            return 0

    def _pick_winner(
        entries: list[tuple[str, str]],
    ) -> tuple[Optional[str], str, int]:
        best_name: Optional[str] = None
        best_text = "none"
        best_count = 0
        for name, val in entries:
            count = _parse_count(val)
            if count > best_count:
                best_count = count
                best_name = name
                best_text = val
        return best_name, best_text, best_count

    left_winner_name, left_winner_text, left_winner_count = _pick_winner(section_lefts)
    right_winner_name, right_winner_text, right_winner_count = _pick_winner(
        section_rights
    )

    # Get the actual left/right runs from per-section matches
    def _get_section_run(
        winner_name: Optional[str], direction: str
    ) -> List[RuleCandidate]:
        if winner_name is None:
            return []
        per_bit = all_matches[winner_name]
        if direction == "left":
            runs = _find_all_left_runs(per_bit)
        else:
            runs = _find_all_right_runs(per_bit)
        if not runs:
            return []
        best_chain, _ = max(runs, key=lambda t: len(t[0]))
        return best_chain

    left_run = _get_section_run(left_winner_name, "left")
    right_run = _get_section_run(right_winner_name, "right")

    lines.append("Lefts")
    for name, lb in section_lefts:
        lines.append(f"{name} {lb}")
    lines.append("")
    lines.append("Rights")
    for name, rb in section_rights:
        lines.append(f"{name} {rb}")
    lines.append("")
    lines.append(f"Left longest: {left_winner_count}")
    lines.append(f"Right longest: {right_winner_count}")
    lines.append("")

    def _matching_line(
        label: str,
        winner_name: Optional[str],
        entries: list[tuple[str, str]],
    ) -> str:
        parts = []
        for name, _val in entries:
            parts.append(f"{name} {'yes' if name == winner_name else 'no'}")
        return f"{label} winner: {', '.join(parts)}"

    if right_winner_count > left_winner_count:
        lines.append(_matching_line("Right", right_winner_name, section_rights))
        lines.append(_matching_line("Left", left_winner_name, section_lefts))
        lines.append("")
        lines.append(f"Best right: {right_winner_text}")
        lines.append(f"Best left: {left_winner_text}")
    else:
        lines.append(_matching_line("Left", left_winner_name, section_lefts))
        lines.append(_matching_line("Right", right_winner_name, section_rights))
        lines.append("")
        lines.append(f"Best left: {left_winner_text}")
        lines.append(f"Best right: {right_winner_text}")
    lines.append("")

    # Truncate if left + right > N_BITS: shorten the shorter one
    left_len_final = left_winner_count
    right_len_final = right_winner_count
    if left_len_final + right_len_final > N_BITS:
        if right_len_final > left_len_final:
            left_len_final = N_BITS - right_len_final
            left_run = left_run[:left_len_final]
        else:
            right_len_final = N_BITS - left_len_final
            right_run = right_run[-right_len_final:] if right_len_final else []
    left_was_truncated = left_len_final < left_winner_count
    right_was_truncated = right_len_final < right_winner_count
    trunc_left = f"Truncated left: {_format_list(left_run, with_count=True)}"
    if left_was_truncated:
        trunc_left += " truncated"
    trunc_right = f"Truncated right: {_format_list(list(reversed(right_run)), with_count=True)}"
    if right_was_truncated:
        trunc_right += " truncated"
    if right_winner_count > left_winner_count:
        lines.append(trunc_right)
        lines.append(trunc_left)
    else:
        lines.append(trunc_left)
        lines.append(trunc_right)
    lines.append("")

    right_start_final = N_BITS - right_len_final
    lines.append("Tentative from right")
    for i in range(N_BITS - 1, -1, -1):
        if i >= right_start_final and right_run:
            lines.append(f"{i} {right_run[i - right_start_final].expr}")
        else:
            lines.append(f"{i} pending")
    lines.append("")
    lines.append("Tentative")
    for i in range(N_BITS):
        if i < left_len_final:
            lines.append(f"{i} {left_run[i].expr}")
        elif i >= right_start_final and right_run:
            lines.append(f"{i} {right_run[i - right_start_final].expr}")
        else:
            lines.append(f"{i} pending")
    lines.append("")

    # Preferred: extrapolate left/right strides into pending slots
    def _extrap_from(
        run: List[RuleCandidate],
        bit: int,
        run_start_bit: int,
        side: str = "left",
    ) -> Optional[str]:
        if not run:
            return None
        r = run[0]
        # Derive offset from first candidate's position at run_start_bit
        # offset = primary - run_start_bit * stride (mod N_BITS)
        p = r.primary
        s = r.secondary
        p_step = r.primary_stride if r.primary_stride is not None else 1
        s_step = r.secondary_stride if r.secondary_stride is not None else 1
        if p is not None:
            p_off = (p - run_start_bit * p_step) % N_BITS
            ep = (p_off + bit * p_step) % N_BITS
        else:
            ep = None
        if s is not None:
            s_off = (s - run_start_bit * s_step) % N_BITS
            es = (s_off + bit * s_step) % N_BITS
        else:
            es = None
        if ep is not None and es is not None:
            return f"?{ep}{es}"
        if ep is not None:
            # Unary: show which slot is known
            if side == "left":
                return f"?{ep}?"
            else:
                return f"??{ep}"
        return None

    left_fam = left_run[0].family if left_run else None
    right_fam = right_run[0].family if right_run else None
    left_is_const = left_fam in CONSTANT_FAMILIES if left_fam else False
    right_is_const = right_fam in CONSTANT_FAMILIES if right_fam else False
    left_is_binary = left_fam in PAIR_FAMILIES if left_fam else False
    right_is_binary = right_fam in PAIR_FAMILIES if right_fam else False
    left_is_unary = left_fam in UNARY_FAMILIES if left_fam else False
    right_is_unary = right_fam in UNARY_FAMILIES if right_fam else False

    # Preferred: extrapolate from the longer side first, then fill from the other
    if right_winner_count > left_winner_count:
        # Right is longer: extrapolate from right first
        preferred: list[str] = []
        for i in range(N_BITS):
            if i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif i < left_len_final:
                preferred.append(left_run[i].expr)
            elif right_is_binary or right_is_unary:
                preferred.append(
                    _extrap_from(right_run, i, right_start_final, "right") or "pending"
                )
            else:
                preferred.append("pending")

        lines.append("Preferred from right")
        for i in range(N_BITS - 1, -1, -1):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")

        # Fill remaining pending from left; merge unary digits
        for i in range(N_BITS):
            if preferred[i] == "pending":
                if left_is_binary or left_is_unary:
                    preferred[i] = _extrap_from(left_run, i, 0, "left") or "?"
                else:
                    preferred[i] = "?"
            elif "?" in preferred[i][1:] and left_is_unary:
                el = _extrap_from(left_run, i, 0, "left")
                if el:
                    # Merge: fill unknown slots
                    merged = list(preferred[i])
                    el_chars = list(el)
                    for j in range(1, min(len(merged), len(el_chars))):
                        if merged[j] == "?" and el_chars[j] != "?":
                            merged[j] = el_chars[j]
                    preferred[i] = "".join(merged)

        lines.append("Preferred from left")
        for i in range(N_BITS):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")
    else:
        # Left is longer or equal: extrapolate from left first
        preferred = []
        for i in range(N_BITS):
            if i < left_len_final:
                preferred.append(left_run[i].expr)
            elif i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif left_is_binary or left_is_unary:
                preferred.append(
                    _extrap_from(left_run, i, 0, "left") or "pending"
                )
            else:
                preferred.append("pending")

        lines.append("Preferred from left")
        for i in range(N_BITS):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")

        # Fill remaining pending from right; merge unary digits
        for i in range(N_BITS):
            if preferred[i] == "pending":
                if right_is_binary or right_is_unary:
                    preferred[i] = _extrap_from(right_run, i, right_start_final, "right") or "?"
                else:
                    preferred[i] = "?"
            elif "?" in preferred[i][1:] and right_is_unary:
                er = _extrap_from(right_run, i, right_start_final, "right")
                if er:
                    # Merge: fill unknown slots
                    merged = list(preferred[i])
                    er_chars = list(er)
                    for j in range(1, min(len(merged), len(er_chars))):
                        if merged[j] == "?" and er_chars[j] != "?":
                            merged[j] = er_chars[j]
                    preferred[i] = "".join(merged)

        lines.append("Preferred from right")
        for i in range(N_BITS - 1, -1, -1):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")

    lines.append("Preferred")
    for i, pref in enumerate(preferred):
        if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
            lines.append(f"{i} {pref} ?{pref[2]}{pref[1]}")
        else:
            lines.append(f"{i} {pref}")
    lines.append("")

    # Build the final vector: left + middle selection + right
    default_cand = RuleCandidate(DEFAULT_FAMILY, None, None, "default 1")
    best: List[RuleCandidate] = [default_cand] * N_BITS

    # Place left and right runs
    for i, rc in enumerate(left_run):
        best[i] = rc
    for i, rc in enumerate(right_run):
        best[right_start_final + i] = rc

    # Fill middle (pending) slots via Matching + Perfect match logic
    lines.append("Matching")
    pending_indices: list[int] = []
    per_bit_cat: dict[str, dict[int, list[RuleCandidate]]] = {
        name: {} for name in SECTION_ORDER
    }

    for i in range(N_BITS):
        pref = preferred[i]
        if not pref.startswith("?") or pref == "?":
            lines.append(f"{i} {_rule_display(best[i])}")
            continue

        pending_indices.append(i)
        digits_str = pref[1:]
        pref_digits = [int(d) for d in digits_str if d != "?"]

        checks: list[str] = []
        for section_name in SECTION_ORDER:
            cands = all_matches[section_name][i]
            if section_name in ("Identity", "NOT"):
                found = [c for c in cands if c.primary in pref_digits]
                if found:
                    checks.append(
                        section_name + " " + " ".join(_rule_display(c) for c in found)
                    )
                    per_bit_cat[section_name][i] = found
                else:
                    checks.append(f"{section_name} absent")
            elif section_name == "Constant":
                if cands:
                    checks.append(
                        "Constant " + " ".join(_rule_display(c) for c in cands)
                    )
                    per_bit_cat["Constant"][i] = list(cands)
                else:
                    checks.append("Constant absent")
            else:
                found_c: Optional[RuleCandidate] = None
                # Try both orderings; prefer the first (as shown in Preferred)
                orderings = []
                want_p = int(pref[1]) if len(pref) > 1 and pref[1] != "?" else None
                want_s = int(pref[2]) if len(pref) > 2 and pref[2] != "?" else None
                orderings.append((want_p, want_s))
                if want_p is not None and want_s is not None and want_p != want_s:
                    orderings.append((want_s, want_p))
                for wp, ws in orderings:
                    for c in cands:
                        if (wp is None or c.primary == wp) and (ws is None or c.secondary == ws):
                            found_c = c
                            break
                    if found_c is not None:
                        break
                if found_c is not None:
                    checks.append(_rule_display(found_c))
                    per_bit_cat[section_name][i] = [found_c]
                else:
                    checks.append(f"{section_name} absent")
        if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
            pref_display = f"{pref} ?{pref[2]}{pref[1]}"
        else:
            pref_display = pref
        lines.append(f"{i} {pref_display} - {', '.join(checks)}")
    lines.append("")

    # Perfect match: first category that covers ALL pending bits wins
    lines.append("Perfect match")
    chosen_cat: Optional[str] = None
    for cat in SECTION_ORDER:
        is_perfect = (
            chosen_cat is None
            and bool(pending_indices)
            and all(i in per_bit_cat[cat] for i in pending_indices)
        )
        lines.append(f"{cat} {'yes' if is_perfect else 'no'}")
        if is_perfect:
            chosen_cat = cat
    lines.append("")

    # Matched: use perfect-match category to fill pending slots
    pending_set = set(pending_indices)
    lines.append("Matched")
    for i in range(N_BITS):
        if i in pending_set:
            if chosen_cat and i in per_bit_cat[chosen_cat]:
                best[i] = per_bit_cat[chosen_cat][i][0]
                lines.append(f"{i} {_rule_display(best[i])}")
            else:
                # No perfect match — list all candidates for this slot
                all_cands: list[RuleCandidate] = []
                for name in SECTION_ORDER:
                    if i in per_bit_cat[name]:
                        all_cands.extend(per_bit_cat[name][i])
                if all_cands:
                    lines.append(
                        f"{i} " + " ".join(_rule_display(c) for c in all_cands)
                    )
                    best[i] = all_cands[0]
                else:
                    lines.append(f"{i} none")
                    best[i] = default_cand
        else:
            lines.append(f"{i} {_rule_display(best[i])}")
    lines.append("")

    if enable_three_bit_repair and any(rule.is_default for rule in best):
        best = _repair_with_three_bit_search(inputs, output_columns, best)

    # Check if we have any non-default rules
    if all(r.is_default for r in best):
        return None

    if compact:
        return _render_reasoning_compact(
            question_bits=question_bits,
            output_columns=output_columns,
            n_examples=n_examples,
            all_matches=all_matches,
            left_winner_text=left_winner_text,
            right_winner_text=right_winner_text,
            best=best,
        )

    lines.append("Selected")
    for i, rule in enumerate(best):
        lines.append(f"{i} {_rule_display(rule)}")

    # 8) Apply to question.
    lines.append("")
    _emit_apply(lines, question_bits, best)

    return "\n".join(lines)
