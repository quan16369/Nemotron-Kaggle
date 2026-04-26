"""Equation numeric reasoning generator."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache

from reasoners.store_types import Problem

_EXPR_RE = re.compile(r"^(\d+)(\D)(\d+)$")


def _common_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Common operations tried first."""
    out: list[tuple[str, str]] = []
    out.append(("concatenation", sa + sb))
    out.append(("reverse concatenation", sb + sa))
    out.append(("addition", str(a + b)))
    out.append(("absolute difference", str(abs(a - b))))
    out.append(("negated absolute difference", str(-abs(a - b))))
    out.append(("subtraction (a-b)", str(a - b)))
    out.append(("reverse subtraction (b-a)", str(b - a)))
    out.append(("multiplication", str(a * b)))
    return out


def _rare_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Rare operations tried if common ones don't match."""
    out: list[tuple[str, str]] = []
    out.append(("multiply+1", str(a * b + 1)))
    out.append(("multiply-1", str(a * b - 1)))
    out.append(("add+1", str(a + b + 1)))
    out.append(("add-1", str(a + b - 1)))
    out.append(("sub+1", str(a - b + 1)))
    out.append(("sub-1", str(a - b - 1)))
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max mod min", str(big % small)))
    if b != 0:
        out.append(("integer division (a/b)", str(a // b)))
        out.append(("modulo (a mod b)", str(a % b)))
    if a != 0:
        out.append(("reverse division (b/a)", str(b // a)))
        out.append(("reverse modulo (b mod a)", str(b % a)))
    if len(sa) == 2 and len(sb) == 2:
        d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
        out.append(("digit absolute diff", str(abs(d1 - d3)) + str(abs(d2 - d4))))
        out.append(("digit add mod10", str((d1 + d3) % 10) + str((d2 + d4) % 10)))
        out.append(("digit sub mod10", str((d1 - d3) % 10) + str((d2 - d4) % 10)))
        out.append(("cross multiply", str(d1 * d3 + d2 * d4)))
        out.append(("cross multiply rev", str(d1 * d4 + d2 * d3)))
        out.append(("digit multiply", str(d1 * d3) + str(d2 * d4)))
        out.append(("digit multiply rev", str(d1 * d4) + str(d2 * d3)))
        out.append(("digit sum diff", str((d1 + d2) - (d3 + d4))))
        out.append(("digit sum sum", str((d1 + d2) + (d3 + d4))))
        out.append(("digit product diff", str(d1 * d2 - d3 * d4)))
        out.append(("digit product sum", str(d1 * d2 + d3 * d4)))
        det_val = d1 * d4 - d2 * d3
        out.append(("determinant", str(det_val)))
        out.append(("abs determinant", str(abs(det_val))))
    return out


def _all_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """All candidates: common first, then rare."""
    return _common_candidates(a, b, sa, sb) + _rare_candidates(a, b, sa, sb)


def _expr(name: str, a: str, b: str) -> str:
    """Return the math expression for an operation, e.g. '94 + 48'."""
    if name == "addition":
        return f"{a} + {b}"
    if name == "subtraction (a-b)":
        return f"{a} - {b}"
    if name == "reverse subtraction (b-a)":
        return f"{b} - {a}"
    if name == "multiplication":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b}"
        return f"{a} * {b}"
    if name == "absolute difference":
        return f"|{a} - {b}|"
    if name == "negated absolute difference":
        return f"-|{a} - {b}|"
    if name == "concatenation":
        return f"{a} || {b}"
    if name == "reverse concatenation":
        return f"{b} || {a}"
    if name == "multiply+1":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b} + 1"
        return f"{a} * {b} + 1"
    if name == "multiply-1":
        if len(a) >= 2:
            decomp = " + ".join(
                str(int(d) * (10 ** (len(a) - 1 - i))) for i, d in enumerate(a)
            )
            return f"({decomp}) * {b} - 1"
        return f"{a} * {b} - 1"
    if name == "add+1":
        return f"{a} + {b} + 1"
    if name == "add-1":
        return f"{a} + {b} - 1"
    if name == "sub+1":
        return f"{a} - {b} + 1"
    if name == "sub-1":
        return f"{a} - {b} - 1"
    if name == "integer division (a/b)":
        return f"{a} / {b}"
    if name == "modulo (a mod b)":
        return f"{a} mod {b}"
    if name == "reverse division (b/a)":
        return f"{b} / {a}"
    if name == "reverse modulo (b mod a)":
        return f"{b} mod {a}"
    if name == "max mod min":
        big, small = (a, b) if int(a) >= int(b) else (b, a)
        return f"max({a},{b}) mod min({a},{b}) = {big} mod {small}"
    if len(a) == 2 and len(b) == 2:
        d1, d2, d3, d4 = a[0], a[1], b[0], b[1]
        if name == "digit absolute diff":
            return f"|{d1}-{d3}| || |{d2}-{d4}|"
        if name == "digit add mod10":
            return f"({d1}+{d3})%10 || ({d2}+{d4})%10"
        if name == "digit sub mod10":
            return f"({d1}-{d3})%10 || ({d2}-{d4})%10"
        if name == "cross multiply":
            return f"{d1}*{d3} + {d2}*{d4}"
        if name == "cross multiply rev":
            return f"{d1}*{d4} + {d2}*{d3}"
        if name == "digit multiply":
            return f"{d1}*{d3} || {d2}*{d4}"
        if name == "digit multiply rev":
            return f"{d1}*{d4} || {d2}*{d3}"
        if name == "digit sum diff":
            return f"({d1}+{d2}) - ({d3}+{d4})"
        if name == "digit sum sum":
            return f"({d1}+{d2}) + ({d3}+{d4})"
        if name == "digit product diff":
            return f"{d1}*{d2} - {d3}*{d4}"
        if name == "digit product sum":
            return f"{d1}*{d2} + {d3}*{d4}"
        if name == "determinant":
            return f"{d1}*{d4} - {d2}*{d3}"
        if name == "abs determinant":
            return f"|{d1}*{d4} - {d2}*{d3}|"
    return ""


def _expr_intermediate(name: str, a: str, b: str) -> str:
    """Return intermediate evaluated form for operations with multiplications, else ''."""
    ia, ib = int(a), int(b)
    if name in ("multiply+1", "multiply-1", "multiplication") and len(a) >= 2:
        # Decompose a by place value: 70 → [70, 0], 73 → [70, 3]
        places = [int(d) * (10 ** (len(a) - 1 - i)) for i, d in enumerate(a)]
        decomp = " + ".join(f"{p} * {b}" for p in places)
        evald = " + ".join(str(p * ib) for p in places)
        product_sum = sum(p * ib for p in places)
        if name == "multiply+1":
            return f"{decomp} + 1 = {evald} + 1 = {product_sum} + 1"
        if name == "multiply-1":
            return f"{decomp} - 1 = {evald} - 1 = {product_sum} - 1"
        return f"{decomp} = {evald}"
    if len(a) == 2 and len(b) == 2:
        d1, d2, d3, d4 = int(a[0]), int(a[1]), int(b[0]), int(b[1])
        if name == "cross multiply":
            return f"{d1 * d3} + {d2 * d4}"
        if name == "cross multiply rev":
            return f"{d1 * d4} + {d2 * d3}"
        if name == "digit multiply":
            return f"{d1 * d3} || {d2 * d4}"
        if name == "digit multiply rev":
            return f"{d1 * d4} || {d2 * d3}"
        if name == "digit product diff":
            return f"{d1 * d2} - {d3 * d4}"
        if name == "digit product sum":
            return f"{d1 * d2} + {d3 * d4}"
        if name == "determinant":
            return f"{d1 * d4} - {d2 * d3}"
        if name == "abs determinant":
            return f"|{d1 * d4} - {d2 * d3}|"
    return ""


def _rev(s: str) -> str:
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


@dataclass
class FoundOp:
    op_name: str
    rev_ops: bool
    rev_res: bool
    fmt: str
    op_char: str


def _rule_key(found: FoundOp) -> tuple[str, bool, bool, str, str]:
    return (
        found.op_name,
        found.rev_ops,
        found.rev_res,
        found.fmt,
        found.op_char,
    )


def _rule_prior_key(found: FoundOp) -> tuple[str, bool, bool, str]:
    return (
        found.op_name,
        found.rev_ops,
        found.rev_res,
        found.fmt,
    )


_SAFE_SPECIAL_GUESS_RULES: dict[
    tuple[str, tuple[tuple[str, str, bool, bool, str], ...]],
    tuple[str, bool, bool, str],
] = {
    (
        "*",
        (
            ("+", "add-1", True, True, "num"),
            ("-", "absolute difference", True, True, "num"),
        ),
    ): ("multiplication", True, True, "num"),
    (
        ":",
        (
            ("<", "multiply+1", False, False, "num"),
            ("@", "reverse concatenation", True, True, "num"),
        ),
    ): ("absolute difference", False, False, "pre"),
    (
        "}",
        (
            ("+", "reverse concatenation", True, True, "num"),
            ("{", "multiply+1", False, False, "num"),
        ),
    ): ("absolute difference", False, False, "pre"),
}

_SAFE_SPECIAL_DEDUCE_RULES: dict[
    tuple[str, tuple[tuple[str, str, bool, bool, str], ...]],
    tuple[tuple[str, bool, bool, str], ...],
] = {
    (
        "-",
        (
            ("*", "multiply-1", True, True, "num"),
        ),
    ): (
        ("subtraction (a-b)", True, True, "num"),
    ),
    (
        "-",
        (
            ("+", "add+1", True, True, "num"),
        ),
    ): (
        ("subtraction (a-b)", True, True, "neg_prefix"),
        ("subtraction (a-b)", True, True, "num"),
    ),
}


def _transform_group(
    op_char: str, group: list[tuple[str, str, str]]
) -> tuple[str, list[tuple[str, str, str]]]:
    """Normalize any operator-prefixed/suffixed outputs before rule matching."""
    any_neg_suffixed = any(out.endswith("-") and len(out) > 1 for _, _, out in group)
    any_neg_prefixed = any(out.startswith("-") and len(out) > 1 for _, _, out in group)
    any_suffixed = any(out.endswith(op_char) and len(out) > 1 for _, _, out in group)
    any_prefixed = any(out.startswith(op_char) and len(out) > 1 for _, _, out in group)
    all_suffixed = all(out.endswith(op_char) and len(out) > 1 for _, _, out in group)
    all_prefixed = all(out.startswith(op_char) and len(out) > 1 for _, _, out in group)

    fmt = "num"
    transformed = list(group)
    if op_char != "-" and all_prefixed:
        fmt = "pre"
        transformed = [(a, b, out[len(op_char) :]) for a, b, out in group]
    elif op_char != "-" and all_suffixed:
        fmt = "suf"
        transformed = [(a, b, out[: -len(op_char)]) for a, b, out in group]
    elif any_neg_suffixed:
        fmt = "neg_suffix"
        transformed = [
            (a, b, "-" + out[:-1] if out.endswith("-") and len(out) > 1 else out)
            for a, b, out in group
        ]
    elif any_neg_prefixed:
        fmt = "neg_prefix"
    elif any_suffixed:
        fmt = "neg_suffix"
        transformed = [
            (
                a,
                b,
                "-" + out[: -len(op_char)] if out.endswith(op_char) and len(out) > 1 else out,
            )
            for a, b, out in group
        ]
    elif any_prefixed:
        fmt = "neg_prefix"
        transformed = [
            (
                a,
                b,
                "-" + out[len(op_char) :] if out.startswith(op_char) and len(out) > 1 else out,
            )
            for a, b, out in group
        ]
    return fmt, transformed


def _apply_format_to_group(
    op_char: str,
    group: list[tuple[str, str, str]],
    fmt: str,
) -> list[tuple[str, str, str]]:
    if fmt == "pre":
        return [
            (a, b, out[len(op_char) :] if out.startswith(op_char) and len(out) > 1 else out)
            for a, b, out in group
        ]
    if fmt == "suf":
        return [
            (a, b, out[: -len(op_char)] if out.endswith(op_char) and len(out) > 1 else out)
            for a, b, out in group
        ]
    if fmt == "neg_prefix":
        return [
            (
                a,
                b,
                "-" + out[len(op_char) :]
                if op_char != "-" and out.startswith(op_char) and len(out) > 1
                else out,
            )
            for a, b, out in group
        ]
    if fmt == "neg_suffix":
        return [
            (
                a,
                b,
                "-" + out[: -len(op_char)]
                if op_char != "-" and out.endswith(op_char) and len(out) > 1
                else out,
            )
            for a, b, out in group
        ]
    return list(group)


def _candidate_format_variants(
    op_char: str,
    group: list[tuple[str, str, str]],
) -> list[tuple[str, list[tuple[str, str, str]]]]:
    base_fmt, base_group = _transform_group(op_char, group)
    variants: list[tuple[str, list[tuple[str, str, str]]]] = [(base_fmt, base_group)]
    if op_char != "-":
        if base_fmt == "num":
            variants.append(("neg_prefix", _apply_format_to_group(op_char, group, "neg_prefix")))
            variants.append(("neg_suffix", _apply_format_to_group(op_char, group, "neg_suffix")))
        elif base_fmt == "pre":
            variants.append(("neg_prefix", _apply_format_to_group(op_char, group, "neg_prefix")))
        elif base_fmt == "suf":
            variants.append(("neg_suffix", _apply_format_to_group(op_char, group, "neg_suffix")))

    unique: list[tuple[str, list[tuple[str, str, str]]]] = []
    seen: set[tuple[str, tuple[tuple[str, str, str], ...]]] = set()
    for fmt, transformed in variants:
        key = (fmt, tuple(transformed))
        if key in seen:
            continue
        seen.add(key)
        unique.append((fmt, transformed))
    return unique


def _find_matching_rules(
    op_char: str,
    group: list[tuple[str, str, str]],
    detected_fmt: str,
) -> list[FoundOp]:
    matches: list[FoundOp] = []
    seen: set[tuple[str, bool, bool, str, str]] = set()
    candidate_sets = (
        _common_candidates,
        _rare_candidates,
    )
    for cand_fn in candidate_sets:
        for rev_ops, rev_res in (
            (True, True),
            (False, False),
            (True, False),
            (False, True),
        ):
            a_str, b_str, _ = group[0]
            ta = a_str[::-1] if rev_ops else a_str
            tb = b_str[::-1] if rev_ops else b_str
            for cand_name, _ in cand_fn(int(ta), int(tb), ta, tb):
                all_pass = True
                for ax, bx, expected in group:
                    rax = ax[::-1] if rev_ops else ax
                    rbx = bx[::-1] if rev_ops else bx
                    raw = next(
                        r
                        for n, r in _all_candidates(int(rax), int(rbx), rax, rbx)
                        if n == cand_name
                    )
                    fin = _rev(raw) if rev_res else raw
                    if fin != expected:
                        all_pass = False
                        break
                if not all_pass:
                    continue
                found = FoundOp(
                    op_name=cand_name,
                    rev_ops=rev_ops,
                    rev_res=rev_res,
                    fmt=detected_fmt,
                    op_char=op_char,
                )
                key = _rule_key(found)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(found)
    return matches


def _find_matching_rule(
    op_char: str,
    group: list[tuple[str, str, str]],
    detected_fmt: str,
) -> FoundOp | None:
    """Return the first matching rule in the same search order as the reasoning trace."""
    matches = _find_matching_rules(op_char, group, detected_fmt)
    return matches[0] if matches else None


def _describe_found_op(found: FoundOp) -> str:
    parts: list[str] = []
    if found.rev_ops:
        parts.append("reversed operands")
    if found.rev_res:
        parts.append("reversed result")
    parts.append(found.op_name)
    if found.fmt == "pre":
        parts.append("operator prefix format")
    elif found.fmt == "suf":
        parts.append("operator suffix format")
    elif found.fmt == "neg_suffix":
        parts.append("negative results use operator suffix")
    elif found.fmt == "neg_prefix":
        parts.append("negative results use operator prefix")
    return ", ".join(parts)


def _rich_example_semantics(
    example_rules: dict[str, FoundOp],
) -> tuple[tuple[str, str, bool, bool, str], ...]:
    return tuple(
        sorted(
            (
                op_char,
                found.op_name,
                found.rev_ops,
                found.rev_res,
                found.fmt,
            )
            for op_char, found in example_rules.items()
        )
    )


@cache
def _load_guess_priors() -> tuple[
    dict[tuple[str, tuple[tuple[str, bool, bool], ...]], Counter[tuple[str, bool, bool, str]]],
    dict[tuple[tuple[str, bool, bool], ...], Counter[tuple[str, bool, bool, str]]],
    dict[str, Counter[tuple[str, bool, bool, str]]],
]:
    """Learn priors for hidden operators from solved equation_numeric_deduce problems."""
    qop_context_priors: defaultdict[
        tuple[str, tuple[tuple[str, bool, bool], ...]],
        Counter[tuple[str, bool, bool, str]],
    ] = defaultdict(Counter)
    context_priors: defaultdict[
        tuple[tuple[str, bool, bool], ...],
        Counter[tuple[str, bool, bool, str]],
    ] = defaultdict(Counter)
    qop_priors: defaultdict[str, Counter[tuple[str, bool, bool, str]]] = defaultdict(
        Counter
    )

    for problem in Problem.load_all():
        if problem.category != "equation_numeric_deduce":
            continue

        parsed: list[tuple[str, str, str, str]] = []
        for ex in problem.examples:
            match = _EXPR_RE.fullmatch(str(ex.input_value))
            if not match:
                continue
            parsed.append(
                (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                    str(ex.output_value),
                )
            )
        if not parsed:
            continue

        by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for a, op, b, out in parsed:
            by_op[op].append((a, b, out))

        detected_fmts: dict[str, str] = {}
        transformed_groups: dict[str, list[tuple[str, str, str]]] = {}
        for op_char, group in by_op.items():
            fmt, transformed = _transform_group(op_char, group)
            detected_fmts[op_char] = fmt
            transformed_groups[op_char] = transformed

        found_ops: dict[str, FoundOp] = {}
        for op_char, group in transformed_groups.items():
            found = _find_matching_rule(op_char, group, detected_fmts[op_char])
            if found is not None:
                found_ops[op_char] = found

        q_match = _EXPR_RE.fullmatch(str(problem.question))
        if not q_match:
            continue
        q_op = q_match.group(2)
        q_found = found_ops.get(q_op)
        if q_found is None:
            continue

        other_semantics = tuple(
            sorted(
                (found.op_name, found.rev_ops, found.rev_res)
                for op, found in found_ops.items()
                if op != q_op
            )
        )
        rule_key = (
            q_found.op_name,
            q_found.rev_ops,
            q_found.rev_res,
            q_found.fmt,
        )
        qop_context_priors[(q_op, other_semantics)][rule_key] += 1
        context_priors[other_semantics][rule_key] += 1
        qop_priors[q_op][rule_key] += 1

    return dict(qop_context_priors), dict(context_priors), dict(qop_priors)


def _guess_missing_question_rule(
    q_op: str,
    example_rules: dict[str, FoundOp],
) -> tuple[FoundOp, str] | None:
    """Predict a hidden question operator using priors over solved deduce problems."""
    qop_context_priors, context_priors, qop_priors = _load_guess_priors()
    rich_semantics = _rich_example_semantics(example_rules)
    safe_rule = _SAFE_SPECIAL_GUESS_RULES.get((q_op, rich_semantics))
    if safe_rule is not None:
        op_name, rev_ops, rev_res, fmt = safe_rule
        return (
            FoundOp(op_name, rev_ops, rev_res, fmt, q_op),
            "a train-fitted heuristic for this hidden-operator semantic context",
        )
    example_semantics = tuple(
        sorted((found.op_name, found.rev_ops, found.rev_res) for found in example_rules.values())
    )

    choices: list[tuple[Counter[tuple[str, bool, bool, str]] | None, str]] = [
        (
            qop_context_priors.get((q_op, example_semantics)),
            "priors conditioned on the question operator and the example semantics",
        ),
        (
            context_priors.get(example_semantics),
            "priors conditioned on the example semantics",
        ),
        (
            qop_priors.get(q_op),
            "the global prior for this operator",
        ),
    ]
    for counter, source in choices:
        if not counter:
            continue
        (op_name, rev_ops, rev_res, fmt), _ = counter.most_common(1)[0]
        return FoundOp(op_name, rev_ops, rev_res, fmt, q_op), source
    return None


def _apply_op(found: FoundOp, a_str: str, b_str: str) -> tuple[str, list[str]]:
    """Apply the found operation and return (result, explanation_lines)."""
    steps: list[str] = []
    ta = a_str[::-1] if found.rev_ops else a_str
    tb = b_str[::-1] if found.rev_ops else b_str

    # Header line always present
    if found.rev_ops and found.rev_res:
        steps.append(
            f"reversed operands [{a_str}->{ta}, {b_str}->{tb}] and reversed result"
        )
    elif found.rev_ops:
        steps.append(f"reversed operands [{a_str}->{ta}, {b_str}->{tb}]")
    elif found.rev_res:
        steps.append("reversed result")
    else:
        steps.append("identity")

    # Find the matching candidate
    raw_result = ""
    for name, res in _all_candidates(int(ta), int(tb), ta, tb):
        if name == found.op_name:
            raw_result = res
            break

    final = _rev(raw_result) if found.rev_res else raw_result

    expr = _expr(found.op_name, ta, tb)
    inter = _expr_intermediate(found.op_name, ta, tb)
    if expr and inter:
        detail = f" {expr} = {inter} ="
    elif expr:
        detail = f" {expr} ="
    else:
        detail = ""
    val = f"{raw_result} -rev-> {final}" if found.rev_res else final
    steps.append(f"{found.op_name} f({ta}, {tb}) ={detail} {val}")

    if found.fmt == "pre":
        final = found.op_char + final
        steps.append(f"Prefix operator: {final}")
    elif found.fmt == "suf":
        final = final + found.op_char
        steps.append(f"Suffix operator: {final}")
    elif found.fmt == "neg_suffix":
        if final.startswith("-"):
            old = final
            final = final[1:] + found.op_char
            steps.append(
                f"Result is negative - we add back the operator suffix 【{found.op_char}】: {old} -> 【{final}】"
            )
        else:
            steps.append(f"Result is non-negative, no suffix needed: 【{final}】")
    elif found.fmt == "neg_prefix":
        if final.startswith("-"):
            old = final
            final = found.op_char + final[1:]
            steps.append(
                f"Result is negative - we add back the operator prefix 【{found.op_char}】: {old} -> 【{final}】"
            )
        else:
            steps.append(f"Result is non-negative, no prefix needed: 【{final}】")

    return final, steps


@cache
def _load_rule_selection_priors() -> tuple[
    dict[tuple[str, tuple[tuple[str, str, bool, bool, str], ...]], Counter[tuple[str, bool, bool, str]]],
    dict[str, Counter[tuple[str, bool, bool, str]]],
]:
    """Learn priors for ambiguous in-example operators from solved deduce problems."""
    qop_context_priors: defaultdict[
        tuple[str, tuple[tuple[str, str, bool, bool, str], ...]],
        Counter[tuple[str, bool, bool, str]],
    ] = defaultdict(Counter)
    qop_priors: defaultdict[str, Counter[tuple[str, bool, bool, str]]] = defaultdict(
        Counter
    )

    for problem in Problem.load_all():
        if problem.category != "equation_numeric_deduce":
            continue

        parsed: list[tuple[str, str, str, str]] = []
        for ex in problem.examples:
            match = _EXPR_RE.fullmatch(str(ex.input_value))
            if not match:
                continue
            parsed.append(
                (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                    str(ex.output_value),
                )
            )
        if not parsed:
            continue

        by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for a, op, b, out in parsed:
            by_op[op].append((a, b, out))

        q_match = _EXPR_RE.fullmatch(str(problem.question))
        if not q_match:
            continue
        qa, q_op, qb = q_match.group(1), q_match.group(2), q_match.group(3)
        if q_op not in by_op:
            continue

        example_rules: dict[str, FoundOp] = {}
        for op_char, group in by_op.items():
            if op_char == q_op:
                continue
            fmt, transformed = _transform_group(op_char, group)
            found = _find_matching_rule(op_char, transformed, fmt)
            if found is not None:
                example_rules[op_char] = found
        context = tuple(
            sorted(
                (
                    op,
                    found.op_name,
                    found.rev_ops,
                    found.rev_res,
                    found.fmt,
                )
                for op, found in example_rules.items()
            )
        )

        candidates: list[FoundOp] = []
        for fmt, transformed in _candidate_format_variants(q_op, by_op[q_op]):
            candidates.extend(_find_matching_rules(q_op, transformed, fmt))

        unique_candidates: list[FoundOp] = []
        seen: set[tuple[str, bool, bool, str, str]] = set()
        for found in candidates:
            key = _rule_key(found)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(found)

        for found in unique_candidates:
            predicted, _ = _apply_op(found, qa, qb)
            if predicted != problem.answer:
                continue
            prior_key = _rule_prior_key(found)
            qop_context_priors[(q_op, context)][prior_key] += 1
            qop_priors[q_op][prior_key] += 1

    return dict(qop_context_priors), dict(qop_priors)


def _choose_best_matching_rule(
    op_char: str,
    group: list[tuple[str, str, str]],
    qa: str,
    qb: str,
    example_rules: dict[str, FoundOp],
) -> FoundOp | None:
    candidates: list[FoundOp] = []
    for fmt, transformed in _candidate_format_variants(op_char, group):
        candidates.extend(_find_matching_rules(op_char, transformed, fmt))

    unique_candidates: list[FoundOp] = []
    seen: set[tuple[str, bool, bool, str, str]] = set()
    for found in candidates:
        key = _rule_key(found)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(found)

    if not unique_candidates:
        return None
    if len(unique_candidates) == 1:
        return unique_candidates[0]

    qop_context_priors, qop_priors = _load_rule_selection_priors()
    context = tuple(
        sorted(
            (
                op,
                found.op_name,
                found.rev_ops,
                found.rev_res,
                found.fmt,
            )
            for op, found in example_rules.items()
        )
    )

    special_preferences = _SAFE_SPECIAL_DEDUCE_RULES.get((op_char, context))
    if special_preferences is not None:
        for op_name, rev_ops, rev_res, fmt in special_preferences:
            for found in unique_candidates:
                if (
                    found.op_name == op_name
                    and found.rev_ops == rev_ops
                    and found.rev_res == rev_res
                    and found.fmt == fmt
                ):
                    return found

    context_counter = qop_context_priors.get((op_char, context), Counter())
    op_counter = qop_priors.get(op_char, Counter())

    def _score(found: FoundOp) -> tuple[int, int, int]:
        prior_key = _rule_prior_key(found)
        subtraction_bonus = 1 if (
            op_char == "-"
            and found.op_name in {"subtraction (a-b)", "reverse subtraction (b-a)"}
        ) else 0
        return (
            context_counter.get(prior_key, 0),
            op_counter.get(prior_key, 0),
            subtraction_bonus,
        )

    return max(unique_candidates, key=_score)


def reasoning_equation_numeric(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append("Examples:")

    parsed: list[tuple[str, str, str, str]] = []
    for ex in problem.examples:
        m = _EXPR_RE.fullmatch(str(ex.input_value))
        if not m:
            continue
        a, op, b = m.group(1), m.group(2), m.group(3)
        parsed.append((a, op, b, str(ex.output_value)))
        lines.append(f"  {ex.input_value} = {ex.output_value}")

    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))

    q_match = _EXPR_RE.fullmatch(str(problem.question))
    q_op = q_match.group(2) if q_match else None

    # Precompute the baseline transform per operator.
    base_detected_fmts: dict[str, str] = {}
    base_transformed_groups: dict[str, list[tuple[str, str, str]]] = {}
    for op_char, group in by_op.items():
        fmt, transformed = _transform_group(op_char, group)
        base_detected_fmts[op_char] = fmt
        base_transformed_groups[op_char] = transformed

    # Resolve the question operator with train-fitted priors before rendering the trace.
    context_example_rules: dict[str, FoundOp] = {}
    if q_op is not None:
        for op_char, group in base_transformed_groups.items():
            if op_char == q_op:
                continue
            found = _find_matching_rule(op_char, group, base_detected_fmts[op_char])
            if found is not None:
                context_example_rules[op_char] = found

    preferred_question_rule: FoundOp | None = None
    if q_match is not None and q_op in by_op:
        preferred_question_rule = _choose_best_matching_rule(
            q_op,
            by_op[q_op],
            q_match.group(1),
            q_match.group(3),
            context_example_rules,
        )

    # Apply the preferred format choice for downstream displays and matching.
    detected_fmts: dict[str, str] = {}
    transformed_groups: dict[str, list[tuple[str, str, str]]] = {}
    has_symbol_suffix = False
    has_symbol_prefix = False
    symbol_suffix_char = ""
    symbol_prefix_char = ""

    for op_char, group in by_op.items():
        fmt = base_detected_fmts[op_char]
        transformed = base_transformed_groups[op_char]
        if preferred_question_rule is not None and op_char == q_op:
            fmt = preferred_question_rule.fmt
            transformed = _apply_format_to_group(op_char, group, fmt)

        any_suffixed = any(out.endswith(op_char) and len(out) > 1 for _, _, out in group)
        any_prefixed = any(out.startswith(op_char) and len(out) > 1 for _, _, out in group)
        if fmt == "pre" and any_prefixed:
            has_symbol_prefix = True
            symbol_prefix_char = op_char
        elif fmt == "suf" and any_suffixed:
            has_symbol_suffix = True
            symbol_suffix_char = op_char
        elif fmt == "neg_suffix" and any_suffixed:
            has_symbol_suffix = True
            symbol_suffix_char = op_char
        elif fmt == "neg_prefix" and any_prefixed:
            has_symbol_prefix = True
            symbol_prefix_char = op_char

        detected_fmts[op_char] = fmt
        transformed_groups[op_char] = transformed

    # Build map from (a, op, b) to transformed output
    transformed_map: dict[tuple[str, str, str], str] = {}
    for oc, tgroup in transformed_groups.items():
        for a, b, tout in tgroup:
            transformed_map[(a, oc, b)] = tout

    # Check inputs for leading zeros
    all_inputs: list[str] = []
    for a, _, b, _ in parsed:
        all_inputs.append(a)
        all_inputs.append(b)
    lines.append("")
    lines.append(f"The inputs are {', '.join(all_inputs)}")

    # Report outputs
    all_outputs = [out for _, _, _, out in parsed]
    lines.append("")
    lines.append(f"The outputs are {', '.join(all_outputs)}")
    if has_symbol_suffix:
        lines.append(
            f"Some outputs have the operator symbol as suffix 【{symbol_suffix_char}】."
        )
    if has_symbol_prefix:
        lines.append(
            f"Some outputs have the operator symbol as prefix 【{symbol_prefix_char}】."
        )
    if not has_symbol_suffix and not has_symbol_prefix:
        lines.append("No outputs have a symbol prefix or suffix.")

    # Show transformed outputs if any transformation occurred
    any_transformed = any(fmt != "num" for fmt in detected_fmts.values())
    if any_transformed:
        t_all = [transformed_map.get((a, op, b), out) for a, op, b, out in parsed]
        lines.append(f"We now consider the outputs to be {', '.join(t_all)}")
        q_fmt = detected_fmts.get(q_op or "", "num")
        if q_fmt == "suf":
            lines.append("We will add back the operator suffix to the final answer.")
        elif q_fmt == "pre":
            lines.append("We will add back the operator prefix to the final answer.")
        elif q_fmt == "neg_suffix":
            lines.append(
                "We will add back the operator suffix if our answer is negative."
            )
        elif q_fmt == "neg_prefix":
            lines.append(
                "We will add back the operator prefix if our answer is negative."
            )

    lines.append("")

    # Show input → operator parsing
    lines.append("Looking at the input of the examples")
    for a, op, b, out in parsed:
        lines.append(f"{a}{op}{b} -> {op}")
    op_names = list(by_op.keys())
    lines.append("")
    lines.append("The operators")
    for op in op_names:
        lines.append(op)

    lines.append("")
    lines.append("Looking at the question")
    if q_match:
        lines.append(f"{problem.question} -> {q_op}")

    example_rules: dict[str, FoundOp] = dict(context_example_rules)
    if preferred_question_rule is not None and q_op is not None:
        example_rules[q_op] = preferred_question_rule
    for op_char, group in transformed_groups.items():
        if op_char in example_rules:
            continue
        found = _find_matching_rule(op_char, group, detected_fmts[op_char])
        if found is not None:
            example_rules[op_char] = found

    guessed_question_rule: FoundOp | None = None
    guessed_question_source = ""

    # If question operator not in examples, analyze one example operator and predict the hidden rule
    effective_q_op = q_op
    if q_op is not None and q_op not in by_op and by_op:
        most_common_op = max(by_op, key=lambda op: len(by_op[op]))
        lines.append("The question operator is not found in the examples.")
        lines.append(
            f"Investigating the most common example operator 【{most_common_op}】 instead."
        )
        guessed = _guess_missing_question_rule(q_op, example_rules)
        if guessed is None:
            lines.append(
                "We do not have a matching prior for the hidden operator. "
                "We will use absolute difference as a last-resort fallback."
            )
        else:
            guessed_question_rule, guessed_question_source = guessed
            lines.append(
                f"We will predict the hidden question rule using {guessed_question_source}."
            )
            lines.append(
                f"Predicted question rule: {_describe_found_op(guessed_question_rule)}."
            )
        effective_q_op = most_common_op
    elif q_op is not None and q_op in by_op:
        lines.append("The question operator is found in the examples.")

    found_ops: dict[str, FoundOp] = {}

    # Analyze each operator (focus on question operator)
    for op_char, group in sorted(by_op.items()):
        if effective_q_op is not None and op_char != effective_q_op and len(by_op) > 1:
            continue

        # Use precomputed format and transformed group
        detected_fmt = detected_fmts[op_char]
        group = transformed_groups[op_char]

        examples_str = ", ".join(f"{a}{op_char}{b} = {out}" for a, b, out in group)
        lines.append("")
        lines.append(f"Looking at operator 【{op_char}】 [{examples_str}]:")

        a_str, b_str, expected = group[0]

        # Try common operations first (all 4 combos), then rare operations
        found = None
        preferred_found = example_rules.get(op_char)
        preferred_key = _rule_key(preferred_found) if preferred_found is not None else None

        candidate_sets = [
            ("common", _common_candidates),
            ("rare", _rare_candidates),
        ]

        n_ex = len(group)
        for set_name, cand_fn in candidate_sets:
            for rev_ops, rev_res in (
                (True, True),
                (False, False),
                (True, False),
                (False, True),
            ):
                # Use fixed example order for paragraph header
                cycled = list(group)

                # Describe what we're trying
                label = f"{set_name} operations"
                if rev_ops:
                    rev_parts = ", ".join(
                        f"{ax}->{ax[::-1]} {bx}->{bx[::-1]}" for ax, bx, _ in cycled
                    )
                    if rev_res:
                        label += f" reversed operands [{rev_parts}] and reversed result"
                    else:
                        label += f" reversed operands [{rev_parts}]"
                elif rev_res:
                    id_parts = ", ".join(f"{ax} {bx}" for ax, bx, _ in cycled)
                    label += f" identity operands [{id_parts}] reversed result"
                else:
                    id_parts = ", ".join(f"{ax} {bx}" for ax, bx, _ in cycled)
                    label += f" on identity [{id_parts}]"
                if rev_ops:
                    all_expected = ", ".join(
                        f"({ax[::-1]},{bx[::-1]})->{exp}" for ax, bx, exp in cycled
                    )
                else:
                    all_expected = ", ".join(
                        f"({ax},{bx})->{exp}" for ax, bx, exp in cycled
                    )
                lines.append(f"  Trying {label} [expected {all_expected}]:")

                def _fmt_result(
                    raw: str, a: str, b: str, detail: str, arrow: bool
                ) -> str:
                    fin = _rev(raw) if rev_res else raw
                    val = f"{raw} -rev-> {fin}" if rev_res else fin
                    if arrow:
                        return f"f({a},{b}) ->{detail} {val}"
                    return f"f({a}, {b}) ={detail} {val}"

                # Use first example for candidate generation
                ca_str, cb_str = cycled[0][0], cycled[0][1]
                cta = ca_str[::-1] if rev_ops else ca_str
                ctb = cb_str[::-1] if rev_ops else cb_str
                candidates = cand_fn(int(cta), int(ctb), cta, ctb)
                cand_idx = 0
                for cand_name, cand_res in candidates:
                    # Rotate which example is tried first within the paragraph
                    rotated = [cycled[(cand_idx + j) % n_ex] for j in range(n_ex)]
                    cand_idx += 1

                    parts = []
                    all_pass = True

                    for i, (ax, bx, exp_x) in enumerate(rotated):
                        rax = ax[::-1] if rev_ops else ax
                        rbx = bx[::-1] if rev_ops else bx
                        raw = next(
                            r
                            for n, r in _all_candidates(int(rax), int(rbx), rax, rbx)
                            if n == cand_name
                        )
                        expr_x = _expr(cand_name, rax, rbx)
                        inter_x = _expr_intermediate(cand_name, rax, rbx)
                        if expr_x and inter_x:
                            detail_x = f" {expr_x} = {inter_x} ="
                        elif expr_x:
                            detail_x = f" {expr_x} ="
                        else:
                            detail_x = ""
                        fin = _rev(raw) if rev_res else raw
                        status = "match" if fin == exp_x else "wrong"
                        if fin != exp_x:
                            all_pass = False
                        parts.append(
                            _fmt_result(raw, rax, rbx, detail_x, arrow=i > 0)
                            + f" {status}"
                        )
                        if fin != exp_x:
                            break

                    if all_pass:
                        current_found = FoundOp(
                            op_name=cand_name,
                            rev_ops=rev_ops,
                            rev_res=rev_res,
                            fmt=detected_fmt,
                            op_char=op_char,
                        )
                        current_key = _rule_key(current_found)
                        if preferred_key is not None and current_key == preferred_key:
                            summary = []
                            if rev_ops:
                                summary.append("reversed operands")
                            if rev_res:
                                summary.append("reversed result")
                            summary.append(cand_name)
                            parts.append("correct, actions: " + ", ".join(summary))
                            found = current_found
                        elif found:
                            parts.append("correct, but skipping")
                        elif preferred_key is None:
                            summary = []
                            if rev_ops:
                                summary.append("reversed operands")
                            if rev_res:
                                summary.append("reversed result")
                            summary.append(cand_name)
                            parts.append("correct, actions: " + ", ".join(summary))
                            found = current_found
                        else:
                            parts.append("correct, but skipping")
                    lines.append(f"    {cand_name} " + ", ".join(parts))

                    if not all_pass:
                        continue

        if found is None and preferred_found is not None:
            found = preferred_found

        if found:
            found_ops[op_char] = found
        else:
            if op_char == effective_q_op:
                return None
            lines.append("  No matching operation found.")

    # Apply to question
    if not q_match:
        return None

    qa, qb = q_match.group(1), q_match.group(3)
    lines.append("")
    lines.append(f"Applying to {problem.question}:")
    if effective_q_op != q_op:
        if guessed_question_rule is None:
            lines.append(
                "  We recall that the question operator is not found in the examples. "
                "No prior matched, so we will use absolute difference."
            )
            fallback_op = FoundOp(
                op_name="absolute difference",
                rev_ops=False,
                rev_res=False,
                fmt="num",
                op_char=q_op or "",
            )
            result_val, steps = _apply_op(fallback_op, qa, qb)
        else:
            lines.append(
                "  We recall that the question operator is not found in the examples. "
                f"We will use {guessed_question_source}."
            )
            lines.append(
                f"  Predicted actions: {_describe_found_op(guessed_question_rule)}."
            )
            result_val, steps = _apply_op(guessed_question_rule, qa, qb)
    else:
        if effective_q_op not in found_ops:
            return None
        result_val, steps = _apply_op(found_ops[effective_q_op], qa, qb)
    if problem.category == "equation_numeric_guess" and result_val.startswith("+"):
        result_val = result_val[1:]
    for step in steps:
        lines.append(f"  {step}")
    lines.append(f"  Result: 【{result_val}】")

    lines.append("")
    if "}" in result_val:
        lines[1] = (
            "I will return the final answer plainly because it contains the symbol }."
        )
        lines.append(
            "I will return the final answer plainly because it contains the symbol }."
        )
        lines.append(f"Final answer is: {result_val}")
    else:
        lines.append("I will now return the answer in \\boxed{}")
        lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{result_val}}}")
    return "\n".join(lines)
