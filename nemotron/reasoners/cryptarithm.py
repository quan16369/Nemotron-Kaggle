"""Equation symbolic reasoning generator.

Currently handles concatenation operators only (forward and reverse).
Operates directly on the original symbols without letter assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from reasoners.store_types import Problem


@dataclass
class _Ex:
    a: tuple[str, str]
    op: str
    b: tuple[str, str]
    out: str


def _concat_type(exs: list[_Ex]) -> str | None:
    """Return 'fwd' if A1A2B1B2, 'rev' if B1B2A1A2, else None."""
    if all(ex.out == ex.a[0] + ex.a[1] + ex.b[0] + ex.b[1] for ex in exs):
        return "fwd"
    if all(ex.out == ex.b[0] + ex.b[1] + ex.a[0] + ex.a[1] for ex in exs):
        return "rev"
    return None


def _box(s: str) -> str:
    """Wrap each character in 【】 brackets."""
    return "".join(f"【{c}】" for c in s)


def _slot_list(items: list[tuple[str, str]]) -> str:
    return ", ".join(f"{name}=【{value}】" for name, value in items)


def reasoning_cryptarithm(problem: Problem) -> str | None:
    """Generate reasoning for cryptarithm problems."""

    def quote(s: str) -> str:
        return f"【{s}】"

    exs: list[_Ex] = []
    for ex in problem.examples:
        inp = str(ex.input_value)
        if len(inp) != 5:
            return None
        exs.append(
            _Ex(
                a=(inp[0], inp[1]),
                op=inp[2],
                b=(inp[3], inp[4]),
                out=str(ex.output_value),
            )
        )

    q = str(problem.question)
    if len(q) != 5:
        return None
    q_a = (q[0], q[1])
    q_op = q[2]
    q_b = (q[3], q[4])

    # Group by operator
    by_op: dict[str, list[_Ex]] = {}
    for parsed_ex in exs:
        by_op.setdefault(parsed_ex.op, []).append(parsed_ex)

    # Detect concat types for each operator
    concat_types: dict[str, str] = {}
    for op, op_exs in by_op.items():
        ct = _concat_type(op_exs)
        if ct is not None:
            concat_types[op] = ct

    # Check question operator for concatenation type (default to fwd if unknown)
    if q_op in by_op:
        q_ct = _concat_type(by_op[q_op])
        if q_ct is None:
            q_ct = "fwd"
    else:
        q_ct = "fwd"

    if q_ct == "fwd":
        answer = q_a[0] + q_a[1] + q_b[0] + q_b[1]
    else:
        answer = q_b[0] + q_b[1] + q_a[0] + q_a[1]
    if answer != problem.answer:
        return None

    # Generate compact slot-copy trace.
    lines: list[str] = []
    lines.append(
        "We need to infer how the operator splits and concatenates the symbols."
    )
    lines.append("I will compute the answer and state it at the end.")
    lines.append("")

    lines.append("Each input has two left symbols, one operator, and two right symbols.")
    lines.append("")

    # Show each example as slots instead of relying on raw joined strings.
    for idx, (ex, ex_parsed) in enumerate(zip(problem.examples, exs), start=1):
        orig_out = str(ex.output_value)
        slot_values = [
            ("L0", ex_parsed.a[0]),
            ("L1", ex_parsed.a[1]),
            ("OP", ex_parsed.op),
            ("R0", ex_parsed.b[0]),
            ("R1", ex_parsed.b[1]),
        ]
        lines.append(f"Example {idx}:")
        lines.append(f"  input slots: {_slot_list(slot_values)}")
        lines.append(f"  left slots: L0, L1")
        lines.append(f"  right slots: R0, R1")
        lines.append(f"  observed output: {_box(orig_out)}")

        fwd = ex_parsed.a[0] + ex_parsed.a[1] + ex_parsed.b[0] + ex_parsed.b[1]
        rev = ex_parsed.b[0] + ex_parsed.b[1] + ex_parsed.a[0] + ex_parsed.a[1]
        is_fwd = orig_out == fwd
        is_rev = orig_out == rev

        lines.append(
            f"  left-then-right slots [L0,L1,R0,R1] -> {_box(fwd)}: {'match' if is_fwd else 'mismatch'}"
        )
        lines.append(
            f"  right-then-left slots [R0,R1,L0,L1] -> {_box(rev)}: {'match' if is_rev else 'mismatch'}"
        )

        # Operator line with type
        ct = concat_types.get(ex_parsed.op)
        if ct == "fwd":
            op_type = "left-then-right"
        elif ct == "rev":
            op_type = "right-then-left"
        else:
            op_type = "unknown"
        lines.append(f"  operator {quote(ex_parsed.op)} rule: {op_type}")
        lines.append("")

    # Apply to question
    q_op_known = q_op in concat_types
    op_label = "left-then-right" if q_ct == "fwd" else "right-then-left"

    q_slot_values = [
        ("L0", q_a[0]),
        ("L1", q_a[1]),
        ("OP", q_op),
        ("R0", q_b[0]),
        ("R1", q_b[1]),
    ]
    lines.append("Question:")
    lines.append(f"  input slots: {_slot_list(q_slot_values)}")
    lines.append(f"  left slots: L0=【{q_a[0]}】, L1=【{q_a[1]}】")
    lines.append(f"  right slots: R0=【{q_b[0]}】, R1=【{q_b[1]}】")
    lines.append("")

    if q_op_known:
        lines.append(
            f"The question operator is {quote(q_op)}, which is {op_label}."
        )
    else:
        lines.append(f"The question operator is {quote(q_op)}, which is unknown.")
        lines.append(
            "As the question operator is unknown, we default to concatenation."
        )
    lines.append("")

    if q_ct == "fwd":
        output_slots = [
            ("out0", "L0", q_a[0]),
            ("out1", "L1", q_a[1]),
            ("out2", "R0", q_b[0]),
            ("out3", "R1", q_b[1]),
        ]
    else:
        output_slots = [
            ("out0", "R0", q_b[0]),
            ("out1", "R1", q_b[1]),
            ("out2", "L0", q_a[0]),
            ("out3", "L1", q_a[1]),
        ]

    lines.append("Apply the rule by copying slots:")
    for out_name, source_name, value in output_slots:
        lines.append(f"  {out_name} = {source_name} = 【{value}】")
    joined_terms = " + ".join(f"【{value}】" for _, _, value in output_slots)
    lines.append(f"Join outputs without spaces: {joined_terms} = {answer}")
    lines.append("")
    lines.append("I will now state the final answer.")
    lines.append(f"Final answer is: {answer}")
    return "\n".join(lines)
