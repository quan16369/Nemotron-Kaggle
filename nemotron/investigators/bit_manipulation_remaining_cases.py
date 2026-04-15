"""Inspect remaining bit_manipulation failures for different solver variants.

Usage:
    python investigators/bit_manipulation_remaining_cases.py
    python investigators/bit_manipulation_remaining_cases.py --repair
    python investigators/bit_manipulation_remaining_cases.py --compare-repair
    python investigators/bit_manipulation_remaining_cases.py --limit 100 --show 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.store_types import Problem
from reasoning import compare_answer, extract_answer

PROBLEMS_INDEX = Path("problems.jsonl")


@dataclass
class ParsedRule:
    family: str
    primary: int | None
    secondary: int | None


def parse_selected_rules(reasoning_text: str) -> list[ParsedRule] | None:
    """Extract selected rules from the legacy trace format."""
    lines = reasoning_text.splitlines()
    in_selected = False
    rules: list[ParsedRule] = []
    for line in lines:
        if line.strip() == "Selected":
            in_selected = True
            continue
        if not in_selected:
            continue
        if not line.strip():
            if rules:
                break
            continue
        match = re.match(r"^\d+\s+(.+)$", line.strip())
        if not match:
            break
        expr = match.group(1)
        family, primary, secondary = _parse_expr(expr)
        rules.append(ParsedRule(family, primary, secondary))
    return rules if len(rules) == 8 else None


def _parse_expr(expr: str) -> tuple[str, int | None, int | None]:
    if expr.startswith("default"):
        return ("DEFAULT", None, None)
    if expr.startswith("C"):
        return ("Constant", None, None)
    if expr.startswith("T3"):
        return ("T3", None, None)
    for prefix in ("XOR-NOT", "OR-NOT", "AND-NOT", "XOR", "OR", "AND", "NOT", "I"):
        if expr.startswith(prefix):
            digits = expr[len(prefix) :]
            if len(digits) == 2 and digits.isdigit():
                return (prefix, int(digits[0]), int(digits[1]))
            if len(digits) == 1 and digits.isdigit():
                return (prefix, int(digits[0]), None)
            return (prefix, None, None)
    return (expr, None, None)


def _stride_consistent(prev: ParsedRule, curr: ParsedRule) -> bool:
    if prev.family != curr.family:
        return False
    if prev.family in ("Constant", "DEFAULT", "T3"):
        return True
    if prev.primary is not None and curr.primary is not None:
        if (prev.primary + 1) % 8 != curr.primary:
            return False
    if prev.secondary is not None and curr.secondary is not None:
        if (prev.secondary + 1) % 8 != curr.secondary:
            return False
    return True


def count_sections(rules: list[ParsedRule]) -> int:
    if not rules:
        return 0
    sections = 1
    for i in range(1, len(rules)):
        if not _stride_consistent(rules[i - 1], rules[i]):
            sections += 1
    return sections


def iter_bit_problem_ids(limit: int | None = None) -> list[str]:
    ids: list[str] = []
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["category"] != "bit_manipulation":
                continue
            ids.append(entry["id"])
            if limit is not None and len(ids) >= limit:
                break
    return ids


def evaluate_variant(
    ids: list[str],
    *,
    repair: bool,
    compact: bool,
) -> tuple[int, list[dict[str, object]], Counter[int]]:
    correct = 0
    failures: list[dict[str, object]] = []
    sections = Counter()

    for pid in ids:
        problem = Problem.load_from_json(pid)
        reasoning_text = reasoning_bit_manipulation(
            problem,
            compact=compact,
            enable_three_bit_repair=repair,
        )
        if reasoning_text is None:
            failures.append(
                {
                    "id": pid,
                    "answer": problem.answer,
                    "predicted": "",
                    "sections": None,
                    "selected_families": (),
                }
            )
            continue

        predicted = extract_answer(reasoning_text)
        is_correct = compare_answer(problem.answer, predicted)
        rules = parse_selected_rules(reasoning_text) if not compact else None
        n_sections = count_sections(rules) if rules is not None else None
        if n_sections is not None:
            sections[n_sections] += 1
        if is_correct:
            correct += 1
            continue
        failures.append(
            {
                "id": pid,
                "answer": problem.answer,
                "predicted": predicted,
                "sections": n_sections,
                "selected_families": tuple(rule.family for rule in rules or []),
            }
        )

    return correct, failures, sections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--show", type=int, default=20, help="Number of failures to print")
    parser.add_argument("--repair", action="store_true", help="Enable 3-bit repair")
    parser.add_argument("--compact", action="store_true", help="Use compact renderer")
    parser.add_argument(
        "--compare-repair",
        action="store_true",
        help="Compare repair=False vs repair=True on the same problem set",
    )
    args = parser.parse_args()

    ids = iter_bit_problem_ids(args.limit)
    print(f"Loaded {len(ids)} bit_manipulation problems")

    if args.compare_repair:
        base_correct, base_failures, _ = evaluate_variant(ids, repair=False, compact=False)
        repair_correct, repair_failures, _ = evaluate_variant(ids, repair=True, compact=False)
        base_by_id = {entry["id"]: entry for entry in base_failures}
        repair_by_id = {entry["id"]: entry for entry in repair_failures}

        gained = sorted(set(base_by_id) - set(repair_by_id))
        lost = sorted(set(repair_by_id) - set(base_by_id))
        unchanged = sorted(set(base_by_id) & set(repair_by_id))

        print(f"Base:   {base_correct}/{len(ids)}")
        print(f"Repair: {repair_correct}/{len(ids)}")
        print(f"Gained: {len(gained)}")
        print(f"Lost:   {len(lost)}")
        if gained:
            print("Sample gained:")
            for pid in gained[: args.show]:
                entry = base_by_id[pid]
                print(f"  {pid} base={entry['predicted']} gold={entry['answer']}")
        if lost:
            print("Sample lost:")
            for pid in lost[: args.show]:
                entry = repair_by_id[pid]
                print(f"  {pid} repair={entry['predicted']} gold={entry['answer']}")
        if unchanged:
            print("Sample unchanged failures:")
            for pid in unchanged[: args.show]:
                entry = repair_by_id[pid]
                print(
                    f"  {pid} pred={entry['predicted']} gold={entry['answer']} "
                    f"sections={entry['sections']} fams={entry['selected_families']}"
                )
        return

    correct, failures, sections = evaluate_variant(
        ids,
        repair=args.repair,
        compact=args.compact,
    )
    print(f"Accuracy: {correct}/{len(ids)} = {correct / len(ids):.1%}")

    if sections:
        print("\nSection counts:")
        for n_sections in sorted(sections):
            print(f"  {n_sections}: {sections[n_sections]}")

    print("\nSample failures:")
    for entry in failures[: args.show]:
        print(
            f"  {entry['id']} pred={entry['predicted']} gold={entry['answer']} "
            f"sections={entry['sections']} fams={entry['selected_families']}"
        )


if __name__ == "__main__":
    main()
