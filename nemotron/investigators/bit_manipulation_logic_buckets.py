"""Bucket remaining bit_manipulation failures by likely logic class.

Usage:
    python investigators/bit_manipulation_logic_buckets.py
    python investigators/bit_manipulation_logic_buckets.py --limit 200 --show 5

The script evaluates the current default solver in `reasoners.bit_manipulation`,
then tries progressively broader explainers for the failures:

1. Existing whole-word transform solver in `investigators.bit_manipulation`
2. Whole-word majority over three transforms: MAJ(a, b, c)
3. Whole-word choice over three transforms: CH(a, b, c)
4. Generic 3-input bitwise boolean over three transforms
5. Manual investigations already written in `investigations/*.txt`

This is meant for analysis, not submission-time inference.
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

from investigators.bit_manipulation import TRANSFORMS, solve_problem
from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.store_types import Problem
from reasoning import compare_answer, extract_answer

PROBLEMS_INDEX = Path("problems.jsonl")
INVESTIGATIONS_DIR = Path("investigations")


@dataclass
class BucketedFailure:
    id: str
    answer: str
    predicted: str
    bucket: str
    rule: str = ""


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


def _majority(a: int, b: int, c: int) -> int:
    return (a & b) | (a & c) | (b & c)


def _choice(a: int, b: int, c: int) -> int:
    return (a & b) | ((~a & 0xFF) & c)


def _find_majority_rule(data: dict[str, object]) -> str | None:
    examples = data["examples"]
    inputs = [int(ex["input_value"], 2) for ex in examples]
    outputs = [int(ex["output_value"], 2) for ex in examples]
    query = int(data["question"], 2)
    n_examples = len(inputs)
    n_transforms = len(TRANSFORMS)

    results = []
    for _, fn in TRANSFORMS:
        results.append([fn(inp) for inp in inputs])

    for t1 in range(n_transforms):
        r1 = results[t1]
        for t2 in range(t1 + 1, n_transforms):
            r2 = results[t2]
            pair = [r1[i] & r2[i] for i in range(n_examples)]
            for t3 in range(t2 + 1, n_transforms):
                r3 = results[t3]
                if not all(
                    (pair[i] | (r1[i] & r3[i]) | (r2[i] & r3[i])) == outputs[i]
                    for i in range(n_examples)
                ):
                    continue
                answer = _majority(
                    TRANSFORMS[t1][1](query),
                    TRANSFORMS[t2][1](query),
                    TRANSFORMS[t3][1](query),
                )
                if format(answer, "08b") == data["answer"]:
                    return (
                        f"MAJ({TRANSFORMS[t1][0]}, "
                        f"{TRANSFORMS[t2][0]}, {TRANSFORMS[t3][0]})"
                    )
    return None


def _find_choice_rule(data: dict[str, object]) -> str | None:
    examples = data["examples"]
    inputs = [int(ex["input_value"], 2) for ex in examples]
    outputs = [int(ex["output_value"], 2) for ex in examples]
    query = int(data["question"], 2)
    n_examples = len(inputs)
    n_transforms = len(TRANSFORMS)

    results = []
    for _, fn in TRANSFORMS:
        results.append([fn(inp) for inp in inputs])

    for t1 in range(n_transforms):
        r1 = results[t1]
        for t2 in range(n_transforms):
            if t2 == t1:
                continue
            r2 = results[t2]
            for t3 in range(n_transforms):
                if t3 == t1 or t3 == t2:
                    continue
                r3 = results[t3]
                if not all(_choice(r1[i], r2[i], r3[i]) == outputs[i] for i in range(n_examples)):
                    continue
                answer = _choice(
                    TRANSFORMS[t1][1](query),
                    TRANSFORMS[t2][1](query),
                    TRANSFORMS[t3][1](query),
                )
                if format(answer, "08b") == data["answer"]:
                    return (
                        f"CH({TRANSFORMS[t1][0]}, "
                        f"{TRANSFORMS[t2][0]}, {TRANSFORMS[t3][0]})"
                    )
    return None


def _truth_table_anf(bits: str) -> str:
    values = [int(bit) for bit in bits]
    coeffs = values[:]
    for axis in range(3):
        step = 1 << axis
        for mask in range(8):
            if mask & step:
                coeffs[mask] ^= coeffs[mask ^ step]
    monomials = ["1", "c", "b", "b*c", "a", "a*c", "a*b", "a*b*c"]
    terms = [monomials[mask] for mask, coeff in enumerate(coeffs) if coeff]
    return " xor ".join(terms) if terms else "0"


def _truth_table_alias(bits: str) -> str:
    if bits == "11010110":
        return "if c then NAND(a,b) else XNOR(a,b)"
    return _truth_table_anf(bits)


def _find_generic_three_input_rule(data: dict[str, object]) -> tuple[str, str] | None:
    """Find any 3-input pointwise boolean rule over three transformed bytes.

    Returns:
    - ("whole_word_generic3_forced", rule_text) when the query is uniquely
      determined by example-observed truth table rows
    - ("whole_word_generic3_possible", rule_text) when the examples allow a
      compatible truth table that matches the gold answer, but the query needs
      at least one unseen row
    """

    examples = data["examples"]
    inputs = [int(ex["input_value"], 2) for ex in examples]
    outputs = [int(ex["output_value"], 2) for ex in examples]
    query = int(data["question"], 2)
    gold = data["answer"]
    n_examples = len(inputs)

    transformed = []
    for name, fn in TRANSFORMS:
        transformed.append((name, [fn(inp) for inp in inputs], fn(query)))

    gold_bits = list(map(int, reversed(gold)))
    for i, (name_a, rows_a, query_a) in enumerate(transformed):
        for j, (name_b, rows_b, query_b) in enumerate(transformed):
            if j == i:
                continue
            for k, (name_c, rows_c, query_c) in enumerate(transformed):
                if k == i or k == j:
                    continue

                truth_table: dict[int, int] = {}
                consistent = True
                for ex_idx in range(n_examples):
                    out = outputs[ex_idx]
                    aval = rows_a[ex_idx]
                    bval = rows_b[ex_idx]
                    cval = rows_c[ex_idx]
                    for bit in range(8):
                        combo = (
                            (((aval >> bit) & 1) << 2)
                            | (((bval >> bit) & 1) << 1)
                            | ((cval >> bit) & 1)
                        )
                        expected = (out >> bit) & 1
                        prev = truth_table.get(combo)
                        if prev is None:
                            truth_table[combo] = expected
                        elif prev != expected:
                            consistent = False
                            break
                    if not consistent:
                        break
                if not consistent:
                    continue

                forced = True
                for bit, gold_bit in enumerate(gold_bits):
                    combo = (
                        (((query_a >> bit) & 1) << 2)
                        | (((query_b >> bit) & 1) << 1)
                        | ((query_c >> bit) & 1)
                    )
                    seen = truth_table.get(combo)
                    if seen is None:
                        forced = False
                        continue
                    if seen != gold_bit:
                        consistent = False
                        break
                if not consistent:
                    continue

                bits = "".join(
                    str(truth_table[combo]) if combo in truth_table else "?"
                    for combo in range(8)
                )
                alias = _truth_table_alias(bits.replace("?", "0")) if "?" not in bits else ""
                rule = (
                    f"GEN3({name_a}, {name_b}, {name_c}) truth={bits}"
                    + (f" ({alias})" if alias else "")
                )
                bucket = "whole_word_generic3_forced" if forced else "whole_word_generic3_possible"
                return (bucket, rule)
    return None


def _manual_bucket(problem_id: str) -> tuple[str, str]:
    path = INVESTIGATIONS_DIR / f"{problem_id}.txt"
    if not path.exists():
        return ("unknown_no_investigation", "")

    text = path.read_text()
    text_lower = text.lower()

    match = re.search(r"^rule:\s*(.+)$", text, flags=re.MULTILINE)
    rule = match.group(1).strip() if match else ""

    if "majority" in text_lower or " maj(" in text_lower:
        return ("manual_majority", rule)
    if "choice function" in text_lower or "`ch(" in text_lower:
        return ("manual_choice", rule)
    if "compute each output bit" in text_lower or re.search(r"\by0\s*=|\by1\s*=", text_lower):
        return ("manual_per_bit_boolean", rule)
    if "inferred rule:" in text_lower or "rule:" in text_lower:
        return ("manual_other_custom", rule)
    return ("manual_uncategorized", rule)


def bucket_failures(ids: list[str]) -> tuple[int, list[BucketedFailure]]:
    correct = 0
    failures: list[BucketedFailure] = []

    for problem_id in ids:
        problem = Problem.load_from_json(problem_id)
        reasoning_text = reasoning_bit_manipulation(problem)
        predicted = extract_answer(reasoning_text) if reasoning_text else ""
        if compare_answer(problem.answer, predicted):
            correct += 1
            continue

        problem_path = Path("problems") / f"{problem_id}.jsonl"
        data = json.loads(problem_path.read_text().splitlines()[0])

        whole_word_answer, whole_word_rule, _ = solve_problem(data)
        if whole_word_answer == data["answer"]:
            failures.append(
                BucketedFailure(
                    id=problem_id,
                    answer=problem.answer,
                    predicted=predicted,
                    bucket="whole_word_existing_solver",
                    rule=whole_word_rule or "",
                )
            )
            continue

        majority_rule = _find_majority_rule(data)
        if majority_rule is not None:
            failures.append(
                BucketedFailure(
                    id=problem_id,
                    answer=problem.answer,
                    predicted=predicted,
                    bucket="whole_word_majority",
                    rule=majority_rule,
                )
            )
            continue

        choice_rule = _find_choice_rule(data)
        if choice_rule is not None:
            failures.append(
                BucketedFailure(
                    id=problem_id,
                    answer=problem.answer,
                    predicted=predicted,
                    bucket="whole_word_choice",
                    rule=choice_rule,
                )
            )
            continue

        generic_three_input = _find_generic_three_input_rule(data)
        if generic_three_input is not None:
            bucket, rule = generic_three_input
            failures.append(
                BucketedFailure(
                    id=problem_id,
                    answer=problem.answer,
                    predicted=predicted,
                    bucket=bucket,
                    rule=rule,
                )
            )
            continue

        bucket, rule = _manual_bucket(problem_id)
        failures.append(
            BucketedFailure(
                id=problem_id,
                answer=problem.answer,
                predicted=predicted,
                bucket=bucket,
                rule=rule,
            )
        )

    return correct, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--show", type=int, default=8, help="Sample failures per bucket")
    args = parser.parse_args()

    ids = iter_bit_problem_ids(args.limit)
    correct, failures = bucket_failures(ids)
    total = len(ids)

    bucket_counts = Counter(failure.bucket for failure in failures)
    bucket_examples: dict[str, list[BucketedFailure]] = defaultdict(list)
    for failure in failures:
        if len(bucket_examples[failure.bucket]) < args.show:
            bucket_examples[failure.bucket].append(failure)

    explained = sum(
        bucket_counts[bucket]
        for bucket in (
            "whole_word_existing_solver",
            "whole_word_majority",
            "whole_word_choice",
            "whole_word_generic3_forced",
            "whole_word_generic3_possible",
            "manual_majority",
            "manual_choice",
            "manual_per_bit_boolean",
            "manual_other_custom",
            "manual_uncategorized",
        )
    )

    print(f"Loaded {total} bit_manipulation problems")
    print(f"Current solver correct: {correct}/{total} = {correct / total:.1%}")
    print(f"Current solver failures: {len(failures)}")
    print(f"Explained by broader logic buckets: {explained}/{len(failures)}")
    print("")
    print("Bucket counts:")
    for bucket, count in bucket_counts.most_common():
        print(f"  {bucket}: {count}")

    print("")
    print("Sample cases:")
    for bucket, examples in bucket_examples.items():
        print(f"[{bucket}]")
        for failure in examples:
            suffix = f" rule={failure.rule}" if failure.rule else ""
            print(
                f"  {failure.id} pred={failure.predicted} gold={failure.answer}{suffix}"
            )


if __name__ == "__main__":
    main()
