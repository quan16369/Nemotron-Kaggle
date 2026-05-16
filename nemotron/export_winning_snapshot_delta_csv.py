"""Export the safe winning-snapshot-plus-delta training set to one CSV manifest.

Each CSV row is a single training record and contains exact token IDs plus mask,
so training can be reproduced from the CSV alone without shipping the full
snapshot directory structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

try:
    from tokenizers import Tokenizer  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional local dependency
    Tokenizer = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional local dependency
    AutoTokenizer = None  # type: ignore[assignment]

from winning_snapshot_delta import (
    COMPETITION_CATEGORIES,
    build_current_correct_base_records,
    load_snapshot_records,
    merge_snapshot_with_current_delta,
)
from three_agent import available_negative_constraints, build_legacy_double_check_completion

BASE_DIR = Path(__file__).parent
DEFAULT_SNAPSHOT_DIR = BASE_DIR / "training" / "sft" / "04-08-16-14"
DEFAULT_OUTPUT = BASE_DIR / "winning_snapshot_delta_manifest.csv"
TOKENIZER_JSON = BASE_DIR / "tokenizer.json"
MAX_SEQ_LEN = 8192
DEFAULT_SAFE_DELTA_CATEGORIES = [
    "bit_manipulation",
    "equation_numeric_guess",
]

MANIFEST_FIELDNAMES = [
    "problem_id",
    "source_problem_id",
    "category",
    "segment",
    "num_loss_tokens",
    "completion_token_count",
    "token_count",
    "input_ids_json",
    "mask_json",
]


def _stable_fraction(key: str) -> float:
    import hashlib

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def record_to_manifest_row(record: dict) -> dict[str, str | int]:
    return {
        "problem_id": record["problem_id"],
        "source_problem_id": record["source_problem_id"],
        "category": record["category"],
        "segment": record["segment"],
        "num_loss_tokens": record["num_loss_tokens"],
        "completion_token_count": record["completion_token_count"],
        "token_count": len(record["input_ids"]),
        "input_ids_json": json.dumps(record["input_ids"]),
        "mask_json": json.dumps(
            [1 if label != -100 else 0 for label in record["labels"]]
        ),
    }


def append_negative_records_to_base_manifest(
    *,
    base_path: Path,
    output_path: Path,
    generated_records: list[dict],
) -> dict[str, object]:
    negative_records = [
        record for record in generated_records if "-neg-" in record["problem_id"]
    ]
    negative_records.sort(key=lambda record: record["problem_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_rows = 0
    appended_rows = 0
    skipped_duplicates = 0
    seen_keys: set[tuple[str, str]] = set()
    base_categories: Counter[str] = Counter()
    appended_categories: Counter[str] = Counter()

    with base_path.open(newline="") as src, output_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames != MANIFEST_FIELDNAMES:
            raise ValueError(
                f"Base manifest header mismatch. Expected {MANIFEST_FIELDNAMES}, "
                f"got {reader.fieldnames}"
            )
        writer = csv.DictWriter(dst, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()

        for row in reader:
            writer.writerow(row)
            base_rows += 1
            base_categories[row["category"]] += 1
            seen_keys.add((row["problem_id"], row["segment"]))

        for record in negative_records:
            key = (record["problem_id"], record["segment"])
            if key in seen_keys:
                skipped_duplicates += 1
                continue
            row = record_to_manifest_row(record)
            writer.writerow(row)
            seen_keys.add(key)
            appended_rows += 1
            appended_categories[str(record["category"])] += 1

    return {
        "base_manifest": str(base_path),
        "base_rows": base_rows,
        "appended_negative_rows": appended_rows,
        "skipped_duplicate_negative_rows": skipped_duplicates,
        "final_rows": base_rows + appended_rows,
        "base_categories": dict(sorted(base_categories.items())),
        "appended_categories": dict(sorted(appended_categories.items())),
    }


def _load_answers(repo_dir: Path) -> dict[str, str]:
    with (repo_dir / "train.csv").open(newline="") as f:
        return {row["id"]: row["answer"] for row in csv.DictReader(f)}


def _completion_from_manifest_row(
    row: dict[str, str],
    completion_tokenizer: Tokenizer,
) -> tuple[list[int], list[int], str]:
    input_ids = json.loads(row["input_ids_json"])
    mask = json.loads(row["mask_json"])
    first_loss = next((idx for idx, value in enumerate(mask) if value == 1), len(mask))
    prompt_ids = input_ids[:first_loss]
    completion_ids = input_ids[first_loss:]
    completion_text = completion_tokenizer.decode(
        completion_ids,
        skip_special_tokens=False,
    )
    return prompt_ids, completion_ids, completion_text


def append_legacy_double_check_to_base_manifest(
    *,
    base_path: Path,
    output_path: Path,
    completion_tokenizer: Tokenizer,
    fraction: float,
    max_seq_len: int,
) -> dict[str, object]:
    answers = _load_answers(BASE_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_rows = 0
    appended_rows = 0
    skipped_rows = 0
    base_categories: Counter[str] = Counter()
    appended_categories: Counter[str] = Counter()

    with base_path.open(newline="") as src, output_path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames != MANIFEST_FIELDNAMES:
            raise ValueError(
                f"Base manifest header mismatch. Expected {MANIFEST_FIELDNAMES}, "
                f"got {reader.fieldnames}"
            )
        writer = csv.DictWriter(dst, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()

        for row in reader:
            writer.writerow(row)
            base_rows += 1
            category = row["category"]
            base_categories[category] += 1
            if category not in {"bit_manipulation", "gravity"}:
                continue

            source_problem_id = row["source_problem_id"]
            answer = answers.get(source_problem_id)
            if answer is None:
                skipped_rows += 1
                continue

            prompt_ids, _, completion_text = _completion_from_manifest_row(
                row,
                completion_tokenizer,
            )
            constraints = available_negative_constraints(
                category=category,
                answer=answer,
                reasoning_text=completion_text,
            )
            for failed_constraint in constraints:
                negative_problem_id = f"{row['problem_id']}-neg-{failed_constraint}"
                negative_key = f"{negative_problem_id}:legacy-double-check"
                if _stable_fraction(negative_key) >= fraction:
                    continue

                negative_completion = build_legacy_double_check_completion(
                    completion_text,
                    category=category,
                    answer=answer,
                    problem_id=negative_problem_id,
                    forced_failed_constraint=failed_constraint,
                )
                negative_completion_ids = completion_tokenizer.encode(
                    negative_completion,
                    add_special_tokens=False,
                ).ids
                negative_input_ids = prompt_ids + negative_completion_ids
                if len(negative_input_ids) > max_seq_len:
                    skipped_rows += 1
                    continue
                negative_mask = [0] * len(prompt_ids) + [1] * len(negative_completion_ids)
                negative_row = dict(row)
                negative_row["problem_id"] = negative_problem_id
                negative_row["num_loss_tokens"] = str(len(negative_completion_ids))
                negative_row["completion_token_count"] = str(len(negative_completion_ids))
                negative_row["token_count"] = str(len(negative_input_ids))
                negative_row["input_ids_json"] = json.dumps(negative_input_ids)
                negative_row["mask_json"] = json.dumps(negative_mask)
                writer.writerow(negative_row)
                appended_rows += 1
                appended_categories[category] += 1

    return {
        "base_manifest": str(base_path),
        "base_rows": base_rows,
        "appended_legacy_double_check_rows": appended_rows,
        "skipped_rows": skipped_rows,
        "final_rows": base_rows + appended_rows,
        "base_categories": dict(sorted(base_categories.items())),
        "appended_categories": dict(sorted(appended_categories.items())),
    }


def parse_category_value_specs(raw_values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Expected CATEGORY=VALUE, got: {raw}")
        category, value = raw.split("=", 1)
        category = category.strip()
        value = value.strip()
        if not category or not value:
            raise ValueError(f"Expected CATEGORY=VALUE, got: {raw}")
        parsed[category] = value
    return parsed


def assign_length_buckets(values: list[int], num_buckets: int) -> list[int]:
    if not values:
        return []
    if len(values) == 1 or len(set(values)) == 1:
        return [0] * len(values)

    ordered = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    buckets = [0] * len(values)
    bucket_count = min(num_buckets, len(values))
    for rank, original_idx in enumerate(ordered):
        buckets[original_idx] = min(bucket_count - 1, (rank * bucket_count) // len(values))
    return buckets


def maybe_sample_records(
    records: list[dict],
    *,
    keep_fraction_specs: list[str],
    keep_problem_specs: list[str],
    sample_seed: int,
    sample_length_buckets: int,
) -> tuple[list[dict], dict | None]:
    if not keep_fraction_specs and not keep_problem_specs:
        return records, None

    keep_fractions = parse_category_value_specs(keep_fraction_specs)
    keep_problems = parse_category_value_specs(keep_problem_specs)
    plan_categories = set(keep_fractions) | set(keep_problems)

    grouped: dict[tuple[str, str], dict] = {}
    for idx, record in enumerate(records):
        key = (record["category"], str(record["source_problem_id"]))
        group = grouped.setdefault(
            key,
            {
                "category": record["category"],
                "source_problem_id": str(record["source_problem_id"]),
                "record_indices": [],
                "completion_tokens": 0,
            },
        )
        group["record_indices"].append(idx)
        group["completion_tokens"] += int(record["completion_token_count"])

    by_category: dict[str, list[dict]] = {}
    for group in grouped.values():
        by_category.setdefault(group["category"], []).append(group)

    rng = random.Random(sample_seed)
    kept_keys: set[tuple[str, str]] = set()
    category_summary: dict[str, dict[str, int | float]] = {}

    for category, category_groups in by_category.items():
        total_problems = len(category_groups)
        total_rows = sum(len(group["record_indices"]) for group in category_groups)
        total_completion_tokens = sum(group["completion_tokens"] for group in category_groups)

        if category in keep_problems:
            keep_count = min(total_problems, int(keep_problems[category]))
        elif category in keep_fractions:
            keep_fraction = float(keep_fractions[category])
            if not (0.0 < keep_fraction <= 1.0):
                raise ValueError(
                    f"keep-fraction for {category} must be in (0, 1], got {keep_fraction}"
                )
            keep_count = max(1, round(total_problems * keep_fraction))
        else:
            keep_count = total_problems

        if keep_count >= total_problems:
            selected_groups = category_groups
        else:
            bucket_values = assign_length_buckets(
                [group["completion_tokens"] for group in category_groups],
                sample_length_buckets,
            )
            strata: dict[tuple[int, int], list[dict]] = {}
            for group, bucket in zip(category_groups, bucket_values):
                key = (len(group["record_indices"]), bucket)
                strata.setdefault(key, []).append(group)

            selected_groups = []
            leftovers: list[dict] = []
            remaining_slots = keep_count
            total = total_problems
            for stratum_groups in strata.values():
                exact = keep_count * len(stratum_groups) / total
                take = min(len(stratum_groups), math.floor(exact))
                if take > 0:
                    chosen = rng.sample(stratum_groups, take)
                    selected_groups.extend(chosen)
                    chosen_ids = {id(group) for group in chosen}
                    leftovers.extend(
                        group for group in stratum_groups if id(group) not in chosen_ids
                    )
                    remaining_slots -= take
                else:
                    leftovers.extend(stratum_groups)

            if remaining_slots > 0 and leftovers:
                selected_groups.extend(rng.sample(leftovers, min(remaining_slots, len(leftovers))))

        for group in selected_groups:
            kept_keys.add((group["category"], group["source_problem_id"]))

        kept_rows = sum(len(group["record_indices"]) for group in selected_groups)
        kept_completion_tokens = sum(group["completion_tokens"] for group in selected_groups)
        category_summary[category] = {
            "problems_before": total_problems,
            "problems_after": len(selected_groups),
            "rows_before": total_rows,
            "rows_after": kept_rows,
            "completion_tokens_before": total_completion_tokens,
            "completion_tokens_after": kept_completion_tokens,
        }

    sampled_records = [
        record
        for record in records
        if (record["category"], str(record["source_problem_id"])) in kept_keys
    ]
    sample_stats = {
        "seed": sample_seed,
        "length_buckets": sample_length_buckets,
        "keep_fractions": keep_fractions,
        "keep_problems": {k: int(v) for k, v in keep_problems.items()},
        "rows_before": len(records),
        "rows_after": len(sampled_records),
        "completion_tokens_before": sum(int(record["completion_token_count"]) for record in records),
        "completion_tokens_after": sum(
            int(record["completion_token_count"]) for record in sampled_records
        ),
        "categories": {
            category: summary
            for category, summary in sorted(category_summary.items())
            if category in plan_categories
        },
    }
    return sampled_records, sample_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="Path to the winning snapshot directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV manifest path",
    )
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Export the raw winning snapshot only",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help=(
            "Export only current generated records. This is the mode to use when "
            "all rows must share the same current 3-agent completion format."
        ),
    )
    parser.add_argument(
        "--use-existing-reasoning-files",
        action="store_true",
        help="Prefer reasoning/*.txt over regenerating with current code when building delta rows",
    )
    parser.add_argument(
        "--chat-tokenizer-path",
        type=str,
        default=None,
        help=(
            "Local path for the Nemotron chat tokenizer/model directory. "
            "If omitted, the script will try common Kaggle input paths."
        ),
    )
    parser.add_argument(
        "--bit-manipulation-compact",
        action="store_true",
        help="Use compact=True when regenerating bit_manipulation delta traces",
    )
    parser.add_argument(
        "--bit-manipulation-three-bit-repair",
        action="store_true",
        help="Use enable_three_bit_repair=True when regenerating bit_manipulation delta traces",
    )
    parser.add_argument(
        "--use-legacy-bit-manipulation",
        action="store_true",
        help="Disable whole-word bit_manipulation solver additions and use the legacy per-bit path",
    )
    parser.add_argument(
        "--delta-categories",
        nargs="+",
        default=DEFAULT_SAFE_DELTA_CATEGORIES,
        help=(
            "Categories allowed to contribute current-code delta records. "
            "Default is the safe hard-category set: bit_manipulation equation_numeric_guess."
        ),
    )
    parser.add_argument(
        "--all-current-categories",
        action="store_true",
        help="With --current-only, include every competition category the current reasoners solve correctly.",
    )
    parser.add_argument(
        "--augment-negative-criteria",
        action="store_true",
        help=(
            "With --current-only, add extra bit_manipulation/gravity rows where one "
            "verifier criterion is forced to fail and then corrected."
        ),
    )
    parser.add_argument(
        "--augment-negative-criteria-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of possible criterion-negative rows to add when "
            "--augment-negative-criteria is enabled. Use 0.10 for 10%% or 0.05 for 5%%."
        ),
    )
    parser.add_argument(
        "--append-negative-to-base",
        type=Path,
        default=None,
        metavar="BASE_MANIFEST",
        help=(
            "With --current-only --augment-negative-criteria, keep this base manifest "
            "unchanged and append only criterion-negative rows to --output."
        ),
    )
    parser.add_argument(
        "--append-negative-format",
        choices=("legacy_double_check", "three_agent"),
        default="legacy_double_check",
        help=(
            "Format for rows appended via --append-negative-to-base. "
            "legacy_double_check preserves the base completion style and inserts a "
            "short Double-check prose paragraph."
        ),
    )
    parser.add_argument(
        "--keep-fraction",
        action="append",
        default=[],
        metavar="CATEGORY=FRACTION",
        help=(
            "After merging snapshot+delta, keep this fraction of source problems for a category. "
            "Sampling is done by source_problem_id, not by row."
        ),
    )
    parser.add_argument(
        "--keep-problems",
        action="append",
        default=[],
        metavar="CATEGORY=COUNT",
        help=(
            "After merging snapshot+delta, keep exactly this many source problems for a category. "
            "Overrides --keep-fraction for that category."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=123,
        help="Random seed for category downsampling.",
    )
    parser.add_argument(
        "--sample-length-buckets",
        type=int,
        default=10,
        help="Number of completion-length buckets used during within-category sampling.",
    )
    return parser.parse_args()


def resolve_chat_tokenizer_path(explicit: str | None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)

    env_value = os.environ.get("NEMOTRON_CHAT_TOKENIZER_PATH")
    if env_value:
        candidates.append(env_value)

    candidates.extend(
        [
            "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
            "/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
            "/kaggle/input/Nemotron-3-Nano-30B-A3B-BF16/transformers/default/1",
        ]
    )

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a local Nemotron chat tokenizer path. "
        "Pass --chat-tokenizer-path or set NEMOTRON_CHAT_TOKENIZER_PATH."
    )


def main() -> None:
    args = parse_args()

    if args.current_only and args.no_delta:
        raise ValueError("--current-only and --no-delta cannot be used together")
    if args.augment_negative_criteria and not args.current_only:
        raise ValueError("--augment-negative-criteria is currently supported only with --current-only")
    if not (0.0 <= args.augment_negative_criteria_fraction <= 1.0):
        raise ValueError("--augment-negative-criteria-fraction must be between 0.0 and 1.0")
    if args.append_negative_to_base is not None and not (
        args.current_only and args.augment_negative_criteria
    ):
        raise ValueError(
            "--append-negative-to-base requires --current-only and --augment-negative-criteria"
        )

    if (
        args.append_negative_to_base is not None
        and args.append_negative_format == "legacy_double_check"
    ):
        if Tokenizer is None:
            raise ModuleNotFoundError("tokenizers is required to append legacy double-check rows")
        completion_tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
        append_stats = append_legacy_double_check_to_base_manifest(
            base_path=args.append_negative_to_base,
            output_path=args.output,
            completion_tokenizer=completion_tokenizer,
            fraction=args.augment_negative_criteria_fraction,
            max_seq_len=MAX_SEQ_LEN,
        )
        print(
            json.dumps(
                {
                    "current_only": args.current_only,
                    "output": str(args.output),
                    "augment_negative_criteria": args.augment_negative_criteria,
                    "augment_negative_criteria_fraction": args.augment_negative_criteria_fraction,
                    "append_negative_to_base": str(args.append_negative_to_base),
                    "append_negative_format": args.append_negative_format,
                    "append_stats": append_stats,
                },
                indent=2,
            )
        )
        return

    snapshot_records = []
    snapshot_config = {}
    if not args.current_only:
        snapshot_records, snapshot_config = load_snapshot_records(
            args.snapshot_dir,
            max_seq_len=MAX_SEQ_LEN,
        )
    final_records = snapshot_records
    delta_stats = None

    if args.current_only or not args.no_delta:
        if Tokenizer is None or AutoTokenizer is None:
            raise ModuleNotFoundError(
                "tokenizers and transformers are required to export the delta manifest"
            )

        chat_tokenizer_path = resolve_chat_tokenizer_path(args.chat_tokenizer_path)
        chat_tokenizer = AutoTokenizer.from_pretrained(
            chat_tokenizer_path,
            trust_remote_code=True,
        )
        completion_tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
        delta_categories = (
            set(COMPETITION_CATEGORIES)
            if args.current_only and args.all_current_categories
            else set(args.delta_categories)
        )
        current_correct_base_records = build_current_correct_base_records(
            repo_dir=BASE_DIR,
            chat_tokenizer=chat_tokenizer,
            completion_tokenizer=completion_tokenizer,
            max_seq_len=MAX_SEQ_LEN,
            use_existing_reasoning_files=args.use_existing_reasoning_files,
            bit_manipulation_compact=args.bit_manipulation_compact,
            bit_manipulation_three_bit_repair=args.bit_manipulation_three_bit_repair,
            bit_manipulation_use_legacy=args.use_legacy_bit_manipulation,
            delta_categories=delta_categories,
            augment_negative_criteria=args.augment_negative_criteria,
            augment_negative_criteria_fraction=args.augment_negative_criteria_fraction,
        )
        if args.current_only:
            final_records = sorted(
                current_correct_base_records.values(),
                key=lambda record: record["problem_id"],
            )
            if args.append_negative_to_base is not None:
                append_stats = append_negative_records_to_base_manifest(
                    base_path=args.append_negative_to_base,
                    output_path=args.output,
                    generated_records=final_records,
                )
                print(
                    json.dumps(
                        {
                            "snapshot_dir": str(args.snapshot_dir),
                            "current_only": args.current_only,
                            "snapshot_examples": len(snapshot_records),
                            "generated_examples": len(final_records),
                            "final_examples": append_stats["final_rows"],
                            "output": str(args.output),
                            "chat_tokenizer_path": resolve_chat_tokenizer_path(
                                args.chat_tokenizer_path
                            ),
                            "bit_manipulation_compact": args.bit_manipulation_compact,
                            "bit_manipulation_three_bit_repair": args.bit_manipulation_three_bit_repair,
                            "bit_manipulation_use_legacy": args.use_legacy_bit_manipulation,
                            "delta_categories": sorted(delta_categories),
                            "all_current_categories": args.all_current_categories,
                            "augment_negative_criteria": args.augment_negative_criteria,
                            "augment_negative_criteria_fraction": args.augment_negative_criteria_fraction,
                            "append_negative_to_base": str(args.append_negative_to_base),
                            "append_negative_format": args.append_negative_format,
                            "append_stats": append_stats,
                        },
                        indent=2,
                    )
                )
                return
        else:
            final_records, delta_stats = merge_snapshot_with_current_delta(
                snapshot_records,
                current_correct_base_records,
            )

    final_records, sample_stats = maybe_sample_records(
        final_records,
        keep_fraction_specs=args.keep_fraction,
        keep_problem_specs=args.keep_problems,
        sample_seed=args.sample_seed,
        sample_length_buckets=args.sample_length_buckets,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for record in final_records:
            writer.writerow(record_to_manifest_row(record))

    print(
        json.dumps(
            {
                "snapshot_dir": str(args.snapshot_dir),
                "current_only": args.current_only,
                "snapshot_examples": len(snapshot_records),
                "final_examples": len(final_records),
                "output": str(args.output),
                "chat_tokenizer_path": (
                    None
                    if args.no_delta
                    else resolve_chat_tokenizer_path(args.chat_tokenizer_path)
                ),
                "bit_manipulation_compact": args.bit_manipulation_compact,
                "bit_manipulation_three_bit_repair": args.bit_manipulation_three_bit_repair,
                "bit_manipulation_use_legacy": args.use_legacy_bit_manipulation,
                "delta_categories": sorted(delta_categories) if "delta_categories" in locals() else args.delta_categories,
                "all_current_categories": args.all_current_categories,
                "augment_negative_criteria": args.augment_negative_criteria,
                "augment_negative_criteria_fraction": args.augment_negative_criteria_fraction,
                "append_negative_to_base": (
                    None
                    if args.append_negative_to_base is None
                    else str(args.append_negative_to_base)
                ),
                "append_negative_format": args.append_negative_format,
                "keep_fraction": args.keep_fraction,
                "keep_problems": args.keep_problems,
                "sample_seed": args.sample_seed,
                "sample_length_buckets": args.sample_length_buckets,
                "snapshot_loss_config": snapshot_config.get("loss_config", {}),
                "snapshot_lr_schedule": snapshot_config.get("lr_schedule", {}),
                "delta_stats": (
                    None
                    if delta_stats is None
                    else {
                        "replaced_source_problems": delta_stats.replaced_source_problems,
                        "replaced_training_records": delta_stats.replaced_training_records,
                        "added_source_problems": delta_stats.added_source_problems,
                        "added_training_records": delta_stats.added_training_records,
                    }
                ),
                "sample_stats": sample_stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
