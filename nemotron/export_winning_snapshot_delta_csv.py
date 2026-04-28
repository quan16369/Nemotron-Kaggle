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
    build_current_correct_base_records,
    load_snapshot_records,
    merge_snapshot_with_current_delta,
)

BASE_DIR = Path(__file__).parent
DEFAULT_SNAPSHOT_DIR = BASE_DIR / "training" / "sft" / "04-08-16-14"
DEFAULT_OUTPUT = BASE_DIR / "winning_snapshot_delta_manifest.csv"
TOKENIZER_JSON = BASE_DIR / "tokenizer.json"
MAX_SEQ_LEN = 8192
DEFAULT_SAFE_DELTA_CATEGORIES = [
    "bit_manipulation",
    "equation_numeric_guess",
]


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

    snapshot_records, snapshot_config = load_snapshot_records(
        args.snapshot_dir,
        max_seq_len=MAX_SEQ_LEN,
    )
    final_records = snapshot_records
    delta_stats = None

    if not args.no_delta:
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
        delta_categories = set(args.delta_categories)
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
        )
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

    fieldnames = [
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in final_records:
            writer.writerow(
                {
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
            )

    print(
        json.dumps(
            {
                "snapshot_dir": str(args.snapshot_dir),
                "snapshot_examples": len(snapshot_records),
                "final_examples": len(final_records),
                "output": str(args.output),
                "chat_tokenizer_path": (
                    None if args.no_delta else resolve_chat_tokenizer_path(args.chat_tokenizer_path)
                ),
                "bit_manipulation_compact": args.bit_manipulation_compact,
                "bit_manipulation_three_bit_repair": args.bit_manipulation_three_bit_repair,
                "bit_manipulation_use_legacy": args.use_legacy_bit_manipulation,
                "delta_categories": args.delta_categories,
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
