from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CategoryPlan:
    keep_fraction: float | None = None
    keep_problems: int | None = None


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


def build_category_plan(
    fractions: dict[str, str], problems: dict[str, str]
) -> dict[str, CategoryPlan]:
    categories = set(fractions) | set(problems)
    plan: dict[str, CategoryPlan] = {}
    for category in categories:
        keep_fraction = None
        keep_problems = None
        if category in fractions:
            keep_fraction = float(fractions[category])
            if not (0.0 < keep_fraction <= 1.0):
                raise ValueError(
                    f"keep-fraction for {category} must be in (0, 1], got {keep_fraction}"
                )
        if category in problems:
            keep_problems = int(problems[category])
            if keep_problems <= 0:
                raise ValueError(
                    f"keep-problems for {category} must be positive, got {keep_problems}"
                )
        plan[category] = CategoryPlan(
            keep_fraction=keep_fraction,
            keep_problems=keep_problems,
        )
    return plan


def assign_length_buckets(series: pd.Series, num_buckets: int) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype="int64")
    if len(series) == 1 or series.nunique() == 1:
        return pd.Series([0] * len(series), index=series.index, dtype="int64")

    ranked = series.rank(method="first")
    buckets = pd.qcut(
        ranked,
        q=min(num_buckets, len(series)),
        labels=False,
        duplicates="drop",
    )
    return buckets.astype("int64")


def stratified_problem_sample(
    grouped: pd.DataFrame,
    *,
    keep_count: int,
    num_buckets: int,
    rng: random.Random,
) -> pd.DataFrame:
    if keep_count >= len(grouped):
        return grouped.copy()

    working = grouped.copy()
    working["len_bucket"] = assign_length_buckets(working["problem_completion_tokens"], num_buckets)
    working["stratum"] = list(zip(working["segment_rows"], working["len_bucket"]))

    picked_indices: list[int] = []
    remaining_frames: list[pd.DataFrame] = []

    grouped_strata = list(working.groupby("stratum", sort=False))
    total = len(working)
    remaining_slots = keep_count

    for _, stratum_df in grouped_strata:
        exact = keep_count * len(stratum_df) / total
        take = min(len(stratum_df), math.floor(exact))
        if take > 0:
            sampled = stratum_df.sample(n=take, random_state=rng.randint(0, 2**31 - 1))
            picked_indices.extend(sampled.index.tolist())
            remaining_slots -= take
            stratum_df = stratum_df.drop(sampled.index)
        if not stratum_df.empty:
            remaining_frames.append(stratum_df)

    if remaining_slots > 0 and remaining_frames:
        leftovers = pd.concat(remaining_frames, axis=0)
        extras = leftovers.sample(
            n=min(remaining_slots, len(leftovers)),
            random_state=rng.randint(0, 2**31 - 1),
        )
        picked_indices.extend(extras.index.tolist())

    return working.loc[sorted(set(picked_indices))].drop(columns=["len_bucket", "stratum"])


def summarize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("category")
        .agg(
            rows=("category", "size"),
            problems=("source_problem_id", "nunique"),
            completion_tokens=("completion_token_count", "sum"),
            total_tokens=("token_count", "sum"),
        )
        .sort_values("completion_tokens", ascending=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample a training manifest by source_problem_id while preserving "
            "within-category difficulty/length coverage."
        )
    )
    parser.add_argument("--input", required=True, help="Input manifest CSV")
    parser.add_argument("--output", required=True, help="Output sampled manifest CSV")
    parser.add_argument(
        "--keep-fraction",
        action="append",
        default=[],
        metavar="CATEGORY=FRACTION",
        help="Keep this fraction of problems for a category. Unspecified categories keep all problems.",
    )
    parser.add_argument(
        "--keep-problems",
        action="append",
        default=[],
        metavar="CATEGORY=COUNT",
        help="Keep this many problems for a category. Overrides keep-fraction if both are present.",
    )
    parser.add_argument(
        "--length-buckets",
        type=int,
        default=10,
        help="Number of within-category completion-length buckets used for stratified sampling.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = pd.read_csv(input_path)
    required_cols = {
        "source_problem_id",
        "category",
        "segment",
        "completion_token_count",
        "token_count",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    plan = build_category_plan(
        parse_category_value_specs(args.keep_fraction),
        parse_category_value_specs(args.keep_problems),
    )

    before = summarize_manifest(df)

    grouped = (
        df.groupby(["category", "source_problem_id"], sort=False)
        .agg(
            segment_rows=("segment", "size"),
            problem_completion_tokens=("completion_token_count", "sum"),
            problem_total_tokens=("token_count", "sum"),
        )
        .reset_index()
    )

    kept_groups: list[pd.DataFrame] = []
    for category, category_group in grouped.groupby("category", sort=False):
        category_group = category_group.reset_index(drop=True)
        rule = plan.get(category, CategoryPlan(keep_fraction=1.0))

        if rule.keep_problems is not None:
            keep_count = min(rule.keep_problems, len(category_group))
        else:
            keep_fraction = 1.0 if rule.keep_fraction is None else rule.keep_fraction
            keep_count = max(1, round(len(category_group) * keep_fraction))

        kept_groups.append(
            stratified_problem_sample(
                category_group,
                keep_count=keep_count,
                num_buckets=args.length_buckets,
                rng=rng,
            )
        )

    kept_group_df = pd.concat(kept_groups, axis=0)
    kept_keys = kept_group_df[["category", "source_problem_id"]].copy()
    kept_keys["__keep__"] = 1

    sampled_df = df.merge(kept_keys, on=["category", "source_problem_id"], how="inner")
    sampled_df = sampled_df.drop(columns="__keep__")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False)

    after = summarize_manifest(sampled_df)

    print("Before:")
    print(before.to_string())
    print()
    print("After:")
    print(after.to_string())
    print()
    print(
        {
            "input_rows": int(len(df)),
            "output_rows": int(len(sampled_df)),
            "input_completion_tokens": int(df["completion_token_count"].sum()),
            "output_completion_tokens": int(sampled_df["completion_token_count"].sum()),
            "input_total_tokens": int(df["token_count"].sum()),
            "output_total_tokens": int(sampled_df["token_count"].sum()),
            "seed": args.seed,
            "length_buckets": args.length_buckets,
            "plan": {
                category: {
                    "keep_fraction": value.keep_fraction,
                    "keep_problems": value.keep_problems,
                }
                for category, value in sorted(plan.items())
            },
        }
    )


if __name__ == "__main__":
    main()
