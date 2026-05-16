"""Append criterion-negative rows to a high-scoring manifest.

This keeps the base manifest unchanged and appends only rows whose problem_id
contains "-neg-" from a separately exported augmented manifest.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


KEY_FIELDS = ("problem_id", "segment")


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return rows, list(reader.fieldnames)


def _key(row: dict[str, str]) -> tuple[str, str]:
    return tuple(row[field] for field in KEY_FIELDS)  # type: ignore[return-value]


def append_negative_rows(base_path: Path, negative_path: Path, output_path: Path) -> None:
    base_rows, fieldnames = _read_manifest(base_path)
    negative_rows, negative_fieldnames = _read_manifest(negative_path)
    if fieldnames != negative_fieldnames:
        raise ValueError("Manifest headers differ")

    seen_keys = {_key(row) for row in base_rows}
    extra_rows: list[dict[str, str]] = []
    skipped_duplicates = 0
    for row in negative_rows:
        if "-neg-" not in row["problem_id"]:
            continue
        key = _key(row)
        if key in seen_keys:
            skipped_duplicates += 1
            continue
        seen_keys.add(key)
        extra_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(base_rows)
        writer.writerows(extra_rows)

    base_counts = Counter(row["category"] for row in base_rows)
    extra_counts = Counter(row["category"] for row in extra_rows)
    print("Base manifest:", base_path)
    print("Negative source:", negative_path)
    print("Output:", output_path)
    print("Base rows:", len(base_rows))
    print("Appended negative rows:", len(extra_rows))
    print("Skipped duplicate negative rows:", skipped_duplicates)
    print("Final rows:", len(base_rows) + len(extra_rows))
    print("Base categories:", dict(sorted(base_counts.items())))
    print("Extra categories:", dict(sorted(extra_counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("winning_snapshot_delta_manifest (1).csv"),
        help="High-scoring base manifest to keep unchanged.",
    )
    parser.add_argument(
        "--negative",
        type=Path,
        required=True,
        help="Manifest exported with --augment-negative-criteria.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("winning_snapshot_delta_manifest_plus_neg.csv"),
        help="Output manifest path.",
    )
    args = parser.parse_args()
    append_negative_rows(args.base, args.negative, args.output)


if __name__ == "__main__":
    main()
