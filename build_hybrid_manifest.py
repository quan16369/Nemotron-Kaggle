"""Build a hybrid training manifest from a high-scoring base and patched traces.

Default policy is conservative:
- use the high-scoring base manifest for most rows;
- replace only long cipher completions with the patched compact-candidate version;
- optionally replace very long gravity/unit_conversion rows if requested.

Both manifests must have identical (problem_id, segment) keys.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
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


def _int_field(row: dict[str, str], field: str) -> int:
    return int(float(row[field]))


def _should_replace(
    base_row: dict[str, str],
    patched_row: dict[str, str],
    *,
    cipher_threshold: int,
    math_threshold: int | None,
    max_patched_completion: int | None,
    patch_math: bool,
) -> tuple[bool, str]:
    category = base_row["category"]
    base_completion = _int_field(base_row, "completion_token_count")
    patched_completion = _int_field(patched_row, "completion_token_count")

    if max_patched_completion is not None and patched_completion > max_patched_completion:
        return False, ""

    if category == "cipher" and base_completion >= cipher_threshold:
        return True, f"cipher>={cipher_threshold}"

    if (
        patch_math
        and math_threshold is not None
        and category in {"gravity", "unit_conversion"}
        and base_completion >= math_threshold
    ):
        return True, f"{category}>={math_threshold}"

    return False, ""


def build_hybrid(
    base_path: Path,
    patched_path: Path,
    output_path: Path,
    *,
    cipher_threshold: int,
    math_threshold: int | None,
    max_patched_completion: int | None,
    patch_math: bool,
) -> None:
    base_rows, fieldnames = _read_manifest(base_path)
    patched_rows, patched_fieldnames = _read_manifest(patched_path)
    if fieldnames != patched_fieldnames:
        raise ValueError("Manifest headers differ")

    patched_by_key = {_key(row): row for row in patched_rows}
    if len(patched_by_key) != len(patched_rows):
        raise ValueError(f"{patched_path} has duplicate keys")

    base_keys = {_key(row) for row in base_rows}
    patched_keys = set(patched_by_key)
    if base_keys != patched_keys:
        missing = sorted(base_keys - patched_keys)[:10]
        extra = sorted(patched_keys - base_keys)[:10]
        raise ValueError(f"Manifest keys differ. missing={missing}, extra={extra}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows": len(base_rows),
        "replaced": 0,
        "base_completion_tokens": 0,
        "patched_selected_completion_tokens": 0,
        "hybrid_completion_tokens": 0,
        "base_total_tokens": 0,
        "hybrid_total_tokens": 0,
    }
    replaced_by_reason: Counter[str] = Counter()
    replaced_by_category: Counter[str] = Counter()
    token_delta_by_category: defaultdict[str, int] = defaultdict(int)
    examples: list[tuple[int, str, str, str, int, int]] = []

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for base_row in base_rows:
            patched_row = patched_by_key[_key(base_row)]
            replace, reason = _should_replace(
                base_row,
                patched_row,
                cipher_threshold=cipher_threshold,
                math_threshold=math_threshold,
                max_patched_completion=max_patched_completion,
                patch_math=patch_math,
            )
            out_row = patched_row if replace else base_row
            writer.writerow(out_row)

            category = base_row["category"]
            base_comp = _int_field(base_row, "completion_token_count")
            patched_comp = _int_field(patched_row, "completion_token_count")
            out_comp = _int_field(out_row, "completion_token_count")
            base_total = _int_field(base_row, "token_count")
            out_total = _int_field(out_row, "token_count")

            stats["base_completion_tokens"] += base_comp
            stats["patched_selected_completion_tokens"] += patched_comp if replace else 0
            stats["hybrid_completion_tokens"] += out_comp
            stats["base_total_tokens"] += base_total
            stats["hybrid_total_tokens"] += out_total

            if replace:
                stats["replaced"] += 1
                replaced_by_reason[reason] += 1
                replaced_by_category[category] += 1
                delta = out_comp - base_comp
                token_delta_by_category[category] += delta
                examples.append(
                    (
                        abs(delta),
                        base_row["problem_id"],
                        base_row["segment"],
                        category,
                        base_comp,
                        out_comp,
                    )
                )

    print("Hybrid manifest written:", output_path)
    print("Rows:", stats["rows"])
    print("Replaced rows:", stats["replaced"])
    print("Replaced by reason:", dict(replaced_by_reason))
    print("Replaced by category:", dict(replaced_by_category))
    print("Completion tokens:")
    print("  base  :", stats["base_completion_tokens"])
    print("  hybrid:", stats["hybrid_completion_tokens"])
    print("  delta :", stats["hybrid_completion_tokens"] - stats["base_completion_tokens"])
    print("Total tokens:")
    print("  base  :", stats["base_total_tokens"])
    print("  hybrid:", stats["hybrid_total_tokens"])
    print("  delta :", stats["hybrid_total_tokens"] - stats["base_total_tokens"])
    print("Token delta by category:", dict(sorted(token_delta_by_category.items())))
    if examples:
        print("Largest changed rows:")
        for _, problem_id, segment, category, base_comp, out_comp in sorted(
            examples, reverse=True
        )[:20]:
            print(
                f"  {problem_id} {segment} {category}: "
                f"{base_comp} -> {out_comp} ({out_comp - base_comp:+d})"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("winning_snapshot_delta_manifest (1).csv"),
        help="High-scoring base manifest.",
    )
    parser.add_argument(
        "--patched",
        type=Path,
        default=Path("winning_snapshot_delta_manifest.csv"),
        help="Patched/compact manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("winning_snapshot_delta_manifest_hybrid.csv"),
    )
    parser.add_argument(
        "--compact-cipher-token-threshold",
        type=int,
        default=6500,
        help="Replace base cipher rows at or above this completion token count.",
    )
    parser.add_argument(
        "--patch-math-token-threshold",
        type=int,
        default=7400,
        help="Optional threshold for gravity/unit_conversion replacement.",
    )
    parser.add_argument(
        "--patch-math",
        action="store_true",
        help="Also replace very long gravity/unit_conversion rows.",
    )
    parser.add_argument(
        "--max-patched-completion",
        type=int,
        default=7680,
        help="Do not select patched rows above this completion token count.",
    )
    args = parser.parse_args()

    build_hybrid(
        args.base,
        args.patched,
        args.output,
        cipher_threshold=args.compact_cipher_token_threshold,
        math_threshold=args.patch_math_token_threshold,
        max_patched_completion=args.max_patched_completion,
        patch_math=args.patch_math,
    )


if __name__ == "__main__":
    main()
