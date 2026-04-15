"""Generate all augmented training data.

Imports from augmenters/ modules and writes individual files to augmentations/.

Each file: augmentations/(problem_id).txt with format:
    [prompt]
    ...
    [completion]
    ...

Usage: uv run python3 augmentation.py
"""

from __future__ import annotations

from pathlib import Path

from augmenters import concatenation, lstrip, matching, splitting, spelling

OUTPUT_DIR = Path(__file__).parent / "augmentations"


def main() -> None:
    problems: list[dict[str, str]] = []

    problems.extend(spelling.generate())
    problems.extend(concatenation.generate())
    problems.extend(splitting.generate())
    problems.extend(matching.generate())
    problems.extend(lstrip.generate())

    if OUTPUT_DIR.exists():
        import shutil

        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    for p in problems:
        path = OUTPUT_DIR / f"{p['id']}.txt"
        path.write_text(
            f"[category]\n{p['category']}\n[prompt]\n{p['prompt']}\n[completion]\n{p['completion']}\n"
        )

    # Summary
    cats: dict[str, int] = {}
    for p in problems:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print(f"\nWrote {len(problems)} problems to {OUTPUT_DIR}/")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
