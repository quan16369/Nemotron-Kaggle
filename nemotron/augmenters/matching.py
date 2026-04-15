"""Bit column matching augmenter.

Extracts all operation sections (Identity, NOT, Constant, AND, OR, XOR,
AND-NOT, OR-NOT, XOR-NOT) from reasoning/*.txt files.

Input: output bit columns + operation bit columns (with match annotations)
Output: matching lines + left chain + right chain

Each problem has 1 demo example (with both x and y) and 1 test item.
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

REASONING_DIR = Path(__file__).parent.parent / "reasoning"

SECTION_NAMES = [
    "Identity",
    "NOT",
    "Constant",
    "AND",
    "OR",
    "XOR",
    "AND-NOT",
    "OR-NOT",
    "XOR-NOT",
]

# Map section name to Best-line prefix to strip
_BEST_PREFIX = {
    "Identity": "I",
    "NOT": "NOT",
    "Constant": "C",
    "AND": "AND",
    "OR": "OR",
    "XOR": "XOR",
    "AND-NOT": "AND-NOT",
    "OR-NOT": "OR-NOT",
    "XOR-NOT": "XOR-NOT",
}


def _extract_section_block(
    lines: list[str], start: int, end: int
) -> dict[str, object] | None:
    """Extract Matching output + Left + Right from lines[start:end]."""
    mo_idx = None
    for i in range(start, end):
        if lines[i].strip() == "Matching output":
            mo_idx = i
            break
    if mo_idx is None:
        return None

    mo_lines: list[str] = []
    for i in range(mo_idx + 1, end):
        stripped = lines[i].strip()
        if stripped == "" or stripped == "Left":
            break
        mo_lines.append(lines[i])

    left_idx = None
    for i in range(mo_idx + 1, end):
        if lines[i].strip() == "Left":
            left_idx = i
            break
    if left_idx is None:
        return None

    left_chain: list[str] = []
    best_left = ""
    for i in range(left_idx + 1, end):
        stripped = lines[i].strip()
        if stripped.startswith("Best:"):
            best_left = lines[i]
            break
        if stripped == "":
            break
        left_chain.append(lines[i])

    right_idx = None
    for i in range(left_idx + 1, end):
        if lines[i].strip() == "Right":
            right_idx = i
            break
    if right_idx is None:
        return None

    right_chain: list[str] = []
    best_right = ""
    for i in range(right_idx + 1, end):
        stripped = lines[i].strip()
        if stripped.startswith("Best:"):
            best_right = lines[i]
            break
        if stripped == "":
            break
        right_chain.append(lines[i])

    return {
        "mo_lines": mo_lines,
        "left_chain": left_chain,
        "best_left": best_left,
        "right_chain": right_chain,
        "best_right": best_right,
    }


def _extract_all_sections() -> list[dict[str, str]]:
    """Extract all operation sections from all reasoning files."""
    sections: list[dict[str, str]] = []

    for filepath in sorted(REASONING_DIR.glob("*.txt")):
        text = filepath.read_text()
        lines = text.split("\n")

        # Find "Output bit columns (with bitsum as hash)"
        obc_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "Output bit columns (with bitsum as hash)":
                obc_idx = i
                break
        if obc_idx is None:
            continue

        # Collect output bit column data lines (skip header)
        obc_block = lines[obc_idx + 1 : obc_idx + 9]

        # Find "Selecting" as the end boundary
        selecting_idx = len(lines)
        for i in range(obc_idx, len(lines)):
            if lines[i].strip() == "Selecting":
                selecting_idx = i
                break

        # Find each section header and its boundary
        sec_positions: list[tuple[str, int]] = []
        for i in range(obc_idx, selecting_idx):
            stripped = lines[i].strip()
            if stripped in SECTION_NAMES:
                sec_positions.append((stripped, i))

        for idx, (sec_name, sec_start) in enumerate(sec_positions):
            # Section ends at next section or Selecting
            if idx + 1 < len(sec_positions):
                sec_end = sec_positions[idx + 1][1]
            else:
                sec_end = selecting_idx

            # Collect data lines (between header and "Matching output")
            # Blank lines separate groups within a section — keep them
            data_lines: list[str] = []
            for i in range(sec_start + 1, sec_end):
                if lines[i].strip() == "Matching output":
                    break
                data_lines.append(lines[i])
            # Strip trailing blank lines
            while data_lines and data_lines[-1].strip() == "":
                data_lines.pop()

            # Extract matching/left/right block
            block = _extract_section_block(lines, sec_start, sec_end)
            if block is None:
                continue

            mo_lines = block["mo_lines"]
            left_chain = block["left_chain"]
            best_left = block["best_left"]
            right_chain = block["right_chain"]
            best_right = block["best_right"]

            # Check for x and y annotations in chains
            all_chain_text = " ".join(left_chain + right_chain)
            has_x = bool(re.search(r"\dx", all_chain_text))
            has_y = bool(re.search(r"\dy", all_chain_text))

            # Count match annotations in data lines
            n_matches = sum(1 for l in data_lines if "match" in l)

            # Check if all matching lines are "absent"
            all_absent = n_matches == 0

            # Check if both left and right chains are "none"
            both_none = (
                left_chain == ["none"] and right_chain == ["none"]
            )

            # Build input text
            input_text = "\n".join(obc_block) + "\n\n" + "\n".join(data_lines)

            # Strip section prefix from Best lines
            prefix = _BEST_PREFIX.get(sec_name, sec_name)
            best_left = re.sub(rf"^(Best: ){re.escape(prefix)}", r"\1", best_left)
            best_right = re.sub(rf"^(Best: ){re.escape(prefix)}", r"\1", best_right)

            # Build output text
            output_parts = (
                mo_lines
                + ["", "Left"]
                + left_chain
                + [best_left]
                + ["", "Right"]
                + right_chain
                + [best_right]
            )
            output_text = "\n".join(output_parts)

            sections.append({
                "file": filepath.name,
                "section": sec_name,
                "input": input_text,
                "output": output_text,
                "has_x": has_x,
                "has_y": has_y,
                "all_absent": all_absent,
                "both_none": both_none,
                "few_matches": n_matches < 4,
            })

    return sections


def generate() -> list[dict[str, str]]:
    """Generate bit column matching problems."""
    sections = _extract_all_sections()
    print(f"[matching] Extracted {len(sections)} sections")

    # Downsample sparse sections (keep 10%), deterministic
    n_all_absent = sum(1 for s in sections if s["all_absent"])
    n_both_none = sum(1 for s in sections if s["both_none"])
    n_few_matches = sum(1 for s in sections if s["few_matches"])

    def _keep(s: dict) -> bool:
        h = int(hashlib.sha256(f"{s['file']}_{s['section']}".encode()).hexdigest(), 16)
        if s["all_absent"]:
            return (h % 100) == 0
        if s["both_none"]:
            return (h % 10) == 0
        if s["few_matches"]:
            return (h % 5) < 1
        return True

    sections = [s for s in sections if _keep(s)]
    print(
        f"[matching] After downsampling all-absent ({n_all_absent}), "
        f"both-none ({n_both_none}), <4 matches ({n_few_matches}): "
        f"{len(sections)} sections"
    )

    problems: list[dict[str, str]] = []

    for i, item in enumerate(sections):
        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "x: not matched anywhere\n"
            "y: matched but wrong position\n\n"
            + item["input"]
        )

        pid = hashlib.sha256(f"matching_{i}".encode()).hexdigest()[:8]
        problems.append({
            "id": pid,
            "prompt": prompt,
            "completion": item["output"],
            "category": "matching",
        })

    print(f"[matching] Generated {len(problems)} problems")
    return problems
