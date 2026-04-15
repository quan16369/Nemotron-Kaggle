"""Create synthetic training corpus with reasoning from reasoning/*.txt files.

The completion for each entry is:
    (reasoning text)</think>\\boxed{(answer)}<|im_end|>

The opening <think>\\n is already part of the prompt (from the chat template),
so the reasoning text flows directly after it.

Outputs:
- corpus.jsonl          - Index with metadata per entry
- corpus/<problem_id>/synthetic.jsonl  - Segment files with interleaved masked/unmasked

Usage:
    uv run corpus.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]
from transformers import AutoTokenizer  # type: ignore[import-untyped]

TRAIN_CSV = Path(__file__).parent / "train.csv"
AUGMENTATIONS_DIR = Path(__file__).parent / "augmentations"
PROBLEMS_INDEX = Path(__file__).parent / "problems.jsonl"
REASONING_DIR = Path(__file__).parent / "reasoning"
CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_INDEX = Path(__file__).parent / "corpus.jsonl"
TOKENIZER_PATH = Path(__file__).parent / "tokenizer.json"
PRIORITY_IDS_PATH = Path(__file__).parent / "investigators" / "priority_problem_ids.txt"

# Must match metric_reference.py / query.py
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

TOKEN_LIMIT = 8192

# Downsample rate per category (fraction to KEEP). Categories not listed are kept at 100%.
DOWNSAMPLE_RATES: dict[str, float] = {
    "numeral": 0.4,
    "gravity": 0.6,
    "unit_conversion": 0.6,
}

# Truncate completion tokens to a random range (min, max) for specified categories.
# Instead of dropping entries, we keep them all but shorten the completion.
TRUNCATE_COMPLETION: dict[str, tuple[int, int]] = {}

# Duplicate rate per category (number of COPIES to add). Categories not listed get 0 extra copies.
DUPLICATE_COUNTS: dict[str, int] = {
    "cryptarithm_deduce": 1,
}


def _keep_by_hash(problem_id: str, rate: float) -> bool:
    """Deterministic downsample: keep entry if hash falls within rate."""
    # rate is considered to introduce some shuffle as we downsample differently
    h = int(hashlib.sha256((problem_id + str(rate)).encode()).hexdigest(), 16) + int(
        10000 * rate
    )
    return (h % 10000) < int(rate * 10000)


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def tokenize_prompt(
    prompt_text: str,
    chat_tokenizer: AutoTokenizer,
    *,
    suffix: str = PROMPT_SUFFIX,
) -> list[int]:
    """Tokenize a problem prompt using the chat template, matching query.py."""
    messages = [{"role": "user", "content": prompt_text + suffix}]
    return chat_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )


@dataclass
class CorpusEntry:
    problem_id: str
    category: str
    tokens: list[int]
    mask: list[int]
    masked_token_count: int
    unmasked_token_count: int
    answer: str
    included: bool = False

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def to_index_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "segment": "synthetic.jsonl",
            "category": self.category,
            "masked_token_count": self.masked_token_count,
            "unmasked_token_count": self.unmasked_token_count,
            "token_count": self.token_count,
            "answer": self.answer,
            "included": self.included,
        }


def choose_entry_to_include(problem_status: str, category: str) -> bool:
    """Include if the problem's rule was found, or if category is a _guess type."""
    if category.endswith("_guess"):
        return True
    return problem_status == "rule_found"


def build_segments(
    tokens: list[int],
    mask: list[int],
) -> list[dict]:
    """Build segment list from tokens and mask."""
    if not tokens:
        return []

    segments: list[dict] = []
    seg_start = 0
    current_type = "unmasked" if mask[0] == 1 else "masked"

    for i in range(1, len(tokens)):
        token_type = "unmasked" if mask[i] == 1 else "masked"
        if token_type != current_type:
            segments.append(
                {
                    "type": current_type,
                    "pos": seg_start,
                    "tokens": tokens[seg_start:i],
                }
            )
            seg_start = i
            current_type = token_type

    segments.append(
        {
            "type": current_type,
            "pos": seg_start,
            "tokens": tokens[seg_start:],
        }
    )

    return segments


def main() -> None:
    if not PROBLEMS_INDEX.exists():
        print(f"No {PROBLEMS_INDEX} found. Run problems.py first.")
        return

    # Load tokenizers
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    chat_tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True
    )

    # Load problem prompts from train.csv
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    with open(TRAIN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["id"]
            prompts[pid] = row["prompt"]
            answers[pid] = row["answer"]

    # Load problem categories and statuses
    problem_cats: dict[str, str] = {}
    problem_statuses: dict[str, str] = {}
    for prob_raw in load_jsonl(PROBLEMS_INDEX):
        problem_cats[prob_raw["id"]] = prob_raw["category"]
        problem_statuses[prob_raw["id"]] = prob_raw["status"]

    # Load priority IDs (never downsampled)
    priority_ids: set[str] = set()
    if PRIORITY_IDS_PATH.exists():
        priority_ids = {
            line.strip()
            for line in PRIORITY_IDS_PATH.read_text().splitlines()
            if line.strip()
        }

    # Clean and recreate corpus directory
    if CORPUS_DIR.exists():
        shutil.rmtree(CORPUS_DIR)
    CORPUS_DIR.mkdir(parents=True)

    entries: list[CorpusEntry] = []

    # Iterate over problems that have reasoning files
    problem_ids = sorted(
        pid
        for pid in problem_cats
        if (REASONING_DIR / f"{pid}.txt").exists() and pid in prompts
    )

    for problem_id in problem_ids:
        category = problem_cats[problem_id]
        answer = answers[problem_id]

        reasoning_text = (REASONING_DIR / f"{problem_id}.txt").read_text().rstrip("\n")

        # Extract answer from reasoning's \boxed{} so they match
        boxed_match = re.findall(r"\\boxed\{([^}]*)\}", reasoning_text)
        reasoning_answer = boxed_match[-1] if boxed_match else answer
        completion_text = (
            f"{reasoning_text}\n</think>\n\\boxed{{{reasoning_answer}}}<|im_end|>"
        )
        completion_ids = tokenizer.encode(completion_text, add_special_tokens=False).ids

        # Tokenize prompt directly (no raw/ dependency)
        prompt_ids = tokenize_prompt(prompts[problem_id], chat_tokenizer)

        all_tokens = prompt_ids + completion_ids
        mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

        # Truncate to token limit
        if len(all_tokens) > TOKEN_LIMIT:
            all_tokens = all_tokens[:TOKEN_LIMIT]
            mask = mask[:TOKEN_LIMIT]

        unmasked_count = sum(mask)
        masked_count = len(mask) - unmasked_count

        entry = CorpusEntry(
            problem_id=problem_id,
            category=category,
            tokens=all_tokens,
            mask=mask,
            masked_token_count=masked_count,
            unmasked_token_count=unmasked_count,
            answer=answer,
        )
        entry.included = choose_entry_to_include(problem_statuses[problem_id], category)

        if not entry.included:
            continue

        # Deterministic downsampling by category (skip for priority IDs)
        rate = DOWNSAMPLE_RATES.get(category, 1.0)
        if (
            rate < 1.0
            and problem_id not in priority_ids
            and not _keep_by_hash(problem_id, rate)
        ):
            # Truncate completion instead of dropping entirely
            if category in TRUNCATE_COMPLETION:
                lo, hi = TRUNCATE_COMPLETION[category]
                rng = random.Random(problem_id)
                max_completion = rng.randint(lo, hi)
                prompt_len = len(prompt_ids)
                keep_len = min(prompt_len + max_completion, len(all_tokens))
                entry.tokens = all_tokens[:keep_len]
                entry.mask = mask[:keep_len]
                entry.unmasked_token_count = sum(entry.mask)
                entry.masked_token_count = len(entry.mask) - entry.unmasked_token_count
            else:
                continue

        # Build interleaved segments and write segment file
        segments = build_segments(all_tokens, mask)

        problem_dir = CORPUS_DIR / problem_id
        problem_dir.mkdir(parents=True, exist_ok=True)
        seg_path = problem_dir / "synthetic.jsonl"

        with open(seg_path, "w") as f:
            for seg in segments:
                json.dump(seg, f)
                f.write("\n")

        entries.append(entry)

    # Process augmentations/*.txt (no reasoning, no \boxed{})
    if AUGMENTATIONS_DIR.exists():
        for aug_path in sorted(AUGMENTATIONS_DIR.glob("*.txt")):
            text = aug_path.read_text()
            # Parse [category], [prompt], and [completion] sections
            category = text.split("[category]\n", 1)[1].split("\n[prompt]\n", 1)[0]
            prompt_text = text.split("[prompt]\n", 1)[1].split("\n[completion]\n", 1)[0]
            completion = text.split("\n[completion]\n", 1)[1].rstrip("\n")

            problem_id = aug_path.stem

            completion_text = f"{completion}\n</think><|im_end|>"
            completion_ids = tokenizer.encode(
                completion_text, add_special_tokens=False
            ).ids

            prompt_ids = tokenize_prompt(prompt_text, chat_tokenizer, suffix="")

            all_tokens = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

            assert len(all_tokens) <= TOKEN_LIMIT, (
                f"augmented entry {problem_id} exceeds token limit: "
                f"{len(all_tokens)} > {TOKEN_LIMIT}"
            )

            unmasked_count = sum(mask)
            masked_count = len(mask) - unmasked_count

            entry = CorpusEntry(
                problem_id=problem_id,
                category=category,
                tokens=all_tokens,
                mask=mask,
                masked_token_count=masked_count,
                unmasked_token_count=unmasked_count,
                answer=completion,
                included=True,
            )

            segments = build_segments(all_tokens, mask)
            problem_dir = CORPUS_DIR / problem_id
            problem_dir.mkdir(parents=True, exist_ok=True)
            with open(problem_dir / "synthetic.jsonl", "w") as sf:
                for seg in segments:
                    json.dump(seg, sf)
                    sf.write("\n")

            entries.append(entry)

    # Collect all duplicates: priority copies + category copies
    def _duplicate_entry(e: CorpusEntry, suffix: str) -> CorpusEntry:
        dup = CorpusEntry(
            problem_id=f"{e.problem_id}{suffix}",
            category=e.category,
            tokens=e.tokens,
            mask=e.mask,
            masked_token_count=e.masked_token_count,
            unmasked_token_count=e.unmasked_token_count,
            answer=e.answer,
            included=e.included,
        )
        dup_dir = CORPUS_DIR / dup.problem_id
        dup_dir.mkdir(parents=True, exist_ok=True)
        segments = build_segments(dup.tokens, dup.mask)
        with open(dup_dir / "synthetic.jsonl", "w") as f:
            for seg in segments:
                json.dump(seg, f)
                f.write("\n")
        return dup

    # Priority problems get one extra copy
    priority_dups = [
        _duplicate_entry(e, "-p0") for e in entries if e.problem_id in priority_ids
    ]
    if priority_dups:
        print(f"Priority duplicated: {len(priority_dups)} extra entries added")
        entries.extend(priority_dups)

    # Category duplicates (skip priority copies to avoid compounding)
    cat_dups = [
        _duplicate_entry(e, f"-d{i}")
        for e in entries
        if "-p0" not in e.problem_id
        for i in range(DUPLICATE_COUNTS.get(e.category, 0))
    ]
    if cat_dups:
        print(f"Duplicated: {len(cat_dups)} extra entries added")
        entries.extend(cat_dups)

    entries.sort(key=lambda e: e.problem_id)

    # Write index JSONL
    with open(CORPUS_INDEX, "w") as f:
        for e in entries:
            json.dump(e.to_index_dict(), f)
            f.write("\n")

    # Stats
    cat_counts: dict[str, int] = {cat: 0 for cat in {e.category for e in entries}}
    cat_tokens: dict[str, int] = {cat: 0 for cat in cat_counts}
    for e in entries:
        cat_counts[e.category] += 1
        cat_tokens[e.category] += e.unmasked_token_count

    total_unmasked = sum(e.unmasked_token_count for e in entries)
    total_masked = sum(e.masked_token_count for e in entries)
    max_tokens = max((e.token_count for e in entries), default=0)

    print(f"Corpus (synthetic): {len(entries)} entries")
    print(f"Unmasked tokens: {total_unmasked:,}")
    print(f"Masked tokens:   {total_masked:,}")
    print(f"Max seq length:  {max_tokens:,}")
    print()
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]} runs, {cat_tokens[cat]:,} unmasked tokens")


if __name__ == "__main__":
    main()
