"""Spelling augmenter: break text into individual characters.

Output format: –s–e–x–v–e–x– (leading/trailing –, all chars merged across tokens)
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]

TOKENIZER_PATH = Path(__file__).parent.parent / "tokenizer.json"

EN_DASH = "\u2013"
LINES_PER_PROBLEM = 100


def load_tokens() -> tuple[list[str], list[str]]:
    """Load bare and space-prefixed lowercase alphabetic tokens from the tokenizer."""
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab = tok.get_vocab()

    bare: list[str] = []
    spaced: list[str] = []
    for token in vocab:
        if token.startswith("\u0120") and len(token) > 1:
            text = token[1:]
            if text.isascii() and text.isalpha() and text.islower() and 2 <= len(text) <= 8:
                spaced.append(text)
        elif (
            token.isascii()
            and token.isalpha()
            and token.islower()
            and 2 <= len(token) <= 8
        ):
            bare.append(token)

    return sorted(bare), sorted(spaced)


def spell_out(text: str) -> str:
    """Break text into individual characters, strip spaces, wrap with en-dashes."""
    chars = [c for c in text if c != " "]
    return EN_DASH + EN_DASH.join(chars) + EN_DASH


def generate() -> list[dict[str, str]]:
    """Generate spelling problems. Returns list of dicts with id, prompt, answer, category."""
    bare_tokens, spaced_tokens = load_tokens()
    print(f"[spelling] Loaded {len(bare_tokens)} bare tokens, {len(spaced_tokens)} spaced tokens")

    rng = random.Random(42)

    bare_shuffled = bare_tokens[:]
    rng.shuffle(bare_shuffled)
    spaced_shuffled = spaced_tokens[:]
    rng.shuffle(spaced_shuffled)

    # Each token appears 4 times: multiply pool sizes by 4
    n_problems = max(
        math.ceil(len(bare_shuffled) * 4 / LINES_PER_PROBLEM),
        math.ceil(len(spaced_shuffled) * 4 / (LINES_PER_PROBLEM * 2)),
    )
    print(f"[spelling] Generating {n_problems} problems")

    bare_idx = 0
    spaced_idx = 0
    problems = []

    for i in range(n_problems):
        # Generate demo inputs (shared between sample input and sample output)
        demo_inputs = []
        for _ in range(3):
            b = rng.choice(bare_tokens)
            s1, s2 = rng.choice(spaced_tokens), rng.choice(spaced_tokens)
            demo_inputs.append(f"{b} {s1} {s2}")

        sample_input_lines = [f"{j:02d} {inp}" for j, inp in enumerate(demo_inputs)]
        sample_output_lines = [f"{j:02d} {inp} -> {spell_out(inp)}" for j, inp in enumerate(demo_inputs)]

        test_inputs = []
        test_answers = []
        for row_num in range(LINES_PER_PROBLEM):
            b = bare_shuffled[bare_idx % len(bare_shuffled)]
            bare_idx += 1
            s1 = spaced_shuffled[spaced_idx % len(spaced_shuffled)]
            spaced_idx += 1
            s2 = spaced_shuffled[spaced_idx % len(spaced_shuffled)]
            spaced_idx += 1

            inp = f"{b} {s1} {s2}"
            test_inputs.append(f"{row_num:02d} {inp}")
            test_answers.append(f"{row_num:02d} {inp} -> {spell_out(inp)}")

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "This is a sample input.\n"
            + "\n".join(sample_input_lines)
            + "\n\n"
            + "This is a sample output.\n"
            + "\n".join(sample_output_lines)
            + "\n\n"
            + "This is your input.\n"
            + "\n".join(test_inputs)
        )

        answer = "\n".join(test_answers)
        pid = hashlib.sha256(f"spelling_{i}".encode()).hexdigest()[:8]
        problems.append({"id": pid, "prompt": prompt, "completion": answer, "category": "spelling"})

    print(f"[spelling] Generated {len(problems)} problems")
    return problems
