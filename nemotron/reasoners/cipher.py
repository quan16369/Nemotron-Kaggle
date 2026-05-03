"""Cipher: substitution cipher reasoning generator."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from reasoners.store_types import Problem

_WONDERLAND_PATH = Path(__file__).parent / "wonderland.txt"


@lru_cache(maxsize=1)
def _load_wonderland() -> list[str]:
    """Load the Wonderland word list (sorted)."""
    with _WONDERLAND_PATH.open() as f:
        words = [line.strip() for line in f if line.strip()]
    return sorted(words)


def _word_pattern(word: str) -> tuple[int, ...]:
    seen: dict[str, int] = {}
    pattern: list[int] = []
    for char in word:
        if char not in seen:
            seen[char] = len(seen)
        pattern.append(seen[char])
    return tuple(pattern)


def _candidate_words_for_partial(
    partial: str,
    cipher_to_plain: dict[str, str],
    plain_to_cipher: dict[str, str],
    cipher_word: str,
) -> list[str]:
    """Find wonderland words matching a partial decryption with unknowns.

    Candidates must be consistent with cipher_to_plain (bijective mapping).
    """
    candidates: list[str] = []
    target_len = len(partial)
    target_pattern = _word_pattern(cipher_word)

    for word in _load_wonderland():
        if len(word) != target_len:
            continue
        if _word_pattern(word) != target_pattern:
            continue
        match = True
        for i, ch in enumerate(partial):
            if ch != "?" and ch != word[i]:
                match = False
                break
        if not match:
            continue
        # Check bijective consistency in both directions.
        consistent = True
        for cc, wc in zip(cipher_word, word):
            if cc in cipher_to_plain and cipher_to_plain[cc] != wc:
                consistent = False
                break
            if wc in plain_to_cipher and plain_to_cipher[wc] != cc:
                consistent = False
                break
        if consistent:
            candidates.append(word)

    candidates.sort()
    return candidates


def _decode_word(word: str, cipher_to_plain: dict[str, str]) -> str | None:
    decoded: list[str] = []
    for ch in word:
        if ch not in cipher_to_plain:
            return None
        decoded.append(cipher_to_plain[ch])
    return "".join(decoded)


def _is_bijective_mapping(cipher_to_plain: dict[str, str]) -> bool:
    return len(set(cipher_to_plain.values())) == len(cipher_to_plain)


def reasoning_cipher(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append("Solve the substitution cipher by direct rule lookup.")
    lines.append("[Rules from examples]")

    wonderland_words = _load_wonderland()
    wonderland_set = set(wonderland_words)

    cipher_to_plain: dict[str, str] = {}
    plain_to_cipher: dict[str, str] = {}
    for ex_i, ex in enumerate(problem.examples, start=1):
        cipher_words = str(ex.input_value).split()
        plain_words = str(ex.output_value).split()
        if len(cipher_words) != len(plain_words):
            continue

        lines.append(f"[Example_{ex_i}] {ex.input_value} -> {ex.output_value}")
        for wi, (cw, pw) in enumerate(zip(cipher_words, plain_words)):
            if len(cw) != len(pw):
                continue
            word_mappings: list[str] = []
            for cc, pc in zip(cw, pw):
                if cc in cipher_to_plain and cipher_to_plain[cc] != pc:
                    return None
                if pc in plain_to_cipher and plain_to_cipher[pc] != cc:
                    return None
                cipher_to_plain[cc] = pc
                plain_to_cipher[pc] = cc
                word_mappings.append(f"{cc}->{pc}")
            lines.append(
                f"[Example_{ex_i}_Word_{wi + 1}] "
                f"{cw} -> {pw}: {' '.join(word_mappings)}"
            )

    if not _is_bijective_mapping(cipher_to_plain):
        return None

    lines.append("")
    rule_table = " ".join(f"{c}->{cipher_to_plain[c]}" for c in sorted(cipher_to_plain))
    lines.append(f"[Rule_Table] {rule_table}")

    lines.append("")
    question_words = problem.question.split()
    decoded_words: list[str] = []
    lines.append(f"[Question] {problem.question}")
    lines.append("[Decode question]")
    for wi, cw in enumerate(question_words, start=1):
        partial = "".join(cipher_to_plain.get(cc, "?") for cc in cw)
        if "?" in partial:
            candidates = _candidate_words_for_partial(
                partial, cipher_to_plain, plain_to_cipher, cw
            )
            if not candidates:
                return None
            chosen = candidates[0]
            pending: dict[str, str] = {}
            for cc, pc in zip(cw, chosen):
                if cc in cipher_to_plain and cipher_to_plain[cc] != pc:
                    return None
                if pc in plain_to_cipher and plain_to_cipher[pc] != cc:
                    return None
                if cc in pending and pending[cc] != pc:
                    return None
                pending[cc] = pc
            for cc, pc in pending.items():
                if cc not in cipher_to_plain:
                    cipher_to_plain[cc] = pc
                    plain_to_cipher[pc] = cc
            if not _is_bijective_mapping(cipher_to_plain):
                return None
            decoded = chosen
        else:
            decoded = partial

        if decoded not in wonderland_set:
            return None
        steps = " ".join(f"{cc}->{cipher_to_plain[cc]}" for cc in cw)
        lines.append(f"[Word_{wi}] {cw}: {steps} = {decoded}")
        decoded_words.append(decoded)

    final_decoded_words: list[str] = []
    lines.append("")
    lines.append("[Double_Check]")
    for cw in question_words:
        decoded = _decode_word(cw, cipher_to_plain)
        if decoded is None:
            return None
        if decoded not in wonderland_set:
            return None
        final_decoded_words.append(decoded)
        lines.append(f"{cw} -> {decoded}")

    if len(final_decoded_words) != len(question_words):
        return None
    if any(w == "" for w in final_decoded_words):
        return None

    computed = " ".join(final_decoded_words)
    lines.append(f"[Final] {computed}")

    lines.append("")
    lines.append(f"The answer is \\boxed{{{computed}}}")
    return "\n".join(lines)
