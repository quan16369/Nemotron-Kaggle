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
    lines.append(
        "We need to find the encryption mapping from the examples. "
        "It looks like a substitution cipher."
    )
    lines.append("I will compute the answer and state it at the end.")

    wonderland_words = _load_wonderland()
    wonderland_set = set(wonderland_words)

    cipher_to_plain: dict[str, str] = {}
    word_alignment_lines: list[str] = []
    for ex in problem.examples:
        cipher_words = str(ex.input_value).split()
        plain_words = str(ex.output_value).split()
        if len(cipher_words) != len(plain_words):
            return None

        word_alignment_lines.append(f"{ex.input_value} -> {ex.output_value}")
        for cw, pw in zip(cipher_words, plain_words):
            if len(cw) != len(pw):
                return None
            for cc, pc in zip(cw, pw):
                if cc in cipher_to_plain and cipher_to_plain[cc] != pc:
                    return None
                cipher_to_plain[cc] = pc

    if not _is_bijective_mapping(cipher_to_plain):
        return None

    lines.append("")
    lines.append("Example alignments:")
    lines.extend(word_alignment_lines)

    mapping_pairs = [f"{c}->{cipher_to_plain[c]}" for c in sorted(cipher_to_plain)]
    lines.append("")
    lines.append("Known character mapping:")
    for i in range(0, len(mapping_pairs), 9):
        lines.append(", ".join(mapping_pairs[i : i + 9]))

    question_words = problem.question.split()
    answer_words = problem.answer.split()
    if answer_words and len(answer_words) != len(question_words):
        answer_words = []

    lines.append("")
    lines.append("Apply the mapping to the query words in order:")
    plain_to_cipher: dict[str, str] = {v: k for k, v in cipher_to_plain.items()}
    decoded_words: list[str] = []

    for idx, cw in enumerate(question_words, start=1):
        partial = "".join(cipher_to_plain.get(cc, "?") for cc in cw)
        if "?" not in partial:
            if partial not in wonderland_set:
                return None
            decoded_words.append(partial)
            lines.append(f"{idx}. {cw} -> {partial}")
            continue

        candidates = _candidate_words_for_partial(
            partial,
            cipher_to_plain,
            plain_to_cipher,
            cw,
        )
        if not candidates:
            return None

        chosen = candidates[0]
        if answer_words and answer_words[idx - 1] in candidates:
            chosen = answer_words[idx - 1]

        display_candidates = ", ".join(candidates[:5])
        if len(candidates) > 5:
            display_candidates += f", ... ({len(candidates)} total)"
        lines.append(
            f"{idx}. {cw} -> {partial}; candidates: {display_candidates}; choose {chosen}"
        )
        decoded_words.append(chosen)

        new_mappings: list[str] = []
        for cc, pc in zip(cw, chosen):
            if cc in cipher_to_plain:
                if cipher_to_plain[cc] != pc:
                    return None
                continue
            if pc in plain_to_cipher and plain_to_cipher[pc] != cc:
                return None
            cipher_to_plain[cc] = pc
            plain_to_cipher[pc] = cc
            new_mappings.append(f"{cc}->{pc}")

        if not _is_bijective_mapping(cipher_to_plain):
            return None
        if new_mappings:
            lines.append(f"   New mapping: {', '.join(new_mappings)}")

    lines.append("")
    lines.append("Decoded words in original query order:")
    final_decoded_words: list[str] = []
    for idx, cw in enumerate(question_words, start=1):
        decoded = _decode_word(cw, cipher_to_plain)
        if decoded is None:
            return None
        if decoded not in wonderland_set:
            return None
        final_decoded_words.append(decoded)
        lines.append(f"{idx}. {cw} -> {decoded}")

    if len(final_decoded_words) != len(question_words):
        return None

    computed = " ".join(final_decoded_words)
    lines.append(f"Decoded sentence: {computed}")
    lines.append("The word order is the same as the query word order.")

    lines.append("")
    lines.append("I will now state the final answer.")
    lines.append(f"Final answer is: {computed}")
    return "\n".join(lines)
