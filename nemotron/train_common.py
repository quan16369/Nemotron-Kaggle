"""Shared utilities for training scripts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import tinker

logger = logging.getLogger(__name__)

Category = Literal[
    "bit_manipulation",
    "cipher",
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "equation_numeric_deduce",
    "equation_numeric_guess",
    "gravity",
    "numeral",
    "unit_conversion",
]

CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_INDEX = Path(__file__).parent / "corpus.jsonl"


class CorpusEntry(TypedDict):
    """Entry from corpus.jsonl."""

    problem_id: str
    segment: str
    category: Category
    masked_token_count: int
    unmasked_token_count: int
    token_count: int
    answer: str
    included: bool


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_corpus_entries() -> list[CorpusEntry]:
    """Load corpus.jsonl and return typed entries."""
    return cast(list[CorpusEntry], load_jsonl(CORPUS_INDEX))


@dataclass
class TrainingExample:
    """Represents a single training example with pre-tokenized data."""

    problem_id: str
    segment: str
    category: Category
    masked_token_count: int
    unmasked_token_count: int

    @classmethod
    def from_dict(cls, entry: CorpusEntry) -> TrainingExample:
        return cls(
            problem_id=entry["problem_id"],
            segment=entry["segment"],
            category=entry["category"],
            masked_token_count=entry["masked_token_count"],
            unmasked_token_count=entry["unmasked_token_count"],
        )

    def get_segment_path(self) -> Path:
        """Get path to the corpus segment file."""
        return CORPUS_DIR / self.problem_id / self.segment

    def load_tokens(self) -> tuple[list[int], list[int]]:
        """Load tokens and mask from segment file.

        Returns (tokens, mask) where mask[i]=1 means unmasked (train on this token).
        """
        segments = load_jsonl(self.get_segment_path())
        tokens: list[int] = []
        mask: list[int] = []
        for seg in segments:
            seg_tokens = seg["tokens"]
            tokens.extend(seg_tokens)
            mask_val = 1 if seg["type"] == "unmasked" else 0
            mask.extend([mask_val] * len(seg_tokens))
        return tokens, mask


def build_datum(
    tokens: list[int],
    mask: list[int],
    max_length: int = 8192,
) -> tinker.Datum | None:
    """Build a Tinker Datum from tokens and mask (0=masked, 1=unmasked)."""
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        mask = mask[:max_length]

    if not any(mask):
        return None

    model_input = tinker.ModelInput(
        chunks=[tinker.types.EncodedTextChunk(tokens=tokens[:-1])]
    )
    target_tokens = tokens[1:]
    # Weight for predicting token[i+1]: use mask[i+1] (train on unmasked targets)
    weights = [float(m) for m in mask[1:]]

    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights,
                dtype="float32",
                shape=[len(weights)],
            ),
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
        },
    )
