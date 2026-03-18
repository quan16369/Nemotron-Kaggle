from __future__ import annotations

import json
import random
from typing import Dict, Any, List

from src.specs.registry import SPEC_REGISTRY


def make_boxed(answer: str) -> str:
    return f"\\boxed{{{answer}}}"


def choose_style() -> str:
    r = random.random()
    if r < 0.70:
        return "concise"
    if r < 0.90:
        return "ultra_short"
    return "answer_only"


def make_sft_record(spec_name: str, config: Dict[str, Any], style: str = "concise") -> Dict[str, Any]:
    spec = SPEC_REGISTRY[spec_name]
    prompt = spec.render(config)
    answer = spec.solve(config)

    if style == "concise":
        completion = (
            "The examples follow one consistent transformation rule. "
            "Applying that same rule to the final input gives the result below.\n\n"
            f"{make_boxed(answer)}"
        )
    elif style == "ultra_short":
        completion = f"Apply the same rule.\n\n{make_boxed(answer)}"
    elif style == "answer_only":
        completion = make_boxed(answer)
    else:
        raise ValueError(f"Unknown style: {style}")

    return {
        "prompt": prompt,
        "answer": answer,
        "completion": completion,
        "meta": {
            "family": spec_name,
            "config": config,
            "canonical_config": spec.canonicalize(config),
            "verifier_pass": spec.verify(config, answer),
        },
    }


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def merge_jsonl(input_paths: List[str], out_path: str, shuffle: bool = True, seed: int = 42) -> None:
    all_records = []
    for p in input_paths:
        all_records.extend(read_jsonl(p))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(all_records)
    write_jsonl(all_records, out_path)


def to_chatml_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = (
        rec["prompt"]
        + "\n\nInfer the hidden rule from the examples."
        + "\nReason concisely."
        + "\nGive exactly one final answer in \\boxed{}."
    )
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": rec["completion"]},
        ],
        "meta": rec.get("meta", {}),
    }


def export_chatml(input_jsonl: str, out_jsonl: str) -> None:
    records = read_jsonl(input_jsonl)
    converted = [to_chatml_record(r) for r in records]
    write_jsonl(converted, out_jsonl)