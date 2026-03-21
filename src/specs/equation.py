from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from .base import PuzzleSpec


SYMBOL_ALPHABET = list("xyzabc123+-=*/()[]{}@#&!?|^%")
TARGET_ALPHABET = list("@#$&!?|^%/\\~;:<>[]{}0123456789")


def random_equation_string() -> str:
    length = random.randint(3, 8)
    return "".join(random.choice(SYMBOL_ALPHABET) for _ in range(length))


def build_mapping(mapping_size: int = 12) -> Dict[str, str]:
    src_pool = SYMBOL_ALPHABET[:]
    tgt_pool = TARGET_ALPHABET[:]
    random.shuffle(src_pool)
    random.shuffle(tgt_pool)
    src = src_pool[:mapping_size]
    tgt = tgt_pool[:mapping_size]
    return dict(zip(src, tgt))


def apply_mapping(text: str, mapping: Dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


def infer_query_from_prompt(prompt: str) -> str:
    candidates = []
    p1 = re.findall(r"transformed form of\s*([^?.!]+)", prompt, flags=re.I)
    p2 = re.findall(r"Transform\s*([^?.!]+)", prompt, flags=re.I)
    p3 = re.findall(r"output for\s*([^?.!]+)", prompt, flags=re.I)
    candidates.extend(p1)
    candidates.extend(p2)
    candidates.extend(p3)
    if candidates:
        return candidates[-1].strip().strip("'\"")

    arrows = re.findall(r"([^\n:;]+?)\s*->\s*([^\n;,.]+)", prompt)
    if arrows:
        return arrows[-1][0].strip().strip("'\"")
    return ""


def _collect_support_pairs(parsed_train_sample: Dict[str, Any]) -> tuple[List[str], List[str]]:
    support_inputs: List[str] = []
    support_outputs: List[str] = []
    for inp, out in parsed_train_sample.get("support_examples", []):
        support_inputs.append(inp.strip())
        support_outputs.append(out.strip())
    return support_inputs, support_outputs


def _infer_mapping_score(support_inputs: List[str], support_outputs: List[str]) -> tuple[Dict[str, str], int]:
    mapping: Dict[str, str] = {}
    score = 0
    for src, tgt in zip(support_inputs, support_outputs):
        if len(src) != len(tgt):
            continue
        if any(a in mapping and mapping[a] != b for a, b in zip(src, tgt)):
            continue
        for a, b in zip(src, tgt):
            mapping[a] = b
        score += 1
    return mapping, score


class EquationSpec(PuzzleSpec):
    name = "equation"

    def sample_config(self, difficulty: Optional[str] = None) -> Dict[str, Any]:
        if difficulty == "easy":
            mapping_size = 8
            n_support = 3
        elif difficulty == "hard":
            mapping_size = 14
            n_support = 4
        else:
            mapping_size = random.choice([8, 10, 12, 14])
            n_support = random.choice([3, 4])

        mapping = build_mapping(mapping_size=mapping_size)
        support_inputs = [random_equation_string() for _ in range(n_support)]
        query_input = random_equation_string()

        return {
            "family": self.name,
            "mapping": mapping,
            "support_inputs": support_inputs,
            "query_input": query_input,
            "difficulty": difficulty or "medium",
            "template_id": random.choice(["alice", "examples", "cipher"]),
        }

    def render(self, config: Dict[str, Any]) -> str:
        examples = [f"{x} -> {apply_mapping(x, config['mapping'])}" for x in config["support_inputs"]]

        t = config["template_id"]
        if t == "alice":
            lines = [
                "In Alice's Wonderland, a secret set of transformation rules is applied to equations.",
                "Here are some examples:",
                *examples,
                f"What is the transformed form of {config['query_input']}?",
            ]
            return " ".join(lines)

        if t == "cipher":
            lines = [
                "A hidden symbolic rewrite system transforms expressions.",
                "Examples:",
                *examples,
                f"Transform {config['query_input']}.",
            ]
            return " ".join(lines)

        lines = [
            "An unknown equation transformation follows these examples:",
            *examples,
            f"Find the transformed output for {config['query_input']}.",
        ]
        return " ".join(lines)

    def solve(self, config: Dict[str, Any]) -> str:
        return apply_mapping(config["query_input"], config["mapping"])

    def verify(self, config: Dict[str, Any], answer: str) -> bool:
        return answer.strip() == self.solve(config)

    def canonicalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        mapping_items = sorted((k, v) for k, v in config["mapping"].items())
        return {
            "family": self.name,
            "mapping": mapping_items,
            "support_inputs": sorted(config["support_inputs"]),
            "query_input": config["query_input"],
            "difficulty": config.get("difficulty", "medium"),
        }

    def reproduce(self, parsed_train_sample: Dict[str, Any]) -> Dict[str, Any]:
        support_inputs, support_outputs = _collect_support_pairs(parsed_train_sample)
        mapping, score = _infer_mapping_score(support_inputs, support_outputs)

        query_input = parsed_train_sample.get("query_input") or ""
        if not query_input:
            query_input = infer_query_from_prompt(parsed_train_sample.get("prompt", ""))

        return {
            "family": self.name,
            "mapping": mapping,
            "support_inputs": support_inputs,
            "query_input": query_input,
            "difficulty": "unknown",
            "template_id": "alice",
            "reproduction_score": score,
        }