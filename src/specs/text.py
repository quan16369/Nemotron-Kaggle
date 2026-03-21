from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .base import PuzzleSpec


WORDS = [
    "alice", "wonderland", "secret", "rabbit", "garden", "clock", "mirror", "puzzle",
    "silver", "golden", "forest", "lantern", "meadow", "castle", "river", "shadow",
    "hidden", "message", "cipher", "pattern", "violet", "winter", "summer", "autumn"
]


def random_phrase(n_words: Optional[int] = None) -> str:
    if n_words is None:
        n_words = random.randint(2, 5)
    return " ".join(random.choice(WORDS) for _ in range(n_words))


def caesar_shift_char(ch: str, k: int) -> str:
    if 'a' <= ch <= 'z':
        return chr((ord(ch) - ord('a') + k) % 26 + ord('a'))
    return ch


def caesar_shift_text(text: str, k: int) -> str:
    return "".join(caesar_shift_char(c, k) for c in text)


def apply_text_rule(text: str, rule_name: str) -> str:
    if rule_name == "caesar_1":
        return caesar_shift_text(text, 1)
    if rule_name == "caesar_2":
        return caesar_shift_text(text, 2)
    if rule_name == "caesar_3":
        return caesar_shift_text(text, 3)
    if rule_name == "caesar_-1":
        return caesar_shift_text(text, -1)
    if rule_name == "reverse_chars":
        return text[::-1]
    if rule_name == "reverse_words":
        return " ".join(text.split()[::-1])
    if rule_name == "reverse_then_caesar_1":
        return caesar_shift_text(text[::-1], 1)
    if rule_name == "reverse_then_caesar_2":
        return caesar_shift_text(text[::-1], 2)
    raise ValueError(f"Unknown rule: {rule_name}")


class TextDecryptSpec(PuzzleSpec):
    name = "text_decrypt"

    def sample_config(self, difficulty: Optional[str] = None) -> Dict[str, Any]:
        if difficulty == "easy":
            rule_name = random.choice(["caesar_1", "caesar_2", "reverse_words"])
            n_support = 3
        elif difficulty == "hard":
            rule_name = random.choice(["reverse_then_caesar_1", "reverse_then_caesar_2", "caesar_3"])
            n_support = 4
        else:
            rule_name = random.choice([
                "caesar_1", "caesar_2", "caesar_3", "caesar_-1",
                "reverse_chars", "reverse_words",
                "reverse_then_caesar_1", "reverse_then_caesar_2"
            ])
            n_support = random.choice([3, 4])

        support_inputs = [random_phrase() for _ in range(n_support)]
        query_input = random_phrase()

        return {
            "family": self.name,
            "rule_name": rule_name,
            "support_inputs": support_inputs,
            "query_input": query_input,
            "difficulty": difficulty or "medium",
            "template_id": random.choice(["alice", "examples", "cipher"]),
        }

    def render(self, config: Dict[str, Any]) -> str:
        examples = [f"'{x}' -> '{apply_text_rule(x, config['rule_name'])}'" for x in config["support_inputs"]]

        t = config["template_id"]
        if t == "alice":
            lines = [
                "In Alice's Wonderland, secret encryption rules are used on text.",
                "Here are some examples:",
                *examples,
                f"What is the transformed text for '{config['query_input']}'?",
            ]
            return " ".join(lines)

        if t == "cipher":
            lines = [
                "A hidden text cipher transforms phrases using one fixed rule.",
                "Examples:",
                *examples,
                f"Transform the phrase '{config['query_input']}'.",
            ]
            return " ".join(lines)

        lines = [
            "An unknown text transformation follows these examples:",
            *examples,
            f"Find the transformed output for '{config['query_input']}'.",
        ]
        return " ".join(lines)

    def solve(self, config: Dict[str, Any]) -> str:
        return apply_text_rule(config["query_input"], config["rule_name"])

    def verify(self, config: Dict[str, Any], answer: str) -> bool:
        return answer.strip() == self.solve(config)

    def canonicalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "family": self.name,
            "rule_name": config["rule_name"],
            "support_inputs": sorted(config["support_inputs"]),
            "query_input": config["query_input"],
            "difficulty": config.get("difficulty", "medium"),
        }

    def reproduce(self, parsed_train_sample: Dict[str, Any]) -> Dict[str, Any]:
        support_inputs = []
        support_outputs = []
        for inp, out in parsed_train_sample.get("support_examples", []):
            support_inputs.append(inp.strip().strip("'\""))
            support_outputs.append(out.strip().strip("'\""))

        rules = [
            "caesar_1",
            "caesar_2",
            "caesar_3",
            "caesar_-1",
            "reverse_chars",
            "reverse_words",
            "reverse_then_caesar_1",
            "reverse_then_caesar_2",
        ]

        best_rule = "reverse_words"
        best_score = -1
        for rule in rules:
            score = 0
            for src, tgt in zip(support_inputs, support_outputs):
                if apply_text_rule(src, rule) == tgt:
                    score += 1
            if score > best_score:
                best_score = score
                best_rule = rule

        query_input = parsed_train_sample.get("query_input")
        if not query_input:
            query_input = support_inputs[-1] if support_inputs else "alice puzzle"

        return {
            "family": self.name,
            "rule_name": best_rule,
            "support_inputs": support_inputs if support_inputs else ["alice puzzle", "hidden message", "silver clock"],
            "query_input": query_input.strip().strip("'\""),
            "difficulty": "unknown",
            "template_id": "alice",
            "reproduction_score": best_score,
        }