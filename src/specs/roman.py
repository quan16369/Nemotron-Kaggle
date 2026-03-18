from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .base import PuzzleSpec


ROMAN_MAP = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def int_to_roman(n: int) -> str:
    if n <= 0:
        raise ValueError("Roman numerals require positive integers")
    out: List[str] = []
    for value, symbol in ROMAN_MAP:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


class RomanNumeralSpec(PuzzleSpec):
    name = "roman_numeral"

    def sample_config(self, difficulty: Optional[str] = None) -> Dict[str, Any]:
        if difficulty == "easy":
            pool = list(range(1, 21))
            n_support = 3
        elif difficulty == "hard":
            pool = list(range(30, 101))
            n_support = 4
        else:
            pool = list(range(1, 101))
            n_support = random.choice([3, 4])

        values = random.sample(pool, n_support + 1)
        support_values = values[:-1]
        query_value = values[-1]

        return {
            "family": self.name,
            "mapping": "standard_roman",
            "support_values": support_values,
            "query_value": query_value,
            "difficulty": difficulty or "medium",
            "template_id": random.choice(["alice", "system", "examples"]),
        }

    def render(self, config: Dict[str, Any]) -> str:
        template_id = config["template_id"]
        support_values = config["support_values"]
        query_value = config["query_value"]

        if template_id == "alice":
            lines = [
                "In Alice's Wonderland, numbers are secretly converted into a different numeral system.",
                "Here are some examples:",
            ]
            lines += [f"{v} -> {int_to_roman(v)}" for v in support_values]
            lines.append(f"What is the secret-system form of {query_value}?")
            return " ".join(lines)

        if template_id == "system":
            lines = [
                "A hidden numeral system maps integers to symbols.",
                "Examples:",
            ]
            lines += [f"{v} -> {int_to_roman(v)}" for v in support_values]
            lines.append(f"Convert {query_value} into the hidden numeral system.")
            return " ".join(lines)

        lines = [
            "An unknown numeral notation follows these examples:",
        ]
        lines += [f"{v} -> {int_to_roman(v)}" for v in support_values]
        lines.append(f"Write the notation for {query_value}.")
        return " ".join(lines)

    def solve(self, config: Dict[str, Any]) -> str:
        return int_to_roman(int(config["query_value"]))

    def verify(self, config: Dict[str, Any], answer: str) -> bool:
        return answer.strip() == self.solve(config)

    def canonicalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "family": self.name,
            "mapping": "standard_roman",
            "support_values": sorted(int(x) for x in config["support_values"]),
            "query_value": int(config["query_value"]),
            "difficulty": config.get("difficulty", "medium"),
        }

    def reproduce(self, parsed_train_sample: Dict[str, Any]) -> Dict[str, Any]:
        support_values = []
        for inp, _out in parsed_train_sample.get("support_examples", []):
            support_values.append(int(inp))
        query_value = int(parsed_train_sample["query_input"])

        return {
            "family": self.name,
            "mapping": "standard_roman",
            "support_values": support_values,
            "query_value": query_value,
            "difficulty": "unknown",
            "template_id": "alice",
        }