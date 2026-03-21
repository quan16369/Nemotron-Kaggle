from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional

from .base import PuzzleSpec


def _parse_value_and_unit(text: str):
    m = re.match(r"\s*([-+]?\d+(?:\.\d+)?)\s*(.*)\s*$", text)
    if not m:
        return None, ""
    return float(m.group(1)), m.group(2).strip()


def _extract_unit_pairs(parsed_train_sample: Dict[str, Any]):
    pairs = []
    for inp, out in parsed_train_sample.get("support_examples", []):
        x, in_unit = _parse_value_and_unit(inp)
        y, out_unit = _parse_value_and_unit(out)
        if x is None or y is None:
            continue
        pairs.append((x, y, in_unit, out_unit))
    return pairs


def _fit_affine_from_pairs(pairs):
    a, b = 1.0, 0.0
    if len(pairs) >= 2:
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                x1, y1, _, _ = pairs[i]
                x2, y2, _, _ = pairs[j]
                if abs(x2 - x1) < 1e-9:
                    continue
                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1
                return a, b
    if len(pairs) == 1:
        x1, y1, _, _ = pairs[0]
        return 1.0, y1 - x1
    return a, b


class UnitConversionSpec(PuzzleSpec):
    name = "unit_conversion"

    def sample_config(self, difficulty: Optional[str] = None) -> Dict[str, Any]:
        if difficulty == "easy":
            a = random.choice([0.5, 2.0, 2.54, 3.6])
            b = random.choice([0.0, 10.0, 32.0])
        elif difficulty == "hard":
            a = random.choice([0.25, 1.25, 1.8, 4.75, 5.5])
            b = random.choice([1.0, 5.0, 17.0, 29.0])
        else:
            a = random.choice([0.25, 0.5, 1.25, 1.8, 2.0, 2.54, 3.6, 4.75, 5.5])
            b = random.choice([0.0, 1.0, 5.0, 10.0, 17.0, 29.0, 32.0])

        unit_in = random.choice(["meters", "grams", "seconds", "liters", "degrees", "newtons"])
        unit_out = random.choice(["blips", "zogs", "flarns", "quonts", "drals", "mibs"])
        n_support = random.choice([3, 4])
        support_inputs = [round(random.uniform(1, 100), 2) for _ in range(n_support)]
        query_input = round(random.uniform(1, 100), 2)

        return {
            "family": self.name,
            "a": a,
            "b": b,
            "unit_in": unit_in,
            "unit_out": unit_out,
            "support_inputs": support_inputs,
            "query_input": query_input,
            "difficulty": difficulty or "medium",
            "template_id": random.choice(["alice", "examples", "system"]),
        }

    def transform(self, x: float, config: Dict[str, Any]) -> float:
        return config["a"] * x + config["b"]

    def fmt(self, x: float) -> str:
        return f"{x:.2f}"

    def render(self, config: Dict[str, Any]) -> str:
        examples = [
            f"{self.fmt(x)} {config['unit_in']} -> {self.fmt(self.transform(x, config))} {config['unit_out']}"
            for x in config["support_inputs"]
        ]

        t = config["template_id"]
        if t == "alice":
            lines = [
                "In Alice's Wonderland, a secret unit conversion is applied to measurements.",
                f"Here are examples converting {config['unit_in']} into {config['unit_out']}:",
                *examples,
                f"Convert {self.fmt(config['query_input'])} {config['unit_in']} into {config['unit_out']}.",
            ]
            return " ".join(lines)

        if t == "system":
            lines = [
                "A hidden conversion system maps measurements into new units.",
                "Examples:",
                *examples,
                f"What is the converted value of {self.fmt(config['query_input'])} {config['unit_in']}?",
            ]
            return " ".join(lines)

        lines = [
            "An unknown measurement conversion follows these examples:",
            *examples,
            f"Find the converted form of {self.fmt(config['query_input'])} {config['unit_in']}.",
        ]
        return " ".join(lines)

    def solve(self, config: Dict[str, Any]) -> str:
        return self.fmt(self.transform(float(config["query_input"]), config))

    def verify(self, config: Dict[str, Any], answer: str) -> bool:
        return answer.strip() == self.solve(config)

    def canonicalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "family": self.name,
            "a": round(float(config["a"]), 6),
            "b": round(float(config["b"]), 6),
            "unit_in": config["unit_in"],
            "unit_out": config["unit_out"],
            "support_inputs": sorted(round(float(x), 6) for x in config["support_inputs"]),
            "query_input": round(float(config["query_input"]), 6),
            "difficulty": config.get("difficulty", "medium"),
        }

    def reproduce(self, parsed_train_sample: Dict[str, Any]) -> Dict[str, Any]:
        pairs = _extract_unit_pairs(parsed_train_sample)
        a, b = _fit_affine_from_pairs(pairs)

        support_inputs = [p[0] for p in pairs]
        unit_in = pairs[0][2] if pairs else "units"
        unit_out = pairs[0][3] if pairs else "mapped_units"
        query_input = parsed_train_sample.get("query_input")
        if query_input is None:
            query_input = support_inputs[-1] if support_inputs else 1.0

        return {
            "family": self.name,
            "a": round(a, 6),
            "b": round(b, 6),
            "unit_in": unit_in,
            "unit_out": unit_out,
            "support_inputs": support_inputs if support_inputs else [1.0, 2.0, 3.0],
            "query_input": float(query_input),
            "difficulty": "unknown",
            "template_id": "alice",
            "reproduction_score": len(pairs),
        }