from __future__ import annotations

import random
from typing import Any, Dict, Optional

from .base import PuzzleSpec


class GravitySpec(PuzzleSpec):
    name = "gravity"

    def sample_config(self, difficulty: Optional[str] = None) -> Dict[str, Any]:
        mode = random.choice(["linear", "scaled"])

        if mode == "linear":
            a = random.choice([1.5, 2.0, 3.0, 4.0, 4.5])
            b = random.choice([0.0, 3.0, 9.81, 12.5, 15.0])
            params = {"mode": mode, "a": a, "b": b}
        else:
            g = random.choice([3.0, 4.0, 5.5, 8.0, 9.81, 12.0])
            params = {"mode": mode, "g": g}

        n_support = 4 if difficulty == "hard" else 3
        support_inputs = [round(random.uniform(1, 40), 2) for _ in range(n_support)]
        query_input = round(random.uniform(1, 40), 2)

        return {
            "family": self.name,
            **params,
            "support_inputs": support_inputs,
            "query_input": query_input,
            "difficulty": difficulty or "medium",
            "template_id": random.choice(["alice", "examples", "physics"]),
        }

    def transform(self, x: float, config: Dict[str, Any]) -> float:
        if config["mode"] == "linear":
            return config["a"] * x + config["b"]
        return config["g"] * x

    def fmt(self, x: float) -> str:
        return f"{x:.2f}"

    def render(self, config: Dict[str, Any]) -> str:
        examples = [f"{self.fmt(x)} -> {self.fmt(self.transform(x, config))}" for x in config["support_inputs"]]

        t = config["template_id"]
        if t == "alice":
            lines = [
                "In Alice's Wonderland, the gravitational constant has been secretly changed.",
                "A measurement is transformed according to the following examples:",
                *examples,
                f"What is the transformed value for {self.fmt(config['query_input'])}?",
            ]
            return " ".join(lines)

        if t == "physics":
            lines = [
                "A hidden physical law changes the measured values in this world.",
                "Examples:",
                *examples,
                f"Compute the transformed value for {self.fmt(config['query_input'])}.",
            ]
            return " ".join(lines)

        lines = [
            "An unknown gravity-based transformation follows these examples:",
            *examples,
            f"Find the output for {self.fmt(config['query_input'])}.",
        ]
        return " ".join(lines)

    def solve(self, config: Dict[str, Any]) -> str:
        return self.fmt(self.transform(float(config["query_input"]), config))

    def verify(self, config: Dict[str, Any], answer: str) -> bool:
        return answer.strip() == self.solve(config)

    def canonicalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "family": self.name,
            "mode": config["mode"],
            "support_inputs": sorted(round(float(x), 6) for x in config["support_inputs"]),
            "query_input": round(float(config["query_input"]), 6),
            "difficulty": config.get("difficulty", "medium"),
        }
        if config["mode"] == "linear":
            out["a"] = round(float(config["a"]), 6)
            out["b"] = round(float(config["b"]), 6)
        else:
            out["g"] = round(float(config["g"]), 6)
        return out