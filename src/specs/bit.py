from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

from .base import PuzzleSpec


def b_not(x: str) -> str:
    return "".join("1" if c == "0" else "0" for c in x)


def b_rev(x: str) -> str:
    return x[::-1]


def b_rotl(x: str, k: int) -> str:
    k %= len(x)
    return x[k:] + x[:k]


def b_rotr(x: str, k: int) -> str:
    k %= len(x)
    return x[-k:] + x[:-k] if k else x


def b_swap_halves(x: str) -> str:
    m = len(x) // 2
    return x[m:] + x[:m]


def b_xor_const(x: str, c: str) -> str:
    return "".join("1" if a != b else "0" for a, b in zip(x, c))


def build_bit_ops() -> Dict[str, Callable[[str], str]]:
    ops: Dict[str, Callable[[str], str]] = {
        "identity": lambda x: x,
        "not": b_not,
        "rev": b_rev,
        "swap_halves": b_swap_halves,
    }
    for k in range(1, 8):
        ops[f"rotl_{k}"] = lambda x, k=k: b_rotl(x, k)
        ops[f"rotr_{k}"] = lambda x, k=k: b_rotr(x, k)

    for c in ["00000001", "00000011", "00000111", "00001111", "01010101", "10101010", "11111111"]:
        ops[f"xor_{c}"] = lambda x, c=c: b_xor_const(x, c)
    return ops


def rand_bin8() -> str:
    return format(random.randint(0, 255), "08b")


def apply_rule_chain(x: str, rule_chain: List[str]) -> str:
    ops = build_bit_ops()
    out = x
    for name in rule_chain:
        out = ops[name](out)
    return out


class BitBinarySpec(PuzzleSpec):
    name = "bit_binary"

    def sample_config(self, difficulty: Optional[str] = None) -> Dict[str, Any]:
        ops = list(build_bit_ops().keys())

        if difficulty == "easy":
            depth = 1
        elif difficulty == "hard":
            depth = 3
        else:
            depth = random.choice([1, 2])

        rule_chain = [random.choice(ops) for _ in range(depth)]
        n_support = 3 if depth <= 2 else 4
        support_inputs = [rand_bin8() for _ in range(n_support)]
        query_input = rand_bin8()

        return {
            "family": self.name,
            "rule_chain": rule_chain,
            "support_inputs": support_inputs,
            "query_input": query_input,
            "difficulty": difficulty or ("easy" if depth == 1 else "medium"),
            "template_id": random.choice(["alice", "machine", "examples"]),
        }

    def render(self, config: Dict[str, Any]) -> str:
        support_inputs = config["support_inputs"]
        query_input = config["query_input"]

        examples = [f"{x} -> {apply_rule_chain(x, config['rule_chain'])}" for x in support_inputs]

        template_id = config["template_id"]
        if template_id == "alice":
            lines = [
                "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.",
                "The transformation works like this:",
                *examples,
                f"What does the rule produce for {query_input}?",
            ]
            return " ".join(lines)

        if template_id == "machine":
            lines = [
                "A hidden machine transforms 8-bit binary strings using one fixed rule.",
                "Examples:",
                *examples,
                f"Apply the same rule to {query_input}.",
            ]
            return " ".join(lines)

        lines = [
            "An unknown binary transformation follows these examples:",
            *examples,
            f"Find the output for {query_input}.",
        ]
        return " ".join(lines)

    def solve(self, config: Dict[str, Any]) -> str:
        return apply_rule_chain(config["query_input"], config["rule_chain"])

    def verify(self, config: Dict[str, Any], answer: str) -> bool:
        ans = answer.strip()
        if len(ans) != 8 or any(c not in "01" for c in ans):
            return False
        return ans == self.solve(config)

    def canonicalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "family": self.name,
            "rule_chain": list(config["rule_chain"]),
            "support_inputs": sorted(config["support_inputs"]),
            "query_input": config["query_input"],
            "difficulty": config.get("difficulty", "medium"),
        }

    def reproduce(self, parsed_train_sample: Dict[str, Any]) -> Dict[str, Any]:
        support_inputs = []
        support_outputs = []
        for inp, out in parsed_train_sample.get("support_examples", []):
            support_inputs.append(inp.strip())
            support_outputs.append(out.strip())

        ops = list(build_bit_ops().keys())
        candidates: List[List[str]] = []

        for op in ops:
            candidates.append([op])

        for op1 in ops:
            for op2 in ops:
                candidates.append([op1, op2])

        best_chain = None
        best_score = -1

        for chain in candidates:
            score = 0
            for x, y in zip(support_inputs, support_outputs):
                if apply_rule_chain(x, chain) == y:
                    score += 1
            if score > best_score:
                best_score = score
                best_chain = chain

        return {
            "family": self.name,
            "rule_chain": best_chain or ["identity"],
            "support_inputs": support_inputs,
            "query_input": parsed_train_sample["query_input"],
            "difficulty": "unknown",
            "template_id": "alice",
            "reproduction_score": best_score,
        }