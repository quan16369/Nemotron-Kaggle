from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.generation.render_dataset import make_sft_record, write_jsonl, read_jsonl
from src.specs.registry import SPEC_REGISTRY


def _load_yaml_optional(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "YAML spec requires PyYAML. Install with `pip install pyyaml` or use JSON spec."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("DSL spec must deserialize to a dictionary")
    return data


def load_spec(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DSL spec not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml_optional(p)
    else:
        raise ValueError("Unsupported DSL extension. Use .json/.yaml/.yml")

    if "family" not in data:
        raise ValueError("DSL spec must contain `family`")
    if data["family"] not in SPEC_REGISTRY:
        raise ValueError(f"Unknown family in DSL spec: {data['family']}")
    return data


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("difficulty_mix must have positive total weight")
    return {k: v / s for k, v in weights.items()}


def _weighted_pick(weights: Dict[str, float], rng: random.Random) -> str:
    w = _normalize_weights(weights)
    r = rng.random()
    running = 0.0
    for key, val in w.items():
        running += val
        if r <= running:
            return key
    return next(iter(w))


def _set_nested(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _apply_override(cfg: Dict[str, Any], key: str, rule: Any, rng: random.Random) -> None:
    if not isinstance(rule, dict):
        _set_nested(cfg, key, rule)
        return

    kind = rule.get("type", "literal")
    if kind == "choice":
        _set_nested(cfg, key, rng.choice(rule["values"]))
        return

    if kind == "int_range":
        lo = int(rule["min"])
        hi = int(rule["max"])
        _set_nested(cfg, key, rng.randint(lo, hi))
        return

    if kind == "float_range":
        lo = float(rule["min"])
        hi = float(rule["max"])
        nd = int(rule.get("round", 6))
        _set_nested(cfg, key, round(rng.uniform(lo, hi), nd))
        return

    if kind == "literal":
        _set_nested(cfg, key, rule.get("value"))
        return

    raise ValueError(f"Unsupported override type: {kind}")


def _sample_config(spec_doc: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    family = spec_doc["family"]
    spec = SPEC_REGISTRY[family]

    difficulty_mix = spec_doc.get("difficulty_mix", {"easy": 0.3, "medium": 0.5, "hard": 0.2})
    difficulty = _weighted_pick(difficulty_mix, rng)

    cfg = spec.sample_config(difficulty=difficulty)

    defaults = spec_doc.get("defaults", {})
    for k, v in defaults.items():
        _set_nested(cfg, k, v)

    overrides = spec_doc.get("overrides", {})
    for k, rule in overrides.items():
        _apply_override(cfg, k, rule, rng)

    return cfg


def generate_configs_from_spec(
    spec_path: str,
    count: int,
    out_path: str,
    seed: int = 42,
) -> Dict[str, Any]:
    spec_doc = load_spec(spec_path)
    family = spec_doc["family"]
    rng = random.Random(seed)

    configs = []
    for _ in range(count):
        cfg = _sample_config(spec_doc, rng)
        configs.append(cfg)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(configs, str(out_file))

    return {
        "family": family,
        "count": len(configs),
        "out": str(out_file),
        "seed": seed,
    }


def render_records_from_configs(
    spec_path: str,
    configs_path: str,
    out_path: str,
    strict_verify: bool = True,
    style_mix: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    spec_doc = load_spec(spec_path)
    family = spec_doc["family"]

    configs = read_jsonl(configs_path)
    rng = random.Random(seed)

    default_mix = {"concise": 0.7, "ultra_short": 0.2, "answer_only": 0.1}
    mix = style_mix or default_mix

    def pick_style() -> str:
        return _weighted_pick(mix, rng)

    records = []
    verify_failed = 0
    for cfg in configs:
        rec = make_sft_record(family, cfg, style=pick_style())
        if rec["meta"].get("verifier_pass", False) or not strict_verify:
            records.append(rec)
        else:
            verify_failed += 1

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, str(out_file))

    return {
        "family": family,
        "input_configs": len(configs),
        "written_records": len(records),
        "verify_failed": verify_failed,
        "out": str(out_file),
    }


def synth_from_spec(
    spec_path: str,
    count: int,
    out_path: str,
    seed: int = 42,
    strict_verify: bool = True,
) -> Dict[str, Any]:
    spec_doc = load_spec(spec_path)
    family = spec_doc["family"]
    rng = random.Random(seed)

    records = []
    verify_failed = 0
    for _ in range(count):
        cfg = _sample_config(spec_doc, rng)
        rec = make_sft_record(family, cfg, style=_weighted_pick({"concise": 0.7, "ultra_short": 0.2, "answer_only": 0.1}, rng))
        if rec["meta"].get("verifier_pass", False) or not strict_verify:
            records.append(rec)
        else:
            verify_failed += 1

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, str(out_file))

    return {
        "family": family,
        "requested": count,
        "written": len(records),
        "verify_failed": verify_failed,
        "out": str(out_file),
        "seed": seed,
    }
