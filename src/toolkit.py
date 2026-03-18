from __future__ import annotations

import csv
import json
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

from src.specs.registry import SPEC_REGISTRY
from src.generation.render_dataset import (
    make_sft_record,
    write_jsonl,
    merge_jsonl,
    export_chatml,
    choose_style,
)

# =========================================================
# =============== PARSING TRAIN DATA ======================
# =========================================================


@dataclass
class ParsedSample:
    sample_id: str
    family: str
    prompt: str
    answer: str
    support_examples: List[Tuple[str, str]]
    query_input: Optional[str]


FAMILY_PATTERNS = [
    ("bit_binary", re.compile(r"bit manipulation rule transforms 8-bit binary", re.I)),
    ("text_decrypt", re.compile(r"secret encryption rules are used on text", re.I)),
    ("roman_numeral", re.compile(r"different numeral system", re.I)),
    ("equation", re.compile(r"transformation rules.*applied to equations", re.I)),
    ("unit_conversion", re.compile(r"unit conversion is applied to measurements", re.I)),
    ("gravity", re.compile(r"gravitational constant has been secretly changed", re.I)),
]


def classify_family(prompt: str) -> str:
    for family, pat in FAMILY_PATTERNS:
        if pat.search(prompt):
            return family
    return "unknown"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedupe_pairs(pairs):
    seen = set()
    out = []
    for a, b in pairs:
        if (a, b) not in seen:
            out.append((a, b))
            seen.add((a, b))
    return out


def find_examples(prompt: str):
    pattern = re.compile(r"([^\n:;]+?)\s*->\s*([^\n;,.]+)")
    matches = pattern.findall(prompt)
    return dedupe_pairs([(normalize_ws(a), normalize_ws(b)) for a, b in matches])


def extract_query(prompt: str, family: str):
    if family == "bit_binary":
        m = re.findall(r"\b[01]{8}\b", prompt)
        return m[-1] if m else None

    if family in {"roman_numeral", "unit_conversion", "gravity"}:
        m = re.findall(r"\d+(?:\.\d+)?", prompt)
        return m[-1] if m else None

    m = re.findall(r"'([^']+)'", prompt)
    return m[-1] if m else None


def parse_sample(row):
    family = classify_family(row["prompt"])
    return ParsedSample(
        sample_id=row["id"],
        family=family,
        prompt=row["prompt"],
        answer=row.get("answer", ""),
        support_examples=find_examples(row["prompt"]),
        query_input=extract_query(row["prompt"], family),
    )


def load_train_csv(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(parse_sample(row))
    return out


def analyze_dataset(samples):
    summary = {"total": len(samples), "by_family": {}}
    for s in samples:
        summary["by_family"].setdefault(s.family, 0)
        summary["by_family"][s.family] += 1
    return summary


# =========================================================
# =============== SYNTHETIC GENERATION =====================
# =========================================================


def canonical_key(spec, config):
    return json.dumps(spec.canonicalize(config), sort_keys=True)


def normalize_weights(w):
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def allocate_counts(n, weights):
    weights = normalize_weights(weights)
    base = {k: int(n * v) for k, v in weights.items()}
    remainder = n - sum(base.values())

    sorted_keys = sorted(weights, key=lambda k: weights[k], reverse=True)
    for i in range(remainder):
        base[sorted_keys[i % len(sorted_keys)]] += 1
    return base


def sample_unique(spec, difficulty, seen, max_try=100):
    for _ in range(max_try):
        cfg = spec.sample_config(difficulty=difficulty)
        key = canonical_key(spec, cfg)
        if key not in seen:
            return cfg, key
    return None, None


def build_synth_records(
    size_map,
    difficulty_mix=None,
    oversample_factor=1.5,
    seed=42,
):
    rng = random.Random(seed)

    default_mix = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    difficulty_mix = difficulty_mix or {}

    out = {}

    for family, target_n in size_map.items():
        spec = SPEC_REGISTRY[family]
        mix = difficulty_mix.get(family, default_mix)

        generate_n = int(target_n * oversample_factor)
        bucket_sizes = allocate_counts(generate_n, mix)

        seen = set()
        bucket_data = {k: [] for k in bucket_sizes}

        # generate
        for diff, n in bucket_sizes.items():
            while len(bucket_data[diff]) < n:
                cfg, key = sample_unique(spec, diff, seen)
                if cfg is None:
                    continue
                seen.add(key)

                bucket_data[diff].append(
                    make_sft_record(family, cfg, style=choose_style())
                )

        # merge
        all_records = []
        for v in bucket_data.values():
            all_records.extend(v)

        # trim back
        if len(all_records) > target_n:
            final_sizes = allocate_counts(target_n, mix)

            grouped = {"easy": [], "medium": [], "hard": []}
            for r in all_records:
                d = r["meta"]["config"]["difficulty"]
                grouped[d].append(r)

            final = []
            for d in ["easy", "medium", "hard"]:
                rng.shuffle(grouped[d])
                final.extend(grouped[d][: final_sizes[d]])

            all_records = final

        rng.shuffle(all_records)
        out[family] = all_records[:target_n]

    return out


# =========================================================
# ================== CLI COMMANDS ==========================
# =========================================================


def cmd_analyze(args):
    samples = load_train_csv(args.csv)
    print(json.dumps(analyze_dataset(samples), indent=2))


def cmd_synth_all(args):
    size_map = json.loads(args.sizes)
    mix = json.loads(args.difficulty_mix) if args.difficulty_mix else None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = build_synth_records(
        size_map=size_map,
        difficulty_mix=mix,
        oversample_factor=args.oversample_factor,
        seed=args.seed,
    )

    for family, records in data.items():
        write_jsonl(records, str(outdir / f"{family}.jsonl"))
        print(f"{family}: {len(records)}")


def cmd_make_mixture(args):
    merge_jsonl(args.inputs, args.out)
    print("merged")


def cmd_export_chatml(args):
    export_chatml(args.input, args.out)
    print("chatml done")


# =========================================================
# ==================== ARGPARSE ============================
# =========================================================


def build_parser():
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("--csv", required=True)
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("synth_all")
    s.add_argument("--outdir", required=True)
    s.add_argument("--sizes", required=True)
    s.add_argument("--difficulty_mix")
    s.add_argument("--oversample_factor", type=float, default=1.5)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_synth_all)

    m = sub.add_parser("make_mixture")
    m.add_argument("--out", required=True)
    m.add_argument("inputs", nargs="+")
    m.set_defaults(func=cmd_make_mixture)

    e = sub.add_parser("export_chatml")
    e.add_argument("--input", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_export_chatml)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)