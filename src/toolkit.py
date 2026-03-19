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
    read_jsonl,
    merge_jsonl,
    export_chatml,
    to_chatml_record,
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

DEFAULT_PROMPT_SUFFIX = (
    "\n\nInfer the hidden rule from the examples."
    "\nReason concisely."
    "\nGive exactly one final answer in \\boxed{}."
)

DEFAULT_REAL_STYLE_MIX = {
    "answer_only": 0.60,
    "ultra_short": 0.25,
    "concise": 0.15,
}


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


def make_real_completion(answer: str, style: str) -> str:
    boxed = f"\\boxed{{{answer.strip()}}}"
    if style == "answer_only":
        return boxed
    if style == "ultra_short":
        return f"Apply the same rule.\n\n{boxed}"
    if style == "concise":
        return (
            "The support examples are consistent with one rule. "
            "Applying it to the final query gives:\n\n"
            f"{boxed}"
        )
    raise ValueError(f"Unknown real-data style: {style}")


def sample_from_mix(weights: Dict[str, float], rng: random.Random) -> str:
    normalized = normalize_weights(weights)
    threshold = rng.random()
    running = 0.0
    for key, value in normalized.items():
        running += value
        if threshold <= running:
            return key
    return next(iter(normalized))


def build_user_message(prompt: str, prompt_suffix: str) -> str:
    return prompt.strip() + prompt_suffix


def make_real_chat_record(
    row: Dict[str, str],
    style: str,
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
) -> Dict[str, Any]:
    family = classify_family(row["prompt"])
    user_prompt = build_user_message(row["prompt"], prompt_suffix)
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": make_real_completion(row["answer"], style)},
        ],
        "meta": {
            "source": "real_train",
            "sample_id": row["id"],
            "family": family,
            "style": style,
        },
    }


def make_validation_record(
    row: Dict[str, str],
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
) -> Dict[str, Any]:
    family = classify_family(row["prompt"])
    return {
        "id": row["id"],
        "prompt": row["prompt"],
        "answer": row["answer"],
        "user_message": build_user_message(row["prompt"], prompt_suffix),
        "meta": {
            "source": "real_train_holdout",
            "family": family,
        },
    }


def coerce_to_chat_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    if "messages" in rec:
        out = dict(rec)
    elif "completion" in rec and "prompt" in rec:
        out = to_chatml_record(rec)
    elif "prompt" in rec and "answer" in rec:
        out = {
            "messages": [
                {
                    "role": "user",
                    "content": build_user_message(rec["prompt"], DEFAULT_PROMPT_SUFFIX),
                },
                {
                    "role": "assistant",
                    "content": make_real_completion(rec["answer"], "answer_only"),
                },
            ],
            "meta": rec.get("meta", {}),
        }
    else:
        raise ValueError("Unsupported record format for chat conversion")

    meta = dict(out.get("meta", {}))
    meta.setdefault("source", "synthetic")
    out["meta"] = meta
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


def cmd_build_comp_data(args):
    style_mix = json.loads(args.real_style_mix)
    rng = random.Random(args.seed)

    with open(args.train_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.holdout_ids_in:
        with open(args.holdout_ids_in, encoding="utf-8") as f:
            val_ids = set(json.load(f))
    else:
        shuffled_rows = rows[:]
        rng.shuffle(shuffled_rows)
        val_count = int(len(shuffled_rows) * args.val_fraction)
        val_ids = {row["id"] for row in shuffled_rows[:val_count]}

    if args.holdout_ids_out:
        holdout_path = Path(args.holdout_ids_out)
        holdout_path.parent.mkdir(parents=True, exist_ok=True)
        holdout_path.write_text(
            json.dumps(sorted(val_ids), indent=2),
            encoding="utf-8",
        )

    real_train_records = []
    val_records = []

    for row in rows:
        if row["id"] in val_ids:
            val_records.append(
                make_validation_record(
                    row,
                    prompt_suffix=args.prompt_suffix,
                )
            )
            continue

        for _ in range(args.real_repeat):
            style = sample_from_mix(style_mix, rng)
            real_train_records.append(
                make_real_chat_record(
                    row,
                    style=style,
                    prompt_suffix=args.prompt_suffix,
                )
            )

    synth_records = []
    for path in args.synth:
        for rec in read_jsonl(path):
            synth_records.append(coerce_to_chat_record(rec))

    if args.synth_cap is not None and len(synth_records) > args.synth_cap:
        rng.shuffle(synth_records)
        synth_records = synth_records[: args.synth_cap]

    train_records = real_train_records + synth_records
    rng.shuffle(train_records)
    rng.shuffle(val_records)

    write_jsonl(train_records, args.out_train)
    if args.out_val:
        write_jsonl(val_records, args.out_val)

    summary = {
        "real_examples": len(rows),
        "train_real_records": len(real_train_records),
        "train_synth_records": len(synth_records),
        "train_total_records": len(train_records),
        "val_records": len(val_records),
        "real_repeat": args.real_repeat,
        "style_mix": style_mix,
    }
    print(json.dumps(summary, indent=2))


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

    b = sub.add_parser("build_comp_data")
    b.add_argument("--train-csv", required=True)
    b.add_argument("--out-train", required=True)
    b.add_argument("--out-val")
    b.add_argument("--synth", nargs="*", default=[])
    b.add_argument("--val-fraction", type=float, default=0.1)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--real-repeat", type=int, default=2)
    b.add_argument(
        "--real-style-mix",
        default=json.dumps(DEFAULT_REAL_STYLE_MIX),
    )
    b.add_argument("--synth-cap", type=int)
    b.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX)
    b.add_argument("--holdout-ids-in")
    b.add_argument("--holdout-ids-out")
    b.set_defaults(func=cmd_build_comp_data)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
