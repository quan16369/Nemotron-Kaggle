from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

from src.specs.registry import SPEC_REGISTRY
from src.generation.render_dataset import make_sft_record, write_jsonl, merge_jsonl, export_chatml, choose_style


@dataclass
class ParsedSample:
    sample_id: str
    family: str
    prompt: str
    answer: str
    support_examples: List[Tuple[str, str]]
    query_input: Optional[str]


FAMILY_PATTERNS: List[Tuple[str, re.Pattern]] = [
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


def dedupe_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for a, b in pairs:
        key = (a, b)
        if key not in seen:
            out.append((a, b))
            seen.add(key)
    return out


def find_arrow_examples(prompt: str) -> List[Tuple[str, str]]:
    patterns = [
        re.compile(r"([^\n:;]+?)\s*->\s*([^\n;,.]+)"),
        re.compile(r"([^\n:;]+?)\s*→\s*([^\n;,.]+)"),
        re.compile(r"input\s*[:=]\s*([^\n;,.]+?)\s*output\s*[:=]\s*([^\n;,.]+)", re.I),
    ]
    out: List[Tuple[str, str]] = []
    for pat in patterns:
        for m in pat.finditer(prompt):
            left = normalize_ws(m.group(1).strip(" \n\t\"'`"))
            right = normalize_ws(m.group(2).strip(" \n\t\"'`"))
            if left and right:
                out.append((left, right))
    return dedupe_pairs(out)


def extract_last_quoted_or_code(prompt: str) -> Optional[str]:
    candidates = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
    if candidates:
        return candidates[-1].strip()
    code_like = re.findall(r"`([^`]+)`", prompt)
    if code_like:
        return code_like[-1].strip()
    return None


def extract_binary_query(prompt: str) -> Optional[str]:
    bins_ = re.findall(r"\b[01]{8}\b", prompt)
    return bins_[-1] if bins_ else None


def extract_number_query(prompt: str) -> Optional[str]:
    nums = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])", prompt)
    return nums[-1] if nums else None


def parse_sample(sample_id: str, prompt: str, answer: str) -> ParsedSample:
    family = classify_family(prompt)
    support_examples = find_arrow_examples(prompt)
    query_input: Optional[str] = None

    if family == "bit_binary":
        query_input = extract_binary_query(prompt)
    elif family in {"roman_numeral", "unit_conversion", "gravity"}:
        query_input = extract_number_query(prompt)
    elif family in {"text_decrypt", "equation"}:
        query_input = extract_last_quoted_or_code(prompt)

    return ParsedSample(
        sample_id=sample_id,
        family=family,
        prompt=prompt,
        answer=answer,
        support_examples=support_examples,
        query_input=query_input,
    )


def load_train_csv(csv_path: str) -> List[ParsedSample]:
    rows: List[ParsedSample] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(parse_sample(row["id"], row["prompt"], row.get("answer", "")))
    return rows


def analyze_dataset(samples: List[ParsedSample]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(samples),
        "by_family": {},
        "example_counts": {},
    }
    for s in samples:
        summary["by_family"].setdefault(s.family, 0)
        summary["by_family"][s.family] += 1

        n = len(s.support_examples)
        summary["example_counts"].setdefault(s.family, {})
        summary["example_counts"][s.family].setdefault(n, 0)
        summary["example_counts"][s.family][n] += 1
    return summary


def build_synth_records(size_map: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for family, n in size_map.items():
        spec = SPEC_REGISTRY[family]
        records = []
        for _ in range(int(n)):
            cfg = spec.sample_config()
            records.append(make_sft_record(family, cfg, style=choose_style()))
        out[family] = records
    return out


def cmd_analyze(args):
    samples = load_train_csv(args.csv)
    summary = analyze_dataset(samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_synth_all(args):
    size_map = json.loads(args.sizes)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records_by_family = build_synth_records(size_map)
    for family, records in records_by_family.items():
        out_path = outdir / f"{family}.jsonl"
        write_jsonl(records, str(out_path))
        print(f"Wrote {len(records)} records to {out_path}")


def cmd_make_mixture(args):
    merge_jsonl(args.inputs, args.out, shuffle=True, seed=args.seed)
    print(f"Merged {len(args.inputs)} files into {args.out}")


def cmd_export_chatml(args):
    export_chatml(args.input, args.out)
    print(f"Exported ChatML dataset to {args.out}")


def build_parser():
    import argparse

    p = argparse.ArgumentParser(description="Nemotron toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--csv", required=True)
    p_analyze.set_defaults(func=cmd_analyze)

    p_synth_all = sub.add_parser("synth_all")
    p_synth_all.add_argument("--outdir", required=True)
    p_synth_all.add_argument("--sizes", required=True)
    p_synth_all.set_defaults(func=cmd_synth_all)

    p_mix = sub.add_parser("make_mixture")
    p_mix.add_argument("--out", required=True)
    p_mix.add_argument("--seed", type=int, default=42)
    p_mix.add_argument("inputs", nargs="+")
    p_mix.set_defaults(func=cmd_make_mixture)

    p_export = sub.add_parser("export_chatml")
    p_export.add_argument("--input", required=True)
    p_export.add_argument("--out", required=True)
    p_export.set_defaults(func=cmd_export_chatml)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)