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
from src.dsl.mini_dsl import (
    generate_configs_from_spec,
    render_records_from_configs,
    synth_from_spec,
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


def _append_synth_record(
    family: str,
    cfg: Dict[str, Any],
    bucket: List[Dict[str, Any]],
    stats: Dict[str, int],
    allow_failed_verify: bool,
) -> None:
    rec = make_sft_record(family, cfg, style=choose_style())
    stats["generated"] += 1
    if rec["meta"].get("verifier_pass", False) or allow_failed_verify:
        bucket.append(rec)
    else:
        stats["verify_failed"] += 1


def _trim_family_records(
    all_records: List[Dict[str, Any]],
    target_n: int,
    mix: Dict[str, float],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if len(all_records) <= target_n:
        return all_records

    final_sizes = allocate_counts(target_n, mix)
    grouped = {"easy": [], "medium": [], "hard": []}
    for rec in all_records:
        grouped[rec["meta"]["config"]["difficulty"]].append(rec)

    final: List[Dict[str, Any]] = []
    for diff in ["easy", "medium", "hard"]:
        rng.shuffle(grouped[diff])
        final.extend(grouped[diff][: final_sizes[diff]])
    return final


def build_synth_records(
    size_map,
    difficulty_mix=None,
    oversample_factor=1.5,
    seed=42,
    allow_failed_verify=False,
):
    rng = random.Random(seed)

    default_mix = {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    difficulty_mix = difficulty_mix or {}

    out = {}
    stats: Dict[str, Dict[str, int]] = {}

    for family, target_n in size_map.items():
        spec = SPEC_REGISTRY[family]
        mix = difficulty_mix.get(family, default_mix)
        stats[family] = {
            "generated": 0,
            "verify_failed": 0,
        }

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
                _append_synth_record(
                    family=family,
                    cfg=cfg,
                    bucket=bucket_data[diff],
                    stats=stats[family],
                    allow_failed_verify=allow_failed_verify,
                )

        # merge
        all_records = []
        for v in bucket_data.values():
            all_records.extend(v)

        all_records = _trim_family_records(
            all_records=all_records,
            target_n=target_n,
            mix=mix,
            rng=rng,
        )

        rng.shuffle(all_records)
        out[family] = all_records[:target_n]

    return out, stats


BOXED_RE = re.compile(r"\\\\boxed\{(.*)\}\s*$")


def normalize_answer_text(answer: str) -> str:
    text = answer.strip()
    m = BOXED_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text.strip()


def _init_repro_summary(total_rows: int) -> Dict[str, Any]:
    return {
        "total_rows": total_rows,
        "matched": 0,
        "mismatched": 0,
        "skipped": 0,
        "errors": 0,
        "by_family": {},
        "details": [],
    }


def _ensure_family_stats(summary: Dict[str, Any], family: str) -> Dict[str, int]:
    return summary["by_family"].setdefault(
        family,
        {"matched": 0, "mismatched": 0, "skipped": 0, "errors": 0},
    )


def _record_status(
    summary: Dict[str, Any],
    fam_stats: Dict[str, int],
    status: str,
    detail: Dict[str, Any],
) -> None:
    if status not in {"matched", "mismatched", "skipped", "errors"}:
        raise ValueError(f"Invalid status: {status}")
    summary[status] += 1
    fam_stats[status] += 1
    summary["details"].append(detail)


def _build_reproduce_input(parsed: ParsedSample) -> Dict[str, Any]:
    return {
        "sample_id": parsed.sample_id,
        "family": parsed.family,
        "prompt": parsed.prompt,
        "answer": parsed.answer,
        "support_examples": parsed.support_examples,
        "query_input": parsed.query_input,
    }


def _try_reproduce(spec, parsed: ParsedSample) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    try:
        cfg = spec.reproduce(_build_reproduce_input(parsed))
        return cfg, None, None
    except NotImplementedError:
        return None, "skipped", "reproduce_not_implemented"
    except Exception as exc:
        return None, "errors", str(exc)


def _compare_prediction(spec, cfg: Dict[str, Any], row: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
    pred = normalize_answer_text(spec.solve(cfg))
    gold = normalize_answer_text(row.get("answer", ""))
    status = "matched" if pred == gold else "mismatched"
    return status, {
        "pred": pred,
        "gold": gold,
        "reproduction_score": cfg.get("reproduction_score"),
    }


def _process_reproduction_row(summary: Dict[str, Any], row: Dict[str, str]) -> None:
    parsed = parse_sample(row)
    family = parsed.family
    fam_stats = _ensure_family_stats(summary, family)

    if family not in SPEC_REGISTRY:
        _record_status(
            summary,
            fam_stats,
            "skipped",
            {
                "id": row.get("id"),
                "family": family,
                "status": "skipped",
                "reason": "unknown_family",
            },
        )
        return

    spec = SPEC_REGISTRY[family]
    cfg, repro_status, repro_info = _try_reproduce(spec, parsed)
    if repro_status is not None:
        _record_status(
            summary,
            fam_stats,
            repro_status,
            {
                "id": row.get("id"),
                "family": family,
                "status": "skipped" if repro_status == "skipped" else "error",
                "reason": repro_info if repro_status == "skipped" else None,
                "error": repro_info if repro_status == "errors" else None,
            },
        )
        return

    try:
        status, payload = _compare_prediction(spec, cfg, row)
        _record_status(
            summary,
            fam_stats,
            status,
            {
                "id": row.get("id"),
                "family": family,
                "status": status,
                **payload,
            },
        )
    except Exception as exc:
        _record_status(
            summary,
            fam_stats,
            "errors",
            {
                "id": row.get("id"),
                "family": family,
                "status": "error",
                "error": str(exc),
            },
        )


def run_reproduction_validation(
    rows: List[Dict[str, str]],
    max_rows: Optional[int] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    checked_rows = rows[:max_rows] if max_rows else rows
    summary: Dict[str, Any] = _init_repro_summary(len(checked_rows))

    for row in checked_rows:
        _process_reproduction_row(summary, row)

    if strict and (summary["mismatched"] > 0 or summary["errors"] > 0):
        raise RuntimeError(
            "Reproduction validation failed in strict mode: "
            f"mismatched={summary['mismatched']} errors={summary['errors']}"
        )

    return summary


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

    data, stats = build_synth_records(
        size_map=size_map,
        difficulty_mix=mix,
        oversample_factor=args.oversample_factor,
        seed=args.seed,
        allow_failed_verify=args.allow_failed_verify,
    )

    for family, records in data.items():
        write_jsonl(records, str(outdir / f"{family}.jsonl"))
        print(
            json.dumps(
                {
                    "family": family,
                    "written": len(records),
                    "generated": stats[family]["generated"],
                    "verify_failed": stats[family]["verify_failed"],
                }
            )
        )


def cmd_validate_reproduction(args):
    with open(args.train_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    summary = run_reproduction_validation(
        rows,
        max_rows=args.max_rows,
        strict=args.strict,
    )

    if args.out_report:
        out_path = Path(args.out_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "total_rows": summary["total_rows"],
                "matched": summary["matched"],
                "mismatched": summary["mismatched"],
                "skipped": summary["skipped"],
                "errors": summary["errors"],
                "by_family": summary["by_family"],
                "out_report": args.out_report,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


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


def cmd_dsl_gen_configs(args):
    summary = generate_configs_from_spec(
        spec_path=args.spec,
        count=args.count,
        out_path=args.out,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_dsl_render_configs(args):
    style_mix = json.loads(args.style_mix) if args.style_mix else None
    summary = render_records_from_configs(
        spec_path=args.spec,
        configs_path=args.configs,
        out_path=args.out,
        strict_verify=args.strict_verify,
        style_mix=style_mix,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_dsl_synth(args):
    summary = synth_from_spec(
        spec_path=args.spec,
        count=args.count,
        out_path=args.out,
        seed=args.seed,
        strict_verify=args.strict_verify,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


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
    s.add_argument("--allow-failed-verify", action="store_true")
    s.set_defaults(func=cmd_synth_all)

    r = sub.add_parser("validate_reproduction")
    r.add_argument("--train-csv", required=True)
    r.add_argument("--out-report")
    r.add_argument("--max-rows", type=int)
    r.add_argument("--strict", action="store_true")
    r.set_defaults(func=cmd_validate_reproduction)

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

    dgc = sub.add_parser("dsl_gen_configs")
    dgc.add_argument("--spec", required=True)
    dgc.add_argument("--out", required=True)
    dgc.add_argument("--count", type=int, required=True)
    dgc.add_argument("--seed", type=int, default=42)
    dgc.set_defaults(func=cmd_dsl_gen_configs)

    drc = sub.add_parser("dsl_render_configs")
    drc.add_argument("--spec", required=True)
    drc.add_argument("--configs", required=True)
    drc.add_argument("--out", required=True)
    drc.add_argument("--seed", type=int, default=42)
    drc.add_argument("--style-mix")
    drc.add_argument("--strict-verify", action="store_true")
    drc.set_defaults(func=cmd_dsl_render_configs)

    ds = sub.add_parser("dsl_synth")
    ds.add_argument("--spec", required=True)
    ds.add_argument("--out", required=True)
    ds.add_argument("--count", type=int, required=True)
    ds.add_argument("--seed", type=int, default=42)
    ds.add_argument("--strict-verify", action="store_true")
    ds.set_defaults(func=cmd_dsl_synth)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
