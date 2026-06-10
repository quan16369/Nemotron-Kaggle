"""Export rejection-sampling SFT data from selected-category pass@k rollouts.

The selected frontier contains only prompts where greedy decoding is wrong but
at least one sampled rollout is exactly correct. Among valid correct rollouts,
the shortest non-truncated completion is selected for continued SFT.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import pandas as pd


PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def extract_final_answer(text: str | None) -> str:
    if text is None:
        return "NOT_FOUND"
    boxed_starts = list(re.finditer(r"\\boxed\{", text))
    matches: list[str] = []
    for index, match in enumerate(boxed_starts):
        start = match.end()
        end = (
            boxed_starts[index + 1].start()
            if index + 1 < len(boxed_starts)
            else len(text)
        )
        segment = text[start:end]
        last_brace = segment.rfind("}")
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [match.strip() for match in matches if match.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else "NOT_FOUND"


def find_train_csv(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True))
    for path in candidates:
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if {"id", "prompt", "answer"}.issubset(columns):
            return path
    raise FileNotFoundError("Could not find train.csv with id,prompt,answer")


def is_bit_prompt(prompt: str) -> bool:
    lowered = str(prompt).lower()
    return "secret bit manipulation rule" in lowered and "8-bit binary" in lowered


def is_equation_numeric_deduce_prompt(prompt: str) -> bool:
    prompt = str(prompt)
    if "secret set of transformation rules is applied to equations" not in prompt.lower():
        return False
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    equation_lines = [
        line for line in lines if re.fullmatch(r"\d+\D\d+\s*=\s*-?\d+\D?", line)
    ]
    query_matches = re.findall(
        r"determine the result for:\s*(\d+)(\D)(\d+)", prompt, re.IGNORECASE
    )
    if not equation_lines or not query_matches:
        return False
    query_operator = query_matches[-1][1]
    example_operators = {
        match.group(1)
        for line in equation_lines
        if (match := re.fullmatch(r"\d+(\D)\d+\s*=\s*-?\d+\D?", line))
    }
    return query_operator in example_operators


def detect_selected_category(prompt: str, categories: set[str]) -> str | None:
    if "bit_manipulation" in categories and is_bit_prompt(prompt):
        return "bit_manipulation"
    if (
        "equation_numeric_deduce" in categories
        and is_equation_numeric_deduce_prompt(prompt)
    ):
        return "equation_numeric_deduce"
    return None


def verify_answer(gold: str, predicted: str) -> bool:
    gold = str(gold).strip()
    predicted = str(predicted).strip()
    if re.fullmatch(r"[01]+", gold):
        return predicted.lower() == gold.lower()
    try:
        return math.isclose(
            float(gold), float(predicted), rel_tol=1e-2, abs_tol=1e-5
        )
    except Exception:
        return predicted.lower() == gold.lower()


def format_prompt(tokenizer, prompt: str) -> str:
    user_content = str(prompt) + PROMPT_SUFFIX
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--output-dir", default="/kaggle/working/passk_sft")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["bit_manipulation", "equation_numeric_deduce"],
        default=["bit_manipulation"],
        help="Categories to mine. Pass one category to disable the other.",
    )
    parser.add_argument("--base-manifest", default=None)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=7680)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--min-completion-tokens", type=int, default=300)
    parser.add_argument("--max-selected-tokens", type=int, default=7000)
    parser.add_argument(
        "--min-correct-rollouts",
        type=int,
        default=2,
        help="Require this many independently sampled exact-gold rollouts before adding a prompt.",
    )
    parser.add_argument(
        "--max-selected-per-category",
        type=int,
        default=30,
        help="Cap added frontier rows per category. Use 0 for no cap.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0 for pass@k sampling")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = find_train_csv(args.train_csv)
    frame = pd.read_csv(train_csv)
    selected_categories = set(args.categories)
    frame["category"] = frame["prompt"].map(
        lambda prompt: detect_selected_category(prompt, selected_categories)
    )
    frame = frame[frame["category"].notna()].copy()
    frame = frame.sample(frac=1.0, random_state=args.sample_seed).reset_index(drop=True)
    if args.max_examples is not None:
        frame = frame.head(args.max_examples).copy()

    llm = LLM(
        model=args.base_model_path,
        tensor_parallel_size=1,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [format_prompt(tokenizer, prompt) for prompt in frame["prompt"]]
    lora_request = LoRARequest("adapter", 1, args.adapter_dir)

    greedy_outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        lora_request=lora_request,
    )
    sampled_outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n=args.k,
        ),
        lora_request=lora_request,
    )

    audit_rows: list[dict] = []
    selected_rows: list[dict] = []
    greedy_correct_count = 0
    passk_correct_count = 0
    category_stats = {
        category: {
            "examples": 0,
            "greedy_correct": 0,
            "passk_correct": 0,
            "frontier_selected": 0,
        }
        for category in args.categories
    }

    for row_index, (row, greedy, sampled) in enumerate(
        zip(frame.itertuples(index=False), greedy_outputs, sampled_outputs)
    ):
        gold = str(row.answer).strip()
        category = str(row.category)
        category_stats[category]["examples"] += 1
        greedy_text = greedy.outputs[0].text
        greedy_extracted = extract_final_answer(greedy_text)
        greedy_correct = verify_answer(gold, greedy_extracted)
        greedy_correct_count += int(greedy_correct)
        category_stats[category]["greedy_correct"] += int(greedy_correct)

        valid_candidates: list[dict] = []
        problem_candidates: list[dict] = []
        for sample_index, candidate in enumerate(sampled.outputs):
            text = candidate.text
            extracted = extract_final_answer(text)
            token_count = len(candidate.token_ids)
            finish_reason = str(candidate.finish_reason)
            correct = verify_answer(gold, extracted)
            exact_gold = extracted.strip().lower() == gold.strip().lower()
            valid = (
                exact_gold
                and "\\boxed{" in text
                and finish_reason != "length"
                and args.min_completion_tokens <= token_count <= args.max_selected_tokens
            )
            candidate_row = {
                "problem_id": str(row.id),
                "category": category,
                "sample_index": sample_index,
                "answer": gold,
                "greedy_extracted": greedy_extracted,
                "greedy_correct": greedy_correct,
                "extracted": extracted,
                "correct": correct,
                "exact_gold": exact_gold,
                "valid_for_sft": valid,
                "completion_token_count": token_count,
                "finish_reason": finish_reason,
                "completion": text,
            }
            audit_rows.append(candidate_row)
            problem_candidates.append(candidate_row)
            if valid:
                valid_candidates.append(candidate_row)

        passk_correct = any(candidate["correct"] for candidate in problem_candidates)
        passk_correct_count += int(passk_correct)
        category_stats[category]["passk_correct"] += int(passk_correct)
        if greedy_correct or len(valid_candidates) < args.min_correct_rollouts:
            continue

        # Avoid selecting an unusually short lucky answer. Pick the lower median
        # exact-gold trajectory, which remains compact but is more representative.
        valid_candidates.sort(key=lambda candidate: candidate["completion_token_count"])
        chosen = valid_candidates[(len(valid_candidates) - 1) // 2]
        selected_rows.append(
            {
                "problem_id": f"{row.id}-passk-sft",
                "source_problem_id": str(row.id),
                "category": category,
                "prompt": str(row.prompt),
                "formatted_prompt": prompts[row_index],
                "completion": chosen["completion"],
                "answer": gold,
                "completion_token_count": chosen["completion_token_count"],
                "sample_index": chosen["sample_index"],
                "exact_correct_rollouts": len(valid_candidates),
            }
        )
        category_stats[category]["frontier_selected"] += 1

    if args.max_selected_per_category > 0:
        capped_rows: list[dict] = []
        for category in args.categories:
            category_rows = [
                row for row in selected_rows if row["category"] == category
            ]
            category_rows.sort(
                key=lambda row: (
                    -row["exact_correct_rollouts"],
                    row["completion_token_count"],
                    row["problem_id"],
                )
            )
            capped_rows.extend(category_rows[: args.max_selected_per_category])
            category_stats[category]["frontier_selected_after_cap"] = min(
                len(category_rows), args.max_selected_per_category
            )
        selected_rows = capped_rows
    else:
        for category in args.categories:
            category_stats[category]["frontier_selected_after_cap"] = category_stats[
                category
            ]["frontier_selected"]

    pd.DataFrame(audit_rows).to_csv(output_dir / "audit.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected.csv", index=False)
    write_jsonl(output_dir / "selected.jsonl", selected_rows)

    manifest_added = 0
    if args.base_manifest:
        output_manifest = Path(
            args.output_manifest or output_dir / "manifest_with_passk_sft.csv"
        )
        base_manifest = pd.read_csv(args.base_manifest)
        manifest_rows: list[dict] = []
        eos_token_id = tokenizer.eos_token_id
        for selected in selected_rows:
            prompt_ids = tokenizer.encode(
                selected["formatted_prompt"], add_special_tokens=False
            )
            completion_ids = tokenizer.encode(
                selected["completion"], add_special_tokens=False
            )
            if eos_token_id is not None and (
                not completion_ids or completion_ids[-1] != eos_token_id
            ):
                completion_ids.append(eos_token_id)
            input_ids = prompt_ids + completion_ids
            if len(input_ids) > args.max_model_len:
                continue
            manifest_rows.append(
                {
                    "problem_id": selected["problem_id"],
                    "source_problem_id": selected["source_problem_id"],
                    "category": selected["category"],
                    "segment": "passk_sft.jsonl",
                    "num_loss_tokens": len(completion_ids),
                    "completion_token_count": len(completion_ids),
                    "token_count": len(input_ids),
                    "input_ids_json": json.dumps(input_ids),
                    "mask_json": json.dumps(
                        [0] * len(prompt_ids) + [1] * len(completion_ids)
                    ),
                }
            )
        manifest_added = len(manifest_rows)
        merged_manifest = pd.concat(
            [base_manifest, pd.DataFrame(manifest_rows)], ignore_index=True
        )
        merged_manifest.to_csv(output_manifest, index=False)

    summary = {
        "train_csv": train_csv,
        "categories": args.categories,
        "examples": len(frame),
        "k": args.k,
        "greedy_correct": greedy_correct_count,
        "greedy_accuracy": greedy_correct_count / len(frame) if len(frame) else 0.0,
        "passk_correct": passk_correct_count,
        "passk_accuracy": passk_correct_count / len(frame) if len(frame) else 0.0,
        "frontier_selected": len(selected_rows),
        "manifest_rows_added": manifest_added,
        "output_dir": str(output_dir),
    }
    for stats in category_stats.values():
        count = stats["examples"]
        stats["greedy_accuracy"] = stats["greedy_correct"] / count if count else 0.0
        stats["passk_accuracy"] = stats["passk_correct"] / count if count else 0.0
        stats["gap"] = stats["passk_accuracy"] - stats["greedy_accuracy"]
    summary["category_stats"] = category_stats
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
