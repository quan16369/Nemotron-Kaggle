"""Export rejection-sampling SFT data from bit-manipulation pass@k rollouts.

The selected frontier contains only prompts where greedy decoding is wrong but
at least one sampled rollout is exactly correct. Among valid correct rollouts,
the shortest non-truncated completion is selected for continued SFT.
"""

from __future__ import annotations

import argparse
import glob
import json
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
    binary = re.findall(r"[01]{8}", text)
    return binary[-1] if binary else "NOT_FOUND"


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
    parser.add_argument("--output-dir", default="/kaggle/working/bit_passk_sft")
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
    frame = frame[frame["prompt"].map(is_bit_prompt)].copy()
    frame = frame[frame["answer"].astype(str).str.fullmatch(r"[01]{8}")].copy()
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

    for row_index, (row, greedy, sampled) in enumerate(
        zip(frame.itertuples(index=False), greedy_outputs, sampled_outputs)
    ):
        gold = str(row.answer).strip()
        greedy_text = greedy.outputs[0].text
        greedy_extracted = extract_final_answer(greedy_text)
        greedy_correct = greedy_extracted == gold
        greedy_correct_count += int(greedy_correct)

        valid_candidates: list[dict] = []
        for sample_index, candidate in enumerate(sampled.outputs):
            text = candidate.text
            extracted = extract_final_answer(text)
            token_count = len(candidate.token_ids)
            finish_reason = str(candidate.finish_reason)
            correct = extracted == gold
            valid = (
                correct
                and "\\boxed{" in text
                and finish_reason != "length"
                and args.min_completion_tokens <= token_count <= args.max_selected_tokens
            )
            candidate_row = {
                "problem_id": str(row.id),
                "category": "bit_manipulation",
                "sample_index": sample_index,
                "answer": gold,
                "greedy_extracted": greedy_extracted,
                "greedy_correct": greedy_correct,
                "extracted": extracted,
                "correct": correct,
                "valid_for_sft": valid,
                "completion_token_count": token_count,
                "finish_reason": finish_reason,
                "completion": text,
            }
            audit_rows.append(candidate_row)
            if valid:
                valid_candidates.append(candidate_row)

        passk_correct = any(candidate["correct"] for candidate in audit_rows[-args.k :])
        passk_correct_count += int(passk_correct)
        if greedy_correct or not valid_candidates:
            continue

        chosen = min(valid_candidates, key=lambda candidate: candidate["completion_token_count"])
        selected_rows.append(
            {
                "problem_id": f"{row.id}-passk-sft",
                "source_problem_id": str(row.id),
                "category": "bit_manipulation",
                "prompt": str(row.prompt),
                "formatted_prompt": prompts[row_index],
                "completion": chosen["completion"],
                "answer": gold,
                "completion_token_count": chosen["completion_token_count"],
                "sample_index": chosen["sample_index"],
            }
        )

    pd.DataFrame(audit_rows).to_csv(output_dir / "audit.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected.csv", index=False)
    write_jsonl(output_dir / "selected.jsonl", selected_rows)

    manifest_added = 0
    if args.base_manifest:
        output_manifest = Path(
            args.output_manifest or output_dir / "manifest_with_bit_passk_sft.csv"
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
        "bit_examples": len(frame),
        "k": args.k,
        "greedy_correct": greedy_correct_count,
        "greedy_accuracy": greedy_correct_count / len(frame) if len(frame) else 0.0,
        "passk_correct": passk_correct_count,
        "passk_accuracy": passk_correct_count / len(frame) if len(frame) else 0.0,
        "frontier_selected": len(selected_rows),
        "manifest_rows_added": manifest_added,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
