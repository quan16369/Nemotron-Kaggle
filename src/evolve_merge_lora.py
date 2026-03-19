from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.metric import read_records, score_predictions


@dataclass
class Candidate:
    weights: List[float]
    score: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--history-path")
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--mutation-scale", type=float, default=0.15)
    parser.add_argument("--combination-type", default="svd")
    parser.add_argument("--svd-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--prompt-column", default="user_message")
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def normalize_positive(weights: Sequence[float], eps: float = 1e-8) -> List[float]:
    clipped = [max(eps, float(w)) for w in weights]
    total = sum(clipped)
    return [w / total for w in clipped]


def candidate_key(weights: Sequence[float], places: int = 6) -> Tuple[float, ...]:
    return tuple(round(float(weight), places) for weight in weights)


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_holdout_records(path: str, max_samples: Optional[int], seed: int) -> List[Dict[str, str]]:
    records = read_records(path)
    if max_samples is None or len(records) <= max_samples:
        return records

    rng = random.Random(seed)
    sampled = records[:]
    rng.shuffle(sampled)
    return sampled[:max_samples]


def build_quant_config(load_in_4bit: bool):
    from transformers import BitsAndBytesConfig
    import torch

    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def first_model_device(model):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def render_generation_prompt(
    tokenizer,
    user_message: str,
    disable_thinking: bool,
) -> str:
    messages = [{"role": "user", "content": user_message}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return f"<|user|>\n{user_message}\n<|assistant|>\n"


def batched(items: Sequence[Dict[str, str]], batch_size: int) -> Iterable[Sequence[Dict[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def generate_predictions(
    model,
    tokenizer,
    records: Sequence[Dict[str, str]],
    id_column: str,
    prompt_column: str,
    batch_size: int,
    max_new_tokens: int,
    disable_thinking: bool,
) -> List[Dict[str, str]]:
    import torch

    device = first_model_device(model)
    predictions: List[Dict[str, str]] = []

    for batch in batched(records, batch_size):
        prompts = [
            render_generation_prompt(
                tokenizer=tokenizer,
                user_message=record[prompt_column],
                disable_thinking=disable_thinking,
            )
            for record in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for record, output, input_length in zip(batch, outputs, input_lengths):
            new_tokens = output[int(input_length):]
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            predictions.append({
                id_column: str(record[id_column]),
                "prediction": raw_text,
            })

    return predictions


def select_parent(population: Sequence[Candidate], rng: random.Random) -> Candidate:
    pool_size = max(2, len(population) // 2)
    contenders = rng.sample(list(population[:pool_size]), k=min(3, pool_size))
    return max(contenders, key=lambda candidate: candidate.score or float("-inf"))


def crossover(parent_a: Candidate, parent_b: Candidate, rng: random.Random) -> List[float]:
    mix = rng.random()
    child = [
        mix * weight_a + (1.0 - mix) * weight_b
        for weight_a, weight_b in zip(parent_a.weights, parent_b.weights)
    ]
    return normalize_positive(child)


def mutate(weights: Sequence[float], mutation_scale: float, rng: random.Random) -> List[float]:
    mutated = [weight + rng.gauss(0.0, mutation_scale) for weight in weights]
    return normalize_positive(mutated)


def initial_population(num_adapters: int, population_size: int, rng: random.Random) -> List[Candidate]:
    seeds: List[List[float]] = []

    for idx in range(num_adapters):
        one_hot = [0.0] * num_adapters
        one_hot[idx] = 1.0
        seeds.append(one_hot)

    seeds.append([1.0 / num_adapters] * num_adapters)

    while len(seeds) < population_size:
        dirichlet = [rng.gammavariate(1.0, 1.0) for _ in range(num_adapters)]
        seeds.append(normalize_positive(dirichlet))

    return [Candidate(weights=weights) for weights in seeds[:population_size]]


def maybe_delete_adapter(model, adapter_name: str) -> None:
    if hasattr(model, "delete_adapter"):
        model.delete_adapter(adapter_name)


def evaluate_candidate(
    candidate: Candidate,
    model,
    tokenizer,
    adapter_names: Sequence[str],
    holdout_records: Sequence[Dict[str, str]],
    args: argparse.Namespace,
    eval_cache: Dict[Tuple[float, ...], float],
    history_fp,
    generation_index: int,
    candidate_index: int,
) -> float:
    key = candidate_key(candidate.weights)
    if key in eval_cache:
        candidate.score = eval_cache[key]
        return candidate.score

    temp_adapter_name = f"evo_merge_g{generation_index}_c{candidate_index}"
    model.add_weighted_adapter(
        adapters=list(adapter_names),
        weights=list(candidate.weights),
        adapter_name=temp_adapter_name,
        combination_type=args.combination_type,
        **({"svd_rank": args.svd_rank} if args.combination_type == "svd" else {}),
    )
    model.set_adapter(temp_adapter_name)

    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        records=holdout_records,
        id_column=args.id_column,
        prompt_column=args.prompt_column,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        disable_thinking=args.disable_thinking,
    )
    accuracy, _ = score_predictions(
        truth_records=holdout_records,
        pred_records=predictions,
        id_column=args.id_column,
        answer_column=args.answer_column,
        prediction_column="prediction",
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )

    candidate.score = accuracy
    eval_cache[key] = accuracy

    if history_fp is not None:
        history_fp.write(json.dumps(
            {
                "generation": generation_index,
                "candidate_index": candidate_index,
                "weights": candidate.weights,
                "score": accuracy,
            }
        ) + "\n")
        history_fp.flush()

    fallback_adapter = adapter_names[0]
    model.set_adapter(fallback_adapter)
    maybe_delete_adapter(model, temp_adapter_name)
    gc.collect()
    torch = __import__("torch")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return accuracy


def save_best_adapter(
    model,
    adapter_names: Sequence[str],
    weights: Sequence[float],
    args: argparse.Namespace,
) -> None:
    final_adapter_name = "evo_best"
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    model.add_weighted_adapter(
        adapters=list(adapter_names),
        weights=list(weights),
        adapter_name=final_adapter_name,
        combination_type=args.combination_type,
        **({"svd_rank": args.svd_rank} if args.combination_type == "svd" else {}),
    )
    model.set_adapter(final_adapter_name)
    model.save_pretrained(out_dir, selected_adapters=[final_adapter_name])

    metadata = {
        "weights": list(weights),
        "combination_type": args.combination_type,
        "svd_rank": args.svd_rank,
        "adapters": list(adapter_names),
    }
    (out_dir / "merge_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_model_and_adapters(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_pad_token(tokenizer)

    quant_config = build_quant_config(args.load_in_4bit)
    model_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    adapter_paths = [str(Path(path)) for path in args.adapters]
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_paths[0],
        adapter_name="adapter_0",
        is_trainable=False,
    )
    adapter_names = ["adapter_0"]

    for idx, adapter_path in enumerate(adapter_paths[1:], start=1):
        adapter_name = f"adapter_{idx}"
        peft_model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
        adapter_names.append(adapter_name)

    peft_model.eval()
    return peft_model, tokenizer, adapter_names


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    holdout_records = load_holdout_records(
        path=args.truth,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    model, tokenizer, adapter_names = load_model_and_adapters(args)

    history_fp = None
    if args.history_path:
        history_path = Path(args.history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_fp = history_path.open("w", encoding="utf-8")

    eval_cache: Dict[Tuple[float, ...], float] = {}
    population = initial_population(
        num_adapters=len(adapter_names),
        population_size=max(args.population_size, len(adapter_names) + 1),
        rng=rng,
    )

    best_candidate: Optional[Candidate] = None

    for generation in range(args.generations):
        for candidate_index, candidate in enumerate(population):
            evaluate_candidate(
                candidate=candidate,
                model=model,
                tokenizer=tokenizer,
                adapter_names=adapter_names,
                holdout_records=holdout_records,
                args=args,
                eval_cache=eval_cache,
                history_fp=history_fp,
                generation_index=generation,
                candidate_index=candidate_index,
            )

        population.sort(key=lambda item: item.score or float("-inf"), reverse=True)
        if best_candidate is None or (population[0].score or -math.inf) > (best_candidate.score or -math.inf):
            best_candidate = Candidate(weights=list(population[0].weights), score=population[0].score)

        print(json.dumps(
            {
                "generation": generation,
                "best_score": population[0].score,
                "best_weights": population[0].weights,
            },
            indent=2,
        ))

        elites = [Candidate(weights=list(item.weights), score=item.score) for item in population[:args.elite_size]]
        next_population = elites[:]

        while len(next_population) < args.population_size:
            parent_a = select_parent(population, rng)
            parent_b = select_parent(population, rng)
            child_weights = crossover(parent_a, parent_b, rng)
            child_weights = mutate(child_weights, mutation_scale=args.mutation_scale, rng=rng)
            next_population.append(Candidate(weights=child_weights))

        population = next_population

    if best_candidate is None:
        raise RuntimeError("Evolutionary search did not evaluate any candidates")

    save_best_adapter(
        model=model,
        adapter_names=adapter_names,
        weights=best_candidate.weights,
        args=args,
    )

    summary = {
        "best_score": best_candidate.score,
        "best_weights": best_candidate.weights,
        "out_dir": args.out_dir,
        "sample_count": len(holdout_records),
    }
    print(json.dumps(summary, indent=2))

    if history_fp is not None:
        history_fp.close()


if __name__ == "__main__":
    main()
