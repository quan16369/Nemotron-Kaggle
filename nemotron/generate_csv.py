import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
BYTE_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def load_vocab():
    vocab = {}
    with open(BASE_DIR / "vocab.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            vocab[str(entry["token_id"])] = entry["token"]
    return vocab


def load_problems():
    problems = {}
    # problems.jsonl has category/status; detail files have answer/question/examples
    with open(BASE_DIR / "problems.jsonl") as f:
        for line in f:
            p = json.loads(line)
            problems[p["id"]] = p
    # Merge in detail from individual problem files
    for pid in problems:
        detail_path = BASE_DIR / "problems" / f"{pid}.jsonl"
        if detail_path.exists():
            with open(detail_path) as f:
                detail = json.loads(f.readline())
            problems[pid].update(detail)
    return problems


def load_generations():
    generations = {}
    with open(BASE_DIR / "generation.jsonl") as f:
        for line in f:
            g = json.loads(line)
            generations[g["id"]] = g
    return generations


def decode_tokens(token_ids, vocab):
    parts = []
    byte_buffer = bytearray()
    for tid in token_ids:
        text = vocab.get(str(tid), f"<unk:{tid}>")
        # Check if token is a byte sequence (single or multi like <0xE2><0x89>)
        byte_matches = BYTE_RE.findall(text)
        if byte_matches and BYTE_RE.sub("", text) == "":
            for hex_str in byte_matches:
                byte_buffer.append(int(hex_str, 16))
            continue
        # Flush accumulated bytes before a non-byte token
        if byte_buffer:
            parts.append(byte_buffer.decode("utf-8", errors="replace"))
            byte_buffer.clear()
        parts.append(text)
    # Flush remaining bytes
    if byte_buffer:
        parts.append(byte_buffer.decode("utf-8", errors="replace"))
    return "".join(parts)


def read_raw_tokens(problem_id, run_name):
    raw_path = BASE_DIR / "raw" / problem_id / run_name
    if not raw_path.exists():
        return [], []
    prompt_token_ids = []
    gen_token_ids = []
    with open(raw_path) as f:
        for line in f:
            t = json.loads(line)
            if t["logprob"] is None:
                prompt_token_ids.append(t["token_id"])
            else:
                gen_token_ids.append(t["token_id"])
    return prompt_token_ids, gen_token_ids


def main():
    vocab = load_vocab()
    problems = load_problems()
    generations = load_generations()

    output_path = BASE_DIR / "dataset.csv"
    fieldnames = [
        "id",
        "category",
        "question",
        "answer",
        "num_examples",
        "run_id",
        "num_prompt_tokens",
        "num_gen_tokens",
        "avg_logprob",
        "prompt_text",
        "raw_text",
        "extracted_answer",
        "is_correct",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pid, gen in sorted(generations.items()):
            prob = problems.get(pid, {})
            for run in gen.get("runs", []):
                prompt_ids, gen_ids = read_raw_tokens(pid, run["run"])
                prompt_text = decode_tokens(prompt_ids, vocab)
                gen_text = decode_tokens(gen_ids, vocab)

                writer.writerow(
                    {
                        "id": pid,
                        "category": prob.get("category", ""),
                        "question": prob.get("question", ""),
                        "answer": prob.get("answer", ""),
                        "num_examples": len(prob.get("examples", [])),
                        "run_id": run["run"],
                        "num_prompt_tokens": run.get("num_prompt_tokens", ""),
                        "num_gen_tokens": run.get("num_gen_tokens", ""),
                        "avg_logprob": run.get("avg_logprob", ""),
                        "prompt_text": prompt_text,
                        "raw_text": gen_text,
                        "extracted_answer": run.get("extracted_answer", ""),
                        "is_correct": run.get("correct", ""),
                    }
                )

    print(f"Written {output_path}")


if __name__ == "__main__":
    main()
