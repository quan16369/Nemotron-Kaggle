"""Export the safe winning-snapshot-plus-delta training set to one CSV manifest.

Each CSV row is a single training record and contains exact token IDs plus mask,
so training can be reproduced from the CSV alone without shipping the full
snapshot directory structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

try:
    from tokenizers import Tokenizer  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional local dependency
    Tokenizer = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - optional local dependency
    AutoTokenizer = None  # type: ignore[assignment]

from winning_snapshot_delta import (
    build_current_correct_base_records,
    load_snapshot_records,
    merge_snapshot_with_current_delta,
)

BASE_DIR = Path(__file__).parent
DEFAULT_SNAPSHOT_DIR = BASE_DIR / "training" / "sft" / "04-08-16-14"
DEFAULT_OUTPUT = BASE_DIR / "winning_snapshot_delta_manifest.csv"
TOKENIZER_JSON = BASE_DIR / "tokenizer.json"
MAX_SEQ_LEN = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
        help="Path to the winning snapshot directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV manifest path",
    )
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Export the raw winning snapshot only",
    )
    parser.add_argument(
        "--use-existing-reasoning-files",
        action="store_true",
        help="Prefer reasoning/*.txt over regenerating with current code when building delta rows",
    )
    parser.add_argument(
        "--chat-tokenizer-path",
        type=str,
        default=None,
        help=(
            "Local path for the Nemotron chat tokenizer/model directory. "
            "If omitted, the script will try common Kaggle input paths."
        ),
    )
    return parser.parse_args()


def resolve_chat_tokenizer_path(explicit: str | None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)

    env_value = os.environ.get("NEMOTRON_CHAT_TOKENIZER_PATH")
    if env_value:
        candidates.append(env_value)

    candidates.extend(
        [
            "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
            "/kaggle/input/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
            "/kaggle/input/Nemotron-3-Nano-30B-A3B-BF16/transformers/default/1",
        ]
    )

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "Could not find a local Nemotron chat tokenizer path. "
        "Pass --chat-tokenizer-path or set NEMOTRON_CHAT_TOKENIZER_PATH."
    )


def main() -> None:
    args = parse_args()

    snapshot_records, snapshot_config = load_snapshot_records(
        args.snapshot_dir,
        max_seq_len=MAX_SEQ_LEN,
    )
    final_records = snapshot_records
    delta_stats = None

    if not args.no_delta:
        if Tokenizer is None or AutoTokenizer is None:
            raise ModuleNotFoundError(
                "tokenizers and transformers are required to export the delta manifest"
            )

        chat_tokenizer_path = resolve_chat_tokenizer_path(args.chat_tokenizer_path)
        chat_tokenizer = AutoTokenizer.from_pretrained(
            chat_tokenizer_path,
            trust_remote_code=True,
        )
        completion_tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
        current_correct_base_records = build_current_correct_base_records(
            repo_dir=BASE_DIR,
            chat_tokenizer=chat_tokenizer,
            completion_tokenizer=completion_tokenizer,
            max_seq_len=MAX_SEQ_LEN,
            use_existing_reasoning_files=args.use_existing_reasoning_files,
        )
        final_records, delta_stats = merge_snapshot_with_current_delta(
            snapshot_records,
            current_correct_base_records,
        )

    fieldnames = [
        "problem_id",
        "source_problem_id",
        "category",
        "segment",
        "num_loss_tokens",
        "completion_token_count",
        "token_count",
        "input_ids_json",
        "mask_json",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in final_records:
            writer.writerow(
                {
                    "problem_id": record["problem_id"],
                    "source_problem_id": record["source_problem_id"],
                    "category": record["category"],
                    "segment": record["segment"],
                    "num_loss_tokens": record["num_loss_tokens"],
                    "completion_token_count": record["completion_token_count"],
                    "token_count": len(record["input_ids"]),
                    "input_ids_json": json.dumps(record["input_ids"]),
                    "mask_json": json.dumps(
                        [1 if label != -100 else 0 for label in record["labels"]]
                    ),
                }
            )

    print(
        json.dumps(
            {
                "snapshot_dir": str(args.snapshot_dir),
                "snapshot_examples": len(snapshot_records),
                "final_examples": len(final_records),
                "output": str(args.output),
                "chat_tokenizer_path": (
                    None if args.no_delta else resolve_chat_tokenizer_path(args.chat_tokenizer_path)
                ),
                "snapshot_loss_config": snapshot_config.get("loss_config", {}),
                "snapshot_lr_schedule": snapshot_config.get("lr_schedule", {}),
                "delta_stats": (
                    None
                    if delta_stats is None
                    else {
                        "replaced_source_problems": delta_stats.replaced_source_problems,
                        "replaced_training_records": delta_stats.replaced_training_records,
                        "added_source_problems": delta_stats.added_source_problems,
                        "added_training_records": delta_stats.added_training_records,
                    }
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
