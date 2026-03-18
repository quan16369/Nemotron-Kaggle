import json
import argparse
from pathlib import Path

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/final_adapter")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser.parse_args()


def format_example(example):
    messages = example["messages"]
    user = messages[0]["content"]
    assistant = messages[1]["content"]
    return {
        "text": f"<|user|>\n{user}\n<|assistant|>\n{assistant}"
    }


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lora_cfg = cfg["lora"]

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.map(format_example)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=SFTConfig(
            output_dir=str(Path(args.output_dir).parent / "train_logs"),
            per_device_train_batch_size=lora_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=lora_cfg["gradient_accumulation_steps"],
            learning_rate=lora_cfg["learning_rate"],
            num_train_epochs=lora_cfg["epochs"],
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            warmup_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()