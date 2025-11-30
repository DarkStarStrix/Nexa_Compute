#!/usr/bin/env python3
"""Smoke-test QLoRA fine-tuning for Falcon3-10B on the scientific SFT dataset."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_NAME = "tiiuae/Falcon3-10B-Base"
DEFAULT_OUTPUT_DIR = "artifacts/runs/falcon3_qlora_smoke"
TEMPLATE = """### Question:\n{prompt}\n\n### Answer:\n{response}\n"""


@dataclass
class SplitConfig:
    train_path: Path
    val_path: Path
    max_train_samples: int = 500
    max_val_samples: int = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/distillation/sft_datasets/sft_scientific_v1_train.jsonl"),
        help="Path to the training JSONL dataset.",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("data/processed/distillation/sft_datasets/sft_scientific_v1_validation.jsonl"),
        help="Path to the validation JSONL dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base model identifier to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory for Trainer outputs (checkpoints, logs).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=500,
        help="Number of training samples for the smoke test.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=200,
        help="Number of validation samples for the smoke test.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="Nexa_Sci",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="falcon3-qlora-smoke",
        help="Optional explicit W&B run name.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps to reach an effective batch size that fits memory.",
    )
    return parser.parse_args()


def configure_wandb(project: str, run_name: str | None) -> None:
    os.environ.setdefault("WANDB_PROJECT", project)
    if run_name:
        os.environ.setdefault("WANDB_NAME", run_name)
    os.environ.setdefault("WANDB_WATCH", "false")
    os.environ.setdefault("WANDB_SILENT", "true")


def load_sft_dataset(config: SplitConfig) -> DatasetDict:
    data_files: Dict[str, str] = {
        "train": str(config.train_path),
        "validation": str(config.val_path),
    }
    dataset = load_dataset("json", data_files=data_files)

    if config.max_train_samples > 0:
        dataset["train"] = dataset["train"].select(range(min(config.max_train_samples, dataset["train"].num_rows)))
    if config.max_val_samples > 0:
        dataset["validation"] = dataset["validation"].select(
            range(min(config.max_val_samples, dataset["validation"].num_rows))
        )

    def format_example(example: Dict[str, str]) -> Dict[str, str]:
        prompt = example.get("prompt", "").strip()
        response = example.get("response", "").strip()
        example["text"] = TEMPLATE.format(prompt=prompt, response=response)
        return example

    dataset = dataset.map(format_example, remove_columns=[col for col in dataset["train"].column_names if col != "text"])
    return dataset


def prepare_tokenizer(model_name: str, max_length: int) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int) -> DatasetDict:
    def tokenize_fn(batch: Dict[str, Iterable[str]]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format(type="torch")
    return tokenized


def load_model(model_name: str) -> AutoModelForCausalLM:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def main() -> None:
    args = parse_args()
    configure_wandb(args.wandb_project, args.wandb_run_name)

    split_config = SplitConfig(
        train_path=args.train_file,
        val_path=args.val_file,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    dataset = load_sft_dataset(split_config)
    tokenizer = prepare_tokenizer(args.model, args.max_length)
    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)

    model = load_model(args.model)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        max_steps=60,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        bf16=torch.cuda.is_available(),
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=5,
        log_level="info",
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="steps",
        save_steps=60,
        save_total_limit=1,
        report_to=["wandb", "tensorboard"],
        run_name=args.wandb_run_name,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()


