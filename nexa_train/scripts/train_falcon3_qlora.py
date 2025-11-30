#!/usr/bin/env python3
"""QLoRA fine-tuning pipeline for Falcon3-10B on the scientific SFT dataset."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
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
DEFAULT_OUTPUT_DIR = "artifacts/runs/falcon3_qlora_full"
PROMPT_TEMPLATE = """### Question:\n{prompt}\n\n### Answer:\n{response}\n"""


@dataclass
class SplitConfig:
    train_path: Path
    val_path: Path
    test_path: Optional[Path]
    max_train_samples: Optional[int]
    max_val_samples: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True, help="Path to train JSONL dataset.")
    parser.add_argument("--val-file", type=Path, required=True, help="Path to validation JSONL dataset.")
    parser.add_argument("--test-file", type=Path, default=None, help="Optional path to test JSONL dataset.")
    parser.add_argument("--tokenized-dir", type=Path, default=None, help="Optional path to pre-tokenized dataset saved via save_to_disk.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Base model identifier to fine-tune.")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Directory for trainer outputs.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Cap on train samples (0 uses full dataset).")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Cap on validation samples (0 uses full dataset).")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Cap on test samples (0 uses full dataset).")
    parser.add_argument("--max-length", type=int, default=2048, help="Tokenization max sequence length.")

    # Training loop configuration
    parser.add_argument("--num-train-epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--max-steps", type=int, default=1800, help="Override total training steps (-1 disables).")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4, help="Per-device train batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4, help="Per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=1.2e-4, help="Base learning rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.02, help="Warmup ratio for scheduler.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Explicit warmup steps (overrides ratio when >0).")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="Optimizer name for TrainingArguments.")
    parser.add_argument("--eval-steps", type=int, default=0, help="Evaluation frequency in steps (0 disables during training).")
    parser.add_argument("--save-steps", type=int, default=0, help="Checkpoint save frequency in steps (0 disables during training).")
    parser.add_argument("--save-strategy", type=str, default="no", help="Checkpoint save strategy (no/steps/epoch).")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Maximum checkpoints to keep.")
    parser.add_argument("--logging-steps", type=int, default=25, help="Training log frequency in steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--group-by-length", action="store_true", help="Enable smart batching by sequence length.")

    # LoRA configuration
    parser.add_argument("--lora-r", type=int, default=64, help="Rank for LoRA adapters.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="Alpha scaling for LoRA.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")

    # Precision / logging flags
    parser.add_argument("--bf16", type=str, default="true", help="Enable bf16 (true/false).")
    parser.add_argument("--fp16", type=str, default="false", help="Enable fp16 (true/false).")
    parser.add_argument("--allow-tf32", action="store_true", help="Enable TF32 matmul kernels on Ampere GPUs.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing in TrainingArguments.")
    parser.add_argument("--report-to", type=str, nargs="*", default=["wandb", "tensorboard"], help="Report to integrations (list).")
    parser.add_argument("--wandb-project", type=str, default="Nexa_Sci", help="Weights & Biases project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable Weights & Biases logging.")

    return parser.parse_args()


def str_to_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes", "y"}


def configure_wandb(project: str, run_name: Optional[str], disabled: bool) -> None:
    if disabled:
        os.environ.setdefault("WANDB_MODE", "disabled")
        return
    os.environ.setdefault("WANDB_PROJECT", project)
    if run_name:
        os.environ.setdefault("WANDB_NAME", run_name)
    os.environ.setdefault("WANDB_WATCH", "false")
    os.environ.setdefault("WANDB_SILENT", "true")


def load_sft_dataset(config: SplitConfig, max_test_samples: Optional[int]) -> DatasetDict:
    data_files: Dict[str, str] = {
        "train": str(config.train_path),
        "validation": str(config.val_path),
    }
    if config.test_path is not None:
        data_files["test"] = str(config.test_path)

    dataset = load_dataset("json", data_files=data_files)

    if config.max_train_samples and config.max_train_samples > 0:
        dataset["train"] = dataset["train"].select(range(min(config.max_train_samples, dataset["train"].num_rows)))
    if config.max_val_samples and config.max_val_samples > 0:
        dataset["validation"] = dataset["validation"].select(
            range(min(config.max_val_samples, dataset["validation"].num_rows))
        )
    if "test" in dataset and max_test_samples and max_test_samples > 0:
        dataset["test"] = dataset["test"].select(range(min(max_test_samples, dataset["test"].num_rows)))

    def format_example(example: Dict[str, str]) -> Dict[str, str]:
        prompt = example.get("prompt", "").strip()
        response = example.get("response", "").strip()
        example["text"] = PROMPT_TEMPLATE.format(prompt=prompt, response=response)
        return example

    columns_to_remove = [col for col in dataset["train"].column_names if col != "text"]
    dataset = dataset.map(format_example, remove_columns=columns_to_remove)
    return dataset


def prepare_tokenizer(model_name: str, max_length: int) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length
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

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch")
    return tokenized


def load_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float, allow_tf32: bool) -> AutoModelForCausalLM:
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
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def save_metrics(output_dir: Path, metrics: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def maybe_cap_split(dataset: DatasetDict, split: str, max_samples: Optional[int]) -> DatasetDict:
    if max_samples and max_samples > 0 and split in dataset:
        max_samples = min(max_samples, dataset[split].num_rows)
        dataset[split] = dataset[split].select(range(max_samples))
    return dataset


def main() -> None:
    args = parse_args()
    configure_wandb(args.wandb_project, args.wandb_run_name, args.disable_wandb)

    tokenizer = prepare_tokenizer(args.model, args.max_length)

    if args.tokenized_dir and args.tokenized_dir.exists():
        tokenized = DatasetDict.load_from_disk(str(args.tokenized_dir))
    else:
        split_config = SplitConfig(
            train_path=args.train_file,
            val_path=args.val_file,
            test_path=args.test_file,
            max_train_samples=None if args.max_train_samples <= 0 else args.max_train_samples,
            max_val_samples=None if args.max_val_samples <= 0 else args.max_val_samples,
        )
        dataset = load_sft_dataset(split_config, None if args.max_test_samples <= 0 else args.max_test_samples)
        tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)

    tokenized = maybe_cap_split(tokenized, "train", None if args.max_train_samples <= 0 else args.max_train_samples)
    tokenized = maybe_cap_split(tokenized, "validation", None if args.max_val_samples <= 0 else args.max_val_samples)
    tokenized = maybe_cap_split(tokenized, "test", None if args.max_test_samples <= 0 else args.max_test_samples)

    model = load_model(args.model, args.lora_r, args.lora_alpha, args.lora_dropout, args.allow_tf32)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    bf16 = str_to_bool(args.bf16) and torch.cuda.is_available()
    fp16 = str_to_bool(args.fp16) and torch.cuda.is_available()

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "overwrite_output_dir": True,
        "do_train": True,
        "do_eval": False,
        "eval_strategy": "no",
        "eval_steps": None,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": None,
        "save_total_limit": args.save_total_limit,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio if args.warmup_steps == 0 else 0.0,
        "warmup_steps": args.warmup_steps,
        "optim": args.optim,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": None if not args.report_to else args.report_to,
        "run_name": args.wandb_run_name,
        "seed": args.seed,
        "dataloader_pin_memory": True,
        "dataloader_num_workers": 4,
        "logging_dir": str(args.output_dir / "logs"),
        "save_safetensors": True,
        "group_by_length": args.group_by_length,
    }

    # Remove None-valued entries to avoid unsupported kwargs
    training_kwargs = {k: v for k, v in training_kwargs.items() if v is not None}
    if args.disable_wandb and training_kwargs.get("report_to"):
        training_kwargs["report_to"] = []

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()

    metrics: Dict[str, float] = {}
    eval_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
    metrics.update(eval_metrics)

    if "test" in tokenized:
        test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
        metrics.update(test_metrics)

    save_metrics(args.output_dir, metrics)
    print("Final metrics:", json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


