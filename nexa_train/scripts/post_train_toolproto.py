#!/usr/bin/env python3
"""Fire-and-forget helper for post-training the scientific assistant."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import List

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-train the scientific assistant with tool usage.")
    parser.add_argument("--project-slug", default="scientific_assistant")
    parser.add_argument("--project-name", default="Scientific Assistant")
    parser.add_argument("--experiment-name", default="toolproto_posttrain_v1")
    parser.add_argument("--output-dir", default="artifacts/scientific_assistant/runs")

    parser.add_argument("--repo-id", default="Allanatrix/Nexa_Sci_distilled_Falcon-10B")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Enable 4-bit loading via bitsandbytes; disabled by default to favor BF16 training.",
    )
    parser.add_argument(
        "--tokenized-dataset-dir",
        default="data/processed/scientific_assistant/distillation/sft_datasets/tokenized/falcon3_10b_v1",
    )

    parser.add_argument(
        "--train-dataset",
        default="data/processed/scientific_assistant/tool_protocol/sft_toolproto_v1_train.jsonl",
    )
    parser.add_argument(
        "--val-dataset",
        default="data/processed/scientific_assistant/tool_protocol/sft_toolproto_v1_validation.jsonl",
    )
    parser.add_argument(
        "--test-dataset",
        default="data/processed/scientific_assistant/tool_protocol/sft_toolproto_v1_test.jsonl",
    )

    parser.add_argument("--wandb-project", default="Nexa_Sci_Assistant")
    parser.add_argument("--wandb-run-name", default=None)

    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--per-device-train-bs", type=int, default=1)
    parser.add_argument("--per-device-eval-bs", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=16)

    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0005)

    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=int, default=120)

    parser.add_argument("--sync-s3-uri", default="s3://nexacompute/ML_Checkpoints/")
    parser.add_argument("--post-sync-cmd", default=None)
    parser.add_argument("--notify-webhook", default=None)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA adapters and fine-tune all parameters.")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of module names to apply LoRA to.",
    )

    return parser.parse_args()


def notify(webhook: str | None, payload: dict) -> None:
    if not webhook or requests is None:
        return
    try:  # pragma: no cover - best effort
        resp = requests.post(webhook, json=payload, timeout=10)
        print(f"[post-train] webhook response: {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        print(f"[post-train] webhook failed: {exc}", file=sys.stderr)


def sync_to_s3(uri: str | None, run_dir: str, checkpoint: str | None, experiment: str) -> None:
    if not uri:
        return
    target_prefix = f"{uri.rstrip('/')}/{experiment}/"
    print(f"[post-train] Syncing run directory to {target_prefix}")
    subprocess.run(["aws", "s3", "sync", run_dir, target_prefix], check=False)
    if checkpoint:
        subprocess.run(["aws", "s3", "cp", checkpoint, f"{target_prefix}best_checkpoint/"], check=False)


def run_post_sync(cmd: str | None, run_dir: str) -> None:
    if not cmd:
        return
    print(f"[post-train] Running post-sync command: {cmd}")
    subprocess.run(shlex.split(cmd) + [run_dir], check=False)


def load_tokenized_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer) -> DatasetDict:
    tokenized_dir = os.fspath(os.path.expanduser(args.tokenized_dataset_dir))
    if os.path.isdir(tokenized_dir):
        print(f"[post-train] Loading tokenized dataset from {tokenized_dir}")
        dataset = load_from_disk(tokenized_dir)
    else:
        data_files: dict[str, str] = {}
        if os.path.isfile(args.train_dataset):
            data_files["train"] = args.train_dataset
        if os.path.isfile(args.val_dataset):
            data_files["validation"] = args.val_dataset
        if os.path.isfile(args.test_dataset):
            data_files["test"] = args.test_dataset

        if not data_files:
            raise FileNotFoundError("No dataset files found. Provide --tokenized-dataset-dir or JSONL files.")

        print(f"[post-train] Tokenizing raw JSONL datasets: {data_files}")
        raw_ds = load_dataset("json", data_files=data_files)
        max_length = args.max_seq_length

        def _format_example(example: dict) -> dict:
            prompt = example.get("prompt", "")
            response = example.get("response", "")
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            try:
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                rendered = f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n"
            encoded = tokenizer(
                rendered,
                truncation=True,
                max_length=max_length,
            )
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded

        dataset = raw_ds.map(_format_example, remove_columns=raw_ds["train"].column_names)

    if "labels" not in dataset["train"].column_names:
        def _add_labels(example: dict) -> dict:
            return {"labels": example["input_ids"][:]}

        dataset = dataset.map(_add_labels)

    def _truncate_features(example: dict) -> dict:
        max_length = args.max_seq_length
        for key in ("input_ids", "attention_mask", "labels"):
            if key in example:
                example[key] = example[key][:max_length]
        return example

    dataset = dataset.map(_truncate_features)
    return dataset


def prepare_model(args: argparse.Namespace) -> AutoModelForCausalLM:
    torch_dtype = getattr(torch, args.torch_dtype) if isinstance(args.torch_dtype, str) else args.torch_dtype
    model_kwargs: dict = {"torch_dtype": torch_dtype}
    if args.device_map and args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )
    if not args.no_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required for LoRA fine-tuning. Install it or pass --no-lora to disable LoRA.")
        target_modules = [module.strip() for module in args.lora_target_modules.split(",") if module.strip()]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM if TaskType is not None else "CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        try:
            model.print_trainable_parameters()  # type: ignore[attr-defined]
        except AttributeError:
            pass
    return model


def main() -> None:
    args = parse_args()
    os.environ["WANDB_MODE"] = "online"
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_run_name:
        os.environ.setdefault("WANDB_RUN_NAME", args.wandb_run_name)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    run_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"[post-train] Run directory: {run_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id or 0)
    tokenizer.padding_side = "left"

    dataset = load_tokenized_dataset(args, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = prepare_model(args)
    model.resize_token_embeddings(len(tokenizer))
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.per_device_train_bs,
        per_device_eval_batch_size=args.per_device_eval_bs,
        gradient_accumulation_steps=args.grad_accumulation,
        do_eval=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=20,
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        report_to=["wandb", "tensorboard"],
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        label_names=["labels"],
        logging_dir=os.path.join(run_dir, "logs"),
        run_name=args.wandb_run_name,
        optim="adafactor",
        adafactor=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    attempt = 0
    metrics: dict[str, float] = {}
    last_error: Exception | None = None
    while attempt < max(1, args.max_retries):
        attempt += 1
        try:
            train_result = trainer.train()
            trainer.save_model(os.path.join(run_dir, "final_model"))
            metrics = train_result.metrics
            eval_metrics = trainer.evaluate() if dataset.get("validation") is not None else {}
            metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[post-train] Attempt {attempt} failed: {exc}", file=sys.stderr)
            if attempt >= args.max_retries:
                raise
            time.sleep(max(0, args.retry_backoff))

    if last_error is not None:
        raise last_error

    best_checkpoint = trainer.state.best_model_checkpoint
    print(f"Run directory: {run_dir}")
    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint}")
    if metrics:
        print(f"Metrics: {metrics}")

    sync_to_s3(
        args.sync_s3_uri,
        run_dir,
        best_checkpoint,
        args.experiment_name,
    )
    run_post_sync(args.post_sync_cmd, str(run_dir))

    notify(
        args.notify_webhook,
        {
            "status": "completed",
            "project": args.project_slug,
            "experiment": args.experiment_name,
            "run_dir": str(run_dir),
            "checkpoint": best_checkpoint,
        },
    )
    print("Post-training run completed.")


if __name__ == "__main__":
    main()
    