"""Hugging Face training runner with NexaCompute conventions."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pynvml
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.core.storage import generate_run_id, get_storage
from nexa_compute.training.smoothing import LossTracker


MODEL_CHOICES = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "albert-base-v2",
]


DATASET_CHOICES = [
    ("glue", "sst2"),
    ("glue", "cola"),
    ("imdb", None),
    ("ag_news", None),
]


@dataclass
class HFTrainingConfig:
    model: Optional[str] = None
    dataset: Optional[str] = None
    dataset_config: Optional[str] = None
    max_length: int = 256
    train_samples: int = 8000
    eval_samples: int = 2000
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 4
    grad_accumulation: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 5
    num_workers: int = 4
    seed: int = 42
    no_wandb: bool = False
    wandb_project: str = "nexa-compute"
    wandb_run_name: Optional[str] = None
    telemetry_interval: int = 10
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    allow_tf32: bool = False
    tags: Optional[List[str]] = None
    s3_uri: Optional[str] = os.environ.get("NEXA_S3_PREFIX")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robust HF training entrypoint for NexaCompute")
    parser.add_argument("--model", type=str, default=None, help="HF model identifier (default: sampled)")
    parser.add_argument("--dataset", type=str, help="HF dataset name")
    parser.add_argument("--dataset-config", type=str, default=None, help="Optional dataset config/split")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length")
    parser.add_argument("--train-samples", type=int, default=8000, help="Training sample cap (0=full)")
    parser.add_argument("--eval-samples", type=int, default=2000, help="Eval sample cap (0=full)")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Per-device eval batch size")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs")
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Base learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--eval-steps", type=int, default=50, help="Eval interval in steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint interval in steps")
    parser.add_argument("--save-total-limit", type=int, default=5, help="Max checkpoints to retain")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="nexa-compute", help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional explicit W&B run name")
    parser.add_argument("--telemetry-interval", type=int, default=10, help="GPU telemetry log interval")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 if GPUs available")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 if GPUs available")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--allow-tf32", action="store_true", help="Enable TF32 matmul on Ampere GPUs")
    parser.add_argument("--tags", type=str, nargs="*", default=None, help="Optional W&B tags")
    parser.add_argument(
        "--s3-uri",
        type=str,
        default=os.environ.get("NEXA_S3_PREFIX"),
        help="Optional S3 destination for checkpoints",
    )
    return parser


def parse_args(argv: Optional[List[str]] = None) -> HFTrainingConfig:
    parser = build_arg_parser()
    ns = parser.parse_args(argv)
    return HFTrainingConfig(
        model=ns.model,
        dataset=ns.dataset,
        dataset_config=ns.dataset_config,
        max_length=ns.max_length,
        train_samples=ns.train_samples,
        eval_samples=ns.eval_samples,
        batch_size=ns.batch_size,
        eval_batch_size=ns.eval_batch_size,
        epochs=ns.epochs,
        grad_accumulation=ns.grad_accumulation,
        learning_rate=ns.learning_rate,
        warmup_ratio=ns.warmup_ratio,
        weight_decay=ns.weight_decay,
        max_grad_norm=ns.max_grad_norm,
        logging_steps=ns.logging_steps,
        eval_steps=ns.eval_steps,
        save_steps=ns.save_steps,
        save_total_limit=ns.save_total_limit,
        num_workers=ns.num_workers,
        seed=ns.seed,
        no_wandb=ns.no_wandb,
        wandb_project=ns.wandb_project,
        wandb_run_name=ns.wandb_run_name,
        telemetry_interval=ns.telemetry_interval,
        fp16=ns.fp16,
        bf16=ns.bf16,
        gradient_checkpointing=ns.gradient_checkpointing,
        allow_tf32=ns.allow_tf32,
        tags=ns.tags,
        s3_uri=ns.s3_uri,
    )


def _select_model_and_dataset(cfg: HFTrainingConfig) -> Tuple[str, Tuple[str, Optional[str]]]:
    if cfg.model and cfg.dataset:
        return cfg.model, (cfg.dataset, cfg.dataset_config)
    model = cfg.model or random.choice(MODEL_CHOICES)
    if cfg.dataset:
        dataset_tuple = (cfg.dataset, cfg.dataset_config)
    else:
        dataset_tuple = random.choice(DATASET_CHOICES)
    return model, dataset_tuple


def _prepare_dataset(
    dataset_info: Tuple[str, Optional[str]],
    tokenizer,
    max_length: int,
    train_limit: int,
    eval_limit: int,
) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset], int]:
    dataset_name, dataset_config = dataset_info
    dataset = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)

    train_columns = dataset["train"].column_names
    text_col = "text"
    if "sentence" in train_columns:
        text_col = "sentence"
    elif train_columns:
        text_col = train_columns[0]
    label_col = "label" if "label" in train_columns else train_columns[-1]

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True)
    num_labels = len(set(dataset["train"][label_col]))

    train_dataset = tokenized["train"].rename_column(label_col, "labels")
    eval_dataset = tokenized.get("validation") or tokenized.get("test")
    if eval_dataset is not None:
        eval_dataset = eval_dataset.rename_column(label_col, "labels")

    feature_keep = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    train_remove = [col for col in train_dataset.column_names if col not in feature_keep]
    if train_remove:
        train_dataset = train_dataset.remove_columns(train_remove)
    if eval_dataset is not None:
        eval_remove = [col for col in eval_dataset.column_names if col not in feature_keep]
        if eval_remove:
            eval_dataset = eval_dataset.remove_columns(eval_remove)

    if train_limit and train_limit > 0:
        train_dataset = train_dataset.select(range(min(train_limit, len(train_dataset))))
    if eval_dataset is not None and eval_limit and eval_limit > 0:
        eval_dataset = eval_dataset.select(range(min(eval_limit, len(eval_dataset))))

    return train_dataset, eval_dataset, num_labels


class LossSmoothingCallback(TrainerCallback):
    def __init__(self, tracker: LossTracker):
        self.tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs is None or "loss" not in logs:
            return
        smoothed = self.tracker.update(logs["loss"])
        logs.update(smoothed)
        if state.global_step and state.global_step % 50 == 0:
            logs.update({f"loss_stats/{k}": v for k, v in self.tracker.get_stats().items()})


class GpuTelemetryCallback(TrainerCallback):
    def __init__(self, interval: int = 10):
        self.interval = max(1, interval)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if logs is None:
            return
        if not torch.cuda.is_available() or state.global_step % self.interval != 0:
            return
        metrics = {}
        for idx in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(idx) / (1024**3)
            metrics[f"gpu/{idx}/mem_allocated_gb"] = round(allocated, 3)
            metrics[f"gpu/{idx}/mem_reserved_gb"] = round(reserved, 3)
        logs.update(metrics)


def _format_gpu_info() -> list[dict[str, str | float]]:
    if not torch.cuda.is_available():
        return []
    info = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        info.append(
            {
                "id": idx,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
            }
        )
    return info


def _snapshot_gpu_utilisation() -> list[dict[str, float]]:
    stats: list[dict[str, float]] = []
    if not torch.cuda.is_available():
        return stats
    try:
        pynvml.nvmlInit()
        for idx in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats.append(
                {
                    "device": int(idx),
                    "gpu_util_percent": float(util.gpu),
                    "mem_util_percent": float(util.memory),
                    "mem_used_gb": round(mem.used / (1024**3), 3),
                    "mem_total_gb": round(mem.total / (1024**3), 3),
                }
            )
    except pynvml.NVMLError as exc:  # pragma: no cover
        print(f"⚠️ NVML snapshot failed: {exc}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass
    return stats


def _collect_system_fingerprint(gpu_info: list[dict[str, str | float]]) -> dict:
    env_keys = [
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "TORCH_DISTRIBUTED_DEBUG",
        "OMP_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    ]
    env_snapshot = {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None}
    return {
        "python": sys.version.split(" ")[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_topology": gpu_info,
        "env": env_snapshot,
    }


def run_training(cfg: HFTrainingConfig) -> dict:
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    run_id = generate_run_id("train")
    print("=" * 70)
    print(f"🚀 NexaCompute Training Run :: {run_id}")
    print("=" * 70)

    if cfg.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(cfg.seed)

    storage = get_storage()
    scratch_ckpt_dir = storage.run_dir_scratch(run_id)
    durable_ckpt_dir = storage.run_dir_durable(run_id)
    scratch_log_dir = storage.scratch("logs_temp", run_id)
    scratch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    scratch_log_dir.mkdir(parents=True, exist_ok=True)

    print("\n📁 Storage configuration:")
    print(f"   Scratch:  {scratch_ckpt_dir}")
    print(f"   Durable:  {durable_ckpt_dir}")
    print(f"   Logs:     {scratch_log_dir}")

    model_name, dataset_info = _select_model_and_dataset(cfg)
    print(f"\n🎯 Model: {model_name}")
    print(f"🗂️  Dataset: {dataset_info[0]}" + (f"/{dataset_info[1]}" if dataset_info[1] else ""))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds, eval_ds, num_labels = _prepare_dataset(
        dataset_info,
        tokenizer,
        cfg.max_length,
        cfg.train_samples,
        cfg.eval_samples,
    )

    print(f"\n📥 Loading model: {model_name} (labels={num_labels})")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if cfg.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    gpu_info = _format_gpu_info()
    if gpu_info:
        print("   GPU inventory:")
        for g in gpu_info:
            print(f"     • cuda:{g['id']} :: {g['name']} ({g['total_memory_gb']} GB)")
    else:
        print("   Running on CPU")

    train_size = len(train_ds)
    eval_size = len(eval_ds) if eval_ds is not None else 0

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch = cfg.batch_size * world_size * cfg.grad_accumulation
    steps_per_epoch = math.ceil(train_size / max(1, effective_batch))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    loss_tracker = LossTracker(smoothing_method="ema", alpha=0.95)

    if cfg.no_wandb:
        os.environ.setdefault("WANDB_MODE", "disabled")

    wandb_run = None
    if not cfg.no_wandb:
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or f"{run_id}-{model_name}",
            tags=cfg.tags,
            config={
                "run_id": run_id,
                "model": model_name,
                "dataset": dataset_info[0],
                "dataset_config": dataset_info[1],
                "num_labels": num_labels,
                "train_samples": train_size,
                "eval_samples": eval_size,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "grad_accumulation": cfg.grad_accumulation,
                "learning_rate": cfg.learning_rate,
                "warmup_steps": warmup_steps,
                "warmup_ratio": cfg.warmup_ratio,
                "weight_decay": cfg.weight_decay,
                "max_grad_norm": cfg.max_grad_norm,
                "gpu_info": gpu_info,
            },
        )

    training_args = TrainingArguments(
        output_dir=str(scratch_ckpt_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        gradient_accumulation_steps=cfg.grad_accumulation,
        logging_dir=str(scratch_log_dir),
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False,
        report_to=([] if cfg.no_wandb else ["wandb"]),
        fp16=cfg.fp16 and torch.cuda.is_available(),
        bf16=cfg.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=cfg.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    callbacks = [LossSmoothingCallback(loss_tracker), GpuTelemetryCallback(cfg.telemetry_interval)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("\n⚙️ Training configuration:")
    print(f"   Epochs: {cfg.epochs}")
    print(f"   Batch size (per device): {cfg.batch_size}")
    print(f"   Effective batch size: {effective_batch}")
    print(f"   Total steps: ~{total_steps} (warmup: {warmup_steps})")
    print(f"   Eval every {cfg.eval_steps} steps | Save every {cfg.save_steps} steps")

    print("\n🏋️ Starting training...")
    print(f"   Train samples: {train_size}")
    if eval_ds is not None:
        print(f"   Eval samples: {eval_size}")

    gpu_util_before = _snapshot_gpu_utilisation()
    train_result = trainer.train()
    gpu_util_after = _snapshot_gpu_utilisation()

    final_stats = loss_tracker.get_stats()
    print("\n📈 Final loss statistics:")
    for key, value in final_stats.items():
        if value is None:
            print(f"   {key}: N/A")
        else:
            print(f"   {key}: {value:.6f}")

    eval_metrics = None
    if eval_ds is not None:
        print("\n📊 Evaluating best model...")
        eval_metrics = trainer.evaluate()
        print(f"   Eval metrics: {json.dumps(eval_metrics, indent=2)}")
        if wandb_run is not None:
            wandb.log(
                {
                    **{f"final/{k}": v for k, v in final_stats.items() if v is not None},
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                }
            )

    print("\n💾 Saving model to scratch storage...")
    trainer.save_model(str(scratch_ckpt_dir / "final"))

    print("\n📦 Syncing artifacts to durable storage...")
    durable_ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_src = scratch_ckpt_dir / "final"
    final_dst = durable_ckpt_dir / "final"
    if final_src.exists():
        import shutil

        if final_src.is_dir():
            shutil.copytree(final_src, final_dst, dirs_exist_ok=True)
        else:
            shutil.copy2(final_src, final_dst)
        print(f"   ✅ Checkpoint synced: {final_dst}")

    system_fp = _collect_system_fingerprint(gpu_info)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": model_name,
        "dataset": dataset_info[0],
        "dataset_config": dataset_info[1],
        "num_labels": num_labels,
        "train_samples": train_size,
        "eval_samples": eval_size,
        "training_args": {
            "epochs": cfg.epochs,
            "batch_size_per_device": cfg.batch_size,
            "grad_accumulation": cfg.grad_accumulation,
            "learning_rate": cfg.learning_rate,
            "warmup_steps": warmup_steps,
            "warmup_ratio": cfg.warmup_ratio,
            "weight_decay": cfg.weight_decay,
            "max_grad_norm": cfg.max_grad_norm,
        },
        "hardware": gpu_info,
        "train_result": train_result.metrics,
        "loss_stats": final_stats,
        "eval_metrics": eval_metrics,
        "wandb_run_id": wandb_run.id if wandb_run is not None else None,
        "gpu_util_start": gpu_util_before,
        "gpu_util_end": gpu_util_after,
        "system": system_fp,
        "checkpoint_scratch": str(scratch_ckpt_dir),
        "checkpoint_durable": str(durable_ckpt_dir),
    }

    manifest_path = storage.manifest_path(run_id)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"   ✅ Manifest saved: {manifest_path}")

    if cfg.s3_uri:
        try:
            final_destination = cfg.s3_uri.rstrip("/") + f"/{run_id}"
            sync_cmd = ["aws", "s3", "sync", str(final_dst), final_destination]
            print(f"\n☁️  Syncing checkpoint to {final_destination}...")
            subprocess.run(sync_cmd, check=True)
        except Exception as s3_error:  # pragma: no cover
            print(f"⚠️ S3 sync failed: {s3_error}")

    try:
        prune_cmd = [
            "find",
            str(storage.durable("checkpoints")),
            "-type",
            "f",
            "-size",
            "+2G",
            "-mtime",
            "+3",
            "-delete",
        ]
        subprocess.run(prune_cmd, check=False)
    except Exception as pruning_error:  # pragma: no cover
        print(f"⚠️ Prune step failed: {pruning_error}")

    if wandb_run is not None:
        wandb.finish()

    print("\n✅ Training complete!")
    print("=" * 70)

    return {"run_id": run_id, "manifest": manifest, "manifest_path": manifest_path}


def cli(argv: Optional[List[str]] = None) -> dict:
    cfg = parse_args(argv)
    return run_training(cfg)


__all__ = [
    "HFTrainingConfig",
    "build_arg_parser",
    "parse_args",
    "run_training",
    "cli",
]


