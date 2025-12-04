"""Training entrypoints wrapping the core training pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nexa_compute.orchestration import TrainingPipeline  # type: ignore
from nexa_compute.config import load_config  # type: ignore
from nexa_compute.core.artifacts import ArtifactMeta, create_artifact  # type: ignore


_DEF_CHECKPOINT_ARTIFACT_DIR = "artifacts/checkpoints"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_training_job(config_path: Path, overrides: Optional[List[str]] = None) -> ArtifactMeta:
    config = load_config(config_path, overrides=overrides or [])
    pipeline = TrainingPipeline(config)
    artifacts = pipeline.run()

    print(f"[nexa-train] run complete at {artifacts.run_dir}")

    artifact_dir = config.output_directory() / _DEF_CHECKPOINT_ARTIFACT_DIR

    def _producer(tmp_dir: Path) -> ArtifactMeta:
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        hasher = hashlib.sha256()
        total_bytes = 0

        if artifacts.checkpoint and Path(artifacts.checkpoint).exists():
            checkpoint_path = Path(artifacts.checkpoint)
            target_checkpoint = tmp_dir / checkpoint_path.name
            if checkpoint_path.is_dir():
                shutil.copytree(checkpoint_path, target_checkpoint, dirs_exist_ok=True)
                for file_path in target_checkpoint.rglob("*"):
                    if file_path.is_file():
                        data = file_path.read_bytes()
                        hasher.update(data)
                        total_bytes += len(data)
            else:
                shutil.copy2(checkpoint_path, target_checkpoint)
                data = target_checkpoint.read_bytes()
                hasher.update(data)
                total_bytes += len(data)
        else:
            placeholder = tmp_dir / "checkpoint.txt"
            placeholder.write_text("No checkpoint emitted by pipeline\n", encoding="utf-8")
            data = placeholder.read_bytes()
            hasher.update(data)
            total_bytes += len(data)

        manifest_bytes = manifest_path.read_bytes()
        hasher.update(manifest_bytes)
        total_bytes += len(manifest_bytes)

        return ArtifactMeta(
            kind="checkpoint",
            uri=str(artifact_dir.resolve()),
            hash=f"sha256:{hasher.hexdigest()}",
            bytes=total_bytes,
            created_at=_now_utc(),
            inputs=[str(config_path)],
            labels={"source": "nexa_train.train"},
        )

    return create_artifact(artifact_dir, _producer)


def parse_args():
    parser = argparse.ArgumentParser(description="Nexa Training Pipeline CLI")
    
    # Configuration Modes
    parser.add_argument("--config-mode", choices=["v1", "v2", "v3"], help="Select base configuration mode")
    
    # Overrides
    parser.add_argument("--run-name", type=str, help="Override run name")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], help="Override model size")
    parser.add_argument("--dataset", type=str, help="Dataset URI")
    parser.add_argument("--dataset-version", type=str, help="Dataset version")
    parser.add_argument("--expected-hash", type=str, help="Expected dataset hash")
    parser.add_argument("--nodes", type=int, help="Number of nodes")
    parser.add_argument("--gpus-per-node", type=int, help="GPUs per node")
    parser.add_argument("--microbatch", type=int, help="Microbatch size")
    parser.add_argument("--global-batch", type=int, help="Global batch size")
    parser.add_argument("--grad-accum", type=int, help="Gradient accumulation steps")
    parser.add_argument("--steps", type=int, help="Total training steps")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--warmup", type=int, help="Warmup steps")
    parser.add_argument("--save-every", type=int, help="Checkpoint save interval")
    parser.add_argument("--eval-every", type=int, help="Evaluation interval")
    parser.add_argument("--val-split", type=str, default="val", help="Validation split name")
    parser.add_argument("--log-every", type=int, help="Logging interval")
    parser.add_argument("--activation-checkpointing", type=str, choices=["true", "false"], help="Enable activation checkpointing")
    parser.add_argument("--max-seq-len", type=int, help="Maximum sequence length")
    parser.add_argument("--local-cache", type=str, help="Local cache directory")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint output directory")
    parser.add_argument("--wandb-project", type=str, help="WandB project name")
    parser.add_argument("--wandb-run", type=str, help="WandB run name")
    parser.add_argument("--dry-run", type=str, choices=["true", "false"], help="Perform a dry run")
    
    return parser.parse_args()


def get_base_config_path(mode: str) -> Path:
    config_map = {
        "v1": "v1_stability.yaml",
        "v2": "v2_performance.yaml",
        "v3": "v3_full_train.yaml"
    }
    
    config_file = config_map.get(mode)
    if not config_file:
        raise ValueError(f"Unknown config mode: {mode}")
        
    # Assuming configs are in nexa_infra/configs relative to the project root
    # Current file is nexa_train/train.py -> ROOT is repo root
    return ROOT / "nexa_infra" / "configs" / config_file


def main():
    args = parse_args()
    
    if not args.config_mode:
        # If no config mode is specified, fall back to standard behavior or require arguments
        print("Error: --config-mode is required (v1, v2, or v3)")
        sys.exit(1)
        
    base_config_path = get_base_config_path(args.config_mode)
    
    if not base_config_path.exists():
        print(f"Error: Configuration file not found at {base_config_path}")
        sys.exit(1)
        
    # Construct overrides list based on provided arguments
    overrides = []
    
    if args.run_name:
        overrides.append(f"run.name={args.run_name}")
        if not args.wandb_run:
             overrides.append(f"run.wandb.run_name={args.run_name}")
    
    # Model Size Logic (simplistic mapping for example, extend as needed)
    if args.model_size == "small":
        overrides.append("model.d_model=2048")
        overrides.append("model.n_layers=24")
        overrides.append("model.n_heads=16")
    elif args.model_size == "medium":
        overrides.append("model.d_model=3072")
        overrides.append("model.n_layers=32")
        overrides.append("model.n_heads=24")
    elif args.model_size == "large":
        overrides.append("model.d_model=4096")
        overrides.append("model.n_layers=36")
        overrides.append("model.n_heads=32")
        
    if args.dataset:
        overrides.append(f"data.remote_uri={args.dataset}")
    if args.dataset_version:
        overrides.append(f"data.dataset_version={args.dataset_version}")
    if args.expected_hash:
        overrides.append(f"data.expected_dataset_hash={args.expected_hash}")
        
    if args.nodes:
        overrides.append(f"cluster.num_nodes={args.nodes}")
    if args.gpus_per_node:
        overrides.append(f"cluster.gpus_per_node={args.gpus_per_node}")
        
    if args.microbatch:
        overrides.append(f"training.microbatch_size_per_gpu={args.microbatch}")
    if args.global_batch:
        overrides.append(f"training.global_batch_size={args.global_batch}")
    if args.grad_accum:
        overrides.append(f"training.grad_accumulation_steps={args.grad_accum}")
    if args.steps:
        overrides.append(f"training.max_steps={args.steps}")
        
    if args.lr:
        overrides.append(f"optimizer.lr={args.lr}")
    if args.warmup:
        overrides.append(f"scheduler.warmup_steps={args.warmup}")
        
    if args.save_every:
        overrides.append(f"training.save_every_steps={args.save_every}")
        overrides.append(f"checkpointing.save_every_steps={args.save_every}")
    if args.eval_every:
        overrides.append(f"training.eval_every_steps={args.eval_every}")
    if args.log_every:
        overrides.append(f"training.log_every_steps={args.log_every}")
        
    if args.val_split:
        overrides.append(f"data.validation.split={args.val_split}")
        
    if args.activation_checkpointing:
        bool_val = "true" if args.activation_checkpointing == "true" else "false"
        overrides.append(f"fsdp.activation_checkpointing={bool_val}")
        
    if args.max_seq_len:
        overrides.append(f"model.max_seq_len={args.max_seq_len}")
        
    if args.local_cache:
        overrides.append(f"data.local_cache_root={args.local_cache}")
    if args.checkpoint_dir:
        overrides.append(f"checkpointing.local_tmp_dir={args.checkpoint_dir}")
        
    if args.wandb_project:
        overrides.append(f"run.wandb.project={args.wandb_project}")
    if args.wandb_run:
        overrides.append(f"run.wandb.run_name={args.wandb_run}")
        
    if args.dry_run:
        bool_val = "true" if args.dry_run == "true" else "false"
        overrides.append(f"preflight.dry_run_batch={bool_val}")

    print(f"Starting training with config mode: {args.config_mode}")
    print(f"Applying {len(overrides)} overrides.")
    
    # Execute training
    run_training_job(base_config_path, overrides=overrides)


if __name__ == "__main__":
    main()
