"""Storage path utilities following NexaCompute storage policy."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class StoragePaths:
    """Manages storage paths according to NexaCompute storage policy."""
    
    def __init__(
        self,
        scratch_root: Optional[str | Path] = None,
        durable_root: Optional[str | Path] = None,
        shared_root: Optional[str | Path] = None,
    ):
        # Use environment variables or defaults
        self.scratch_root = Path(scratch_root or os.getenv("NEXA_SCRATCH", "/workspace/tmp"))
        self.durable_root = Path(durable_root or os.getenv("NEXA_DURABLE", "/mnt/nexa_durable"))
        self.shared_root = Path(shared_root or os.getenv("NEXA_SHARED", "/workspace/shared"))
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        dirs = {
            self.scratch_root: [
                "dataloader_cache",
                "checkpoints_temp",
                "logs_temp",
                "wandb_offline",
            ],
            self.durable_root: [
                "datasets",
                "checkpoints",
                "evals/reports",
                "evals/outputs",
                "manifests",
            ],
            self.shared_root: [
                "common_datasets",
                "eval_prompts",
                "active_jobs",
            ],
        }
        
        for root, subdirs in dirs.items():
            root.mkdir(parents=True, exist_ok=True)
            for subdir in subdirs:
                (root / subdir).mkdir(parents=True, exist_ok=True)
    
    def scratch(self, *parts: str) -> Path:
        """Get ephemeral scratch path."""
        return self.scratch_root / Path(*parts)
    
    def durable(self, *parts: str) -> Path:
        """Get durable storage path."""
        return self.durable_root / Path(*parts)
    
    def shared(self, *parts: str) -> Path:
        """Get shared storage path."""
        return self.shared_root / Path(*parts)
    
    def run_dir_scratch(self, run_id: str) -> Path:
        """Get scratch directory for a run."""
        return self.scratch("checkpoints_temp", run_id)
    
    def run_dir_durable(self, run_id: str) -> Path:
        """Get durable directory for a run."""
        return self.durable("checkpoints", run_id)
    
    def dataset_path(self, dataset_name: str, version: Optional[str] = None) -> Path:
        """Get path for a dataset."""
        if version:
            return self.durable("datasets", f"{dataset_name}_v{version}.parquet")
        return self.durable("datasets", f"{dataset_name}.parquet")
    
    def manifest_path(self, run_id: str) -> Path:
        """Get manifest path for a run."""
        return self.durable("manifests", f"{run_id}.json")
    
    def eval_report_path(self, run_id: str) -> Path:
        """Get evaluation report path."""
        return self.durable("evals", "reports", f"leaderboard_{run_id}.parquet")
    
    def sync_checkpoint_to_durable(self, run_id: str, checkpoint_name: str = "final.pt") -> Path:
        """Sync checkpoint from scratch to durable storage."""
        scratch_checkpoint = self.run_dir_scratch(run_id) / checkpoint_name
        durable_checkpoint = self.run_dir_durable(run_id) / checkpoint_name
        
        if not scratch_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found in scratch: {scratch_checkpoint}")
        
        durable_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scratch_checkpoint, durable_checkpoint)
        
        return durable_checkpoint
    
    def sync_run_to_durable(self, run_id: str) -> None:
        """Sync entire run directory from scratch to durable."""
        scratch_run = self.run_dir_scratch(run_id)
        durable_run = self.run_dir_durable(run_id)
        
        if not scratch_run.exists():
            return
        
        durable_run.mkdir(parents=True, exist_ok=True)
        
        # Copy all files
        for item in scratch_run.iterdir():
            dest = durable_run / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)


def generate_run_id(prefix: str = "run") -> str:
    """Generate a run ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


# Global storage instance
_STORAGE: Optional[StoragePaths] = None


def get_storage() -> StoragePaths:
    """Get global storage instance."""
    global _STORAGE
    if _STORAGE is None:
        _STORAGE = StoragePaths()
    return _STORAGE

