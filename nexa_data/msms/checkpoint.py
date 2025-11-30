"""Checkpoint and resume functionality for pipeline."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set


class CheckpointManager:
    """Manages pipeline checkpoints for resume capability."""

    def __init__(self, checkpoint_dir: Path, run_id: str):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            run_id: Run identifier
        """
        self.checkpoint_dir = checkpoint_dir
        self.run_id = run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = checkpoint_dir / f"checkpoint_{run_id}.json"

    def save(
        self,
        processed_samples: int,
        last_sample_id: str,
        shard_index: int,
        processed_sample_ids: Set[str],
        metrics: Dict,
    ) -> None:
        """Save checkpoint.

        Args:
            processed_samples: Number of samples processed
            last_sample_id: Last processed sample ID
            shard_index: Current shard index
            processed_sample_ids: Set of all processed sample IDs
            metrics: Current metrics dictionary
        """
        checkpoint = {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processed_samples": processed_samples,
            "last_sample_id": last_sample_id,
            "shard_index": shard_index,
            "processed_sample_ids": list(processed_sample_ids),
            "metrics": metrics,
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load(self) -> Optional[Dict]:
        """Load checkpoint if exists.

        Returns:
            Checkpoint dictionary or None
        """
        if not self.checkpoint_file.exists():
            return None

        with open(self.checkpoint_file) as f:
            return json.load(f)

    def clear(self) -> None:
        """Clear checkpoint file."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

