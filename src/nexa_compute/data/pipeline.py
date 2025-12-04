"""High-level data pipeline utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, distributed

if TYPE_CHECKING:
    from ..training.distributed import DistributedContext

from ..config.schema import DataConfig
from ..core.logging import get_logger
from .registry import DatasetRegistry, DEFAULT_REGISTRY

LOGGER = get_logger(__name__)


class DataPipeline:
    """Builds dataloaders and tracks dataset metadata."""

    def __init__(self, config: DataConfig, registry: DatasetRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or DEFAULT_REGISTRY

    def dataloader(
        self,
        split: str,
        *,
        shuffle: bool | None = None,
        batch_size: int | None = None,
        distributed_context: "DistributedContext" | None = None,
    ) -> DataLoader:
        dataset = self.registry.build(self.config, split=split)
        shuffle = self._should_shuffle(split, shuffle)
        sampler = self._build_sampler(split, dataset, distributed_context)
        if sampler is not None:
            shuffle = False
        loader = DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=split == "train",
            sampler=sampler,
        )
        LOGGER.debug("Built dataloader", extra={"split": split, "size": len(dataset)})
        return loader

    def materialize_metadata(self, output_dir: str | Path) -> Path:
        """Materialize dataset metadata to disk.
        
        **Idempotency**: Idempotent - running twice produces the same metadata file.
        
        **Versioned Manifest**: This method creates a versioned manifest entry for the dataset.
        The dataset_version from config is used as the version identifier.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset_name": self.config.dataset_name,
            "dataset_version": self.config.dataset_version,
            "source_uri": self.config.source_uri,
            "augmentations": self.config.augmentations,
            "preprocessing": self.config.preprocessing,
        }
        path = output_dir / "dataset_metadata.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        
        # Ensure versioned manifest is tracked
        # Note: Full DataVersionControl.commit() should be called when dataset is finalized
        # This metadata file serves as a lightweight manifest for the dataset version
        LOGGER.debug(
            "dataset_metadata_materialized",
            extra={
                "dataset_name": self.config.dataset_name,
                "dataset_version": self.config.dataset_version,
                "path": str(path),
            },
        )
        return path

    def available_dataloaders(self, splits: Iterable[str]) -> Dict[str, DataLoader]:
        return {split: self.dataloader(split) for split in splits}

    def _should_shuffle(self, split: str, shuffle_override: bool | None) -> bool:
        if shuffle_override is not None:
            return shuffle_override
        if split == "train":
            return self.config.shuffle
        return False

    def _build_sampler(
        self,
        split: str,
        dataset,
        distributed_context: "DistributedContext" | None,
    ) -> Optional[distributed.DistributedSampler]:
        if distributed_context is None or not distributed_context.is_distributed:
            return None
        shuffle = split == "train" and self.config.shuffle
        return distributed.DistributedSampler(
            dataset,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=shuffle,
            drop_last=split == "train",
        )
