"""Python wrapper for nexa_train_pack Rust extension."""

from __future__ import annotations

import json
from typing import Any, Dict, List

try:
    import nexa_train_pack as _rust
except ImportError:
    _rust = None


class TrainPackError(Exception):
    """Base exception for training packing errors."""
    pass


class NexaTrainPack:
    """Interface to the high-performance Rust packing engine."""

    @property
    def available(self) -> bool:
        return _rust is not None

    def _check_available(self) -> None:
        if not self.available:
            raise TrainPackError("nexa_train_pack Rust extension not installed or failed to load.")

    def pack_sequences(self, shards: List[str], config: Dict[str, Any]) -> str:
        """Pack sequences from shards."""
        self._check_available()
        try:
            return _rust.pack_sequences(shards, json.dumps(config))
        except Exception as e:
            raise TrainPackError(f"Packing failed: {e}") from e


# Global instance
pack_core = NexaTrainPack()

