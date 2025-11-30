"""Embedding caching utilities."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Optional

import numpy as np


class EmbeddingCache:
    """LRU cache with optional TTL."""

    def __init__(self, capacity: int = 1024, ttl_seconds: int = 3600) -> None:
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, np.ndarray]] = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key not in self._store:
            return None
        timestamp, embedding = self._store[key]
        if time.time() - timestamp > self.ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return embedding

    def set(self, key: str, embedding: np.ndarray) -> None:
        self._store[key] = (time.time(), embedding)
        self._store.move_to_end(key)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

