"""Query expansion strategies for embeddings."""

from __future__ import annotations

from typing import List

import numpy as np


class QueryExpander:
    """Generate diverse query variants."""

    def __init__(self, noise_scale: float = 0.05, variants: int = 3) -> None:
        self.noise_scale = noise_scale
        self.variants = variants

    def expand(self, query_embedding: np.ndarray) -> List[np.ndarray]:
        variants = [query_embedding]
        for _ in range(self.variants):
            noise = np.random.normal(0, self.noise_scale, size=query_embedding.shape)
            variants.append(query_embedding + noise)
        return variants

