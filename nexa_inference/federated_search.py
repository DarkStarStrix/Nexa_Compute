"""Federated vector search across multiple Qdrant collections."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .vector_db import VectorDBClient


class FederatedSearch:
    """Runs search queries across multiple vector DB clients."""

    def __init__(self, clients: Sequence[VectorDBClient]) -> None:
        self.clients = list(clients)

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        aggregated: List[Dict] = []
        for client in self.clients:
            results = client.search(query_vector, top_k=k)
            aggregated.extend(results)
        aggregated.sort(key=lambda r: r.get("score", 0), reverse=True)
        return aggregated[:k]

