"""Property prediction heads for molecular attributes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class PropertyHead(nn.Module):
    """Simple feed-forward head for scalar property prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(embedding).squeeze(-1)


class PropertyPredictor:
    """Predict molecular properties from embeddings."""

    def __init__(
        self,
        *,
        input_dim: int,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.heads: Dict[str, PropertyHead] = {
            "retention_time": PropertyHead(input_dim),
            "ion_mobility": PropertyHead(input_dim),
            "molecular_weight": PropertyHead(input_dim),
        }
        self.to(self.device)
        if checkpoint_path and checkpoint_path.exists():
            self._load(checkpoint_path)

    def _load(self, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device)
        for name, head in self.heads.items():
            state_dict = payload.get(name)
            if state_dict:
                head.load_state_dict(state_dict)

    def to(self, device: torch.device) -> None:
        for head in self.heads.values():
            head.to(device)

    def _predict(self, head_name: str, embedding: torch.Tensor) -> float:
        head = self.heads[head_name]
        embedding = embedding.to(self.device)
        with torch.no_grad():
            value = head(embedding)
        return float(value.item())

    def predict_retention_time(self, embedding: torch.Tensor) -> float:
        return self._predict("retention_time", embedding)

    def predict_ion_mobility(self, embedding: torch.Tensor) -> float:
        return self._predict("ion_mobility", embedding)

    def predict_molecular_weight(self, embedding: torch.Tensor) -> float:
        return self._predict("molecular_weight", embedding)

    def save(self, checkpoint_path: Path) -> None:
        payload = {name: head.state_dict() for name, head in self.heads.items()}
        torch.save(payload, checkpoint_path)

