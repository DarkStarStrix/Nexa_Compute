"""Multi-model ensemble embedding utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from .core import embed_single_spectrum, load_nexa_model
from .preprocessor import SpectrumPreprocessor


class EnsembleEmbedder:
    """Average embeddings across multiple checkpoints."""

    def __init__(self, checkpoint_paths: Sequence[Path], device: str = "cuda") -> None:
        self.models = [load_nexa_model(path, device=device) for path in checkpoint_paths]
        self.preprocessor = SpectrumPreprocessor()
        self.device = device

    def embed(self, spectrum: Dict) -> np.ndarray:
        embeddings = [
            embed_single_spectrum(model, spectrum, self.preprocessor, device=self.device) for model in self.models
        ]
        return np.mean(np.stack(embeddings, axis=0), axis=0)

