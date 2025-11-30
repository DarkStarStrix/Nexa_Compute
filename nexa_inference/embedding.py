"""Embedding generation engine for MS/MS spectra."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

try:
    from nexa_train.models import DEFAULT_MODEL_REGISTRY
except ImportError:
    DEFAULT_MODEL_REGISTRY = None


class EmbeddingEngine:
    """Generates embeddings from preprocessed spectra using encoder model."""

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Optional[Path] = None,
        device: Optional[str] = None,
        embedding_dim: int = 768,
    ):
        """Initialize embedding engine with trained model.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to model config
            device: Device to run inference on (cuda/cpu)
            embedding_dim: Expected embedding dimension
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.embedding_dim = embedding_dim
        self.model = self._load_model(checkpoint_path, config_path)
        self.model.eval()
        LOGGER.info(f"Embedding engine initialized on {self.device}")

    def _load_model(self, checkpoint_path: Path, config_path: Optional[Path]) -> nn.Module:
        """Load encoder model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        elif config_path and config_path.exists():
            import yaml

            config = yaml.safe_load(config_path.read_text())
            model_config = config.get("model", {})
        else:
            raise ValueError("Cannot determine model architecture. Provide config_path or ensure checkpoint contains model_config.")

        model_name = model_config.get("name", "transformer_encoder")
        if DEFAULT_MODEL_REGISTRY:
            model = DEFAULT_MODEL_REGISTRY.build(model_name, model_config.get("parameters", {}))
        else:
            raise ValueError("Model registry not available. Ensure nexa_train is installed.")

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        return model

    def embed(self, spectrum_tensor: torch.Tensor) -> torch.Tensor:
        """Generate embedding from preprocessed spectrum tensor.

        Args:
            spectrum_tensor: Tensor of shape (B, N, D) or (N, D)

        Returns:
            Embedding tensor of shape (B, embedding_dim) or (embedding_dim,)
        """
        with torch.no_grad():
            if spectrum_tensor.dim() == 2:
                spectrum_tensor = spectrum_tensor.unsqueeze(0)

            spectrum_tensor = spectrum_tensor.to(self.device)
            embedding = self._extract_cls_token(spectrum_tensor)

            if embedding.dim() == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)

            return embedding.cpu()

    def _extract_cls_token(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token embedding from encoder output.

        For encoder-only transformers, this typically means pooling the sequence.
        """
        if hasattr(self.model, "encoder"):
            encoded = self.model.encoder(inputs)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            pooled = encoded.mean(dim=1)
            return pooled
        elif hasattr(self.model, "pool"):
            encoded = self.model.encoder(inputs) if hasattr(self.model, "encoder") else inputs
            pooled = self.model.pool(encoded.transpose(1, 2)).squeeze(-1)
            return pooled
        else:
            output = self.model(inputs)
            if isinstance(output, tuple):
                output = output[0]
            if output.dim() == 3:
                pooled = output.mean(dim=1)
            else:
                pooled = output
            return pooled

    def embed_batch(self, spectrum_tensors: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """Generate embeddings for a batch of spectra.

        Args:
            spectrum_tensors: Tensor of shape (B, N, D)
            batch_size: Batch size for processing

        Returns:
            Embedding tensor of shape (B, embedding_dim)
        """
        embeddings = []
        for i in range(0, len(spectrum_tensors), batch_size):
            batch = spectrum_tensors[i : i + batch_size]
            batch_embeddings = self.embed(batch)
            embeddings.append(batch_embeddings)

        return torch.cat(embeddings, dim=0)

    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L2-normalize embeddings for cosine similarity."""
        norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings / (norms + 1e-8)

