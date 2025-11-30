"""Core inference functions for Nexa-Spec 7B embedding + retrieval pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .embedding import EmbeddingEngine
from .preprocessor import SpectrumPreprocessor

LOGGER = logging.getLogger(__name__)


def load_nexa_model(checkpoint_path: str | Path, device: str = "cuda", config_path: Optional[Path] = None) -> nn.Module:
    """Load the pretrained Nexa-Spec encoder-only model.

    This function abstracts away Lightning / raw PyTorch checkpoints.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on ('cuda' or 'cpu')
        config_path: Optional path to model config YAML

    Returns:
        Loaded model in eval mode with gradients disabled
    """
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    checkpoint_path = Path(checkpoint_path)

    engine = EmbeddingEngine(checkpoint_path, config_path, device=str(device_obj))
    model = engine.model
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    LOGGER.info(f"Loaded Nexa-Spec model from {checkpoint_path} on {device_obj}")
    return model


def preprocess_spectrum(raw_spectrum: Dict[str, Any], preproc: SpectrumPreprocessor) -> torch.Tensor:
    """Converts raw m/z-intensity lists into standardized model-ready tensors.

    Expected raw_spectrum format:
    {
        "mz": [...],
        "intensity": [...],
        "precursor_mz": float (optional),
        "charge": int (optional),
        "retention_time": float (optional)
    }

    Args:
        raw_spectrum: Dictionary with spectrum data
        preproc: SpectrumPreprocessor instance

    Returns:
        Preprocessed tensor ready for model input
    """
    mz_values = np.array(raw_spectrum.get("mz", []))
    intensity_values = np.array(raw_spectrum.get("intensity", []))
    precursor_mz = raw_spectrum.get("precursor_mz")
    retention_time = raw_spectrum.get("retention_time")

    tensor = preproc.preprocess(mz_values, intensity_values, precursor_mz, retention_time)
    return tensor


def collate_batch(batch: List[torch.Tensor]) -> torch.Tensor:
    """Simple batch collation for inference.

    Args:
        batch: List of preprocessed spectrum tensors

    Returns:
        Batched tensor of shape (B, N, D)
    """
    return torch.stack(batch, dim=0)


def compute_embedding(model: nn.Module, batch_tensor: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Returns the [CLS] embedding for each spectrum.

    Assumes encoder returns (batch, seq, hidden_dim) and extracts CLS token or pools sequence.

    Args:
        model: Loaded encoder model
        batch_tensor: Batched input tensor of shape (B, N, D)
        device: Device to run inference on

    Returns:
        Normalized embeddings of shape (B, embedding_dim)
    """
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    batch_tensor = batch_tensor.to(device_obj)

    with torch.no_grad():
        outputs = model(batch_tensor)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if outputs.dim() == 3:
            cls_emb = outputs[:, 0, :]
        elif outputs.dim() == 2:
            cls_emb = outputs
        else:
            cls_emb = outputs.mean(dim=1) if outputs.dim() == 3 else outputs

        cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=-1)
        return cls_emb.cpu()


def embed_single_spectrum(
    model: nn.Module,
    spectrum: Dict[str, Any],
    preproc: SpectrumPreprocessor,
    device: str = "cuda",
) -> List[float]:
    """Preprocess → encode → return embedding as Python list.

    Args:
        model: Loaded encoder model
        spectrum: Raw spectrum dictionary
        preproc: SpectrumPreprocessor instance
        device: Device to run inference on

    Returns:
        Embedding as list of floats
    """
    tensor = preprocess_spectrum(spectrum, preproc)
    batch = tensor.unsqueeze(0)
    emb = compute_embedding(model, batch, device)
    return emb.squeeze(0).tolist()


def embed_batch(
    model: nn.Module,
    spectra: List[Dict[str, Any]],
    preproc: SpectrumPreprocessor,
    device: str = "cuda",
    batch_size: int = 32,
) -> List[List[float]]:
    """Preprocess a list of spectra → return embeddings as lists.

    Used during large-scale indexing.

    Args:
        model: Loaded encoder model
        spectra: List of raw spectrum dictionaries
        preproc: SpectrumPreprocessor instance
        device: Device to run inference on
        batch_size: Batch size for processing

    Returns:
        List of embeddings, each as list of floats
    """
    processed = [preprocess_spectrum(s, preproc) for s in spectra]
    embeddings = []

    for i in range(0, len(processed), batch_size):
        batch = collate_batch(processed[i : i + batch_size])
        embs = compute_embedding(model, batch, device)
        embeddings.extend(embs.tolist())

    return embeddings


def insert_into_qdrant(
    client: Any,
    collection: str,
    embeddings: List[List[float]],
    metadata: List[Dict[str, Any]],
    batch_size: int = 100,
) -> None:
    """Insert embeddings into Qdrant vector database.

    Args:
        client: QdrantClient instance
        collection: Collection name
        embeddings: List of embedding vectors
        metadata: List of metadata dictionaries
        batch_size: Batch size for uploads
    """
    from .vector_db import VectorDBClient

    if not isinstance(client, VectorDBClient):
        raise TypeError("client must be a VectorDBClient instance")

    embeddings_array = np.array(embeddings)
    ids = [meta.get("spectrum_id", f"spec_{i}") for i, meta in enumerate(metadata)]

    client.upload_embeddings(embeddings_array, ids, metadata, batch_size=batch_size)
    LOGGER.info(f"Inserted {len(embeddings)} embeddings into collection '{collection}'")


def search_qdrant(
    client: Any,
    collection: str,
    query_vector: List[float],
    k: int = 10,
    filter_conditions: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """ANN search in Qdrant.

    Args:
        client: QdrantClient instance
        collection: Collection name
        query_vector: Query embedding vector
        k: Number of results to return
        filter_conditions: Optional metadata filters

    Returns:
        List of search results with scores and metadata
    """
    from .vector_db import VectorDBClient

    if not isinstance(client, VectorDBClient):
        raise TypeError("client must be a VectorDBClient instance")

    query_array = np.array(query_vector)
    results = client.search(query_array, top_k=k, filter_conditions=filter_conditions)
    return results


def query_spectrum(
    model: nn.Module,
    spectrum: Dict[str, Any],
    preproc: SpectrumPreprocessor,
    client: Any,
    collection: str,
    k: int = 10,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Embed → query DB → return results.

    This is the top-level function that FastAPI will call.

    Args:
        model: Loaded encoder model
        spectrum: Raw spectrum dictionary
        preproc: SpectrumPreprocessor instance
        client: VectorDBClient instance
        collection: Collection name
        k: Number of results to return
        device: Device to run inference on

    Returns:
        List of search results with scores and metadata
    """
    vector = embed_single_spectrum(model, spectrum, preproc, device)
    results = search_qdrant(client, collection, vector, k)
    return results

