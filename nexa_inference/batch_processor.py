"""Batch processing utilities for large-scale embedding generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch

from .core import embed_batch, insert_into_qdrant, load_nexa_model
from .preprocessor import SpectrumPreprocessor
from .storage import EmbeddingStorage
from .vector_db import VectorDBClient

LOGGER = logging.getLogger(__name__)

try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError:
    pq = None
    pd = None


class BatchProcessor:
    """Process large batches of spectra for embedding generation and indexing."""

    def __init__(
        self,
        model_path: Path,
        preprocessor: Optional[SpectrumPreprocessor] = None,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """Initialize batch processor.

        Args:
            model_path: Path to model checkpoint
            preprocessor: Optional preprocessor instance
            device: Device to run inference on
            batch_size: Batch size for processing
        """
        self.model = load_nexa_model(model_path, device=device)
        self.preprocessor = preprocessor or SpectrumPreprocessor()
        self.device = device
        self.batch_size = batch_size

    def process_shard(
        self,
        shard_path: Path,
        output_dir: Path,
        collection_name: Optional[str] = None,
        vector_db_client: Optional[VectorDBClient] = None,
        save_to_disk: bool = True,
        upload_to_db: bool = False,
    ) -> Dict[str, Any]:
        """Process a single Arrow/HDF5 shard.

        Args:
            shard_path: Path to shard file
            output_dir: Directory to save embeddings
            collection_name: Optional Qdrant collection name
            vector_db_client: Optional VectorDBClient instance
            save_to_disk: Whether to save embeddings to disk
            upload_to_db: Whether to upload to vector database

        Returns:
            Dictionary with processing statistics
        """
        LOGGER.info(f"Processing shard: {shard_path}")

        spectra, metadata = self._load_shard(shard_path)
        embeddings = embed_batch(self.model, spectra, self.preprocessor, self.device, self.batch_size)

        stats = {"processed": len(spectra), "embeddings": len(embeddings)}

        if save_to_disk:
            storage = EmbeddingStorage(output_dir)
            spectrum_ids = [meta.get("spectrum_id", f"spec_{i}") for i, meta in enumerate(metadata)]
            storage.save_embeddings(
                embeddings=np.array(embeddings),
                spectrum_ids=spectrum_ids,
                metadata=metadata,
                filename=shard_path.stem + "_embeddings.jsonl",
            )
            stats["saved_to"] = str(output_dir)

        if upload_to_db and vector_db_client:
            insert_into_qdrant(vector_db_client, collection_name or "nexa_spectra", embeddings, metadata)
            stats["uploaded_to_db"] = True

        return stats

    def process_stream(
        self,
        spectra_stream: Iterator[Dict[str, Any]],
        output_path: Path,
        collection_name: Optional[str] = None,
        vector_db_client: Optional[VectorDBClient] = None,
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """Process a stream of spectra.

        Args:
            spectra_stream: Iterator of spectrum dictionaries
            output_path: Path to save embeddings
            collection_name: Optional Qdrant collection name
            vector_db_client: Optional VectorDBClient instance
            chunk_size: Chunk size for batching

        Returns:
            Dictionary with processing statistics
        """
        all_embeddings = []
        all_metadata = []
        chunk = []

        for spectrum in spectra_stream:
            chunk.append(spectrum)
            if len(chunk) >= chunk_size:
                embeddings = embed_batch(self.model, chunk, self.preprocessor, self.device, self.batch_size)
                all_embeddings.extend(embeddings)
                all_metadata.extend([s.get("metadata", {}) for s in chunk])
                chunk = []

        if chunk:
            embeddings = embed_batch(self.model, chunk, self.preprocessor, self.device, self.batch_size)
            all_embeddings.extend(embeddings)
            all_metadata.extend([s.get("metadata", {}) for s in chunk])

        stats = {"processed": len(all_embeddings)}

        if output_path:
            storage = EmbeddingStorage(output_path.parent)
            spectrum_ids = [meta.get("spectrum_id", f"spec_{i}") for i, meta in enumerate(all_metadata)]
            storage.save_embeddings(
                embeddings=np.array(all_embeddings),
                spectrum_ids=spectrum_ids,
                metadata=all_metadata,
                filename=output_path.name,
            )
            stats["saved_to"] = str(output_path)

        if vector_db_client:
            insert_into_qdrant(vector_db_client, collection_name or "nexa_spectra", all_embeddings, all_metadata)
            stats["uploaded_to_db"] = True

        return stats

    def _load_shard(self, shard_path: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load spectra from Arrow/HDF5 shard.

        Args:
            shard_path: Path to shard file

        Returns:
            Tuple of (spectra list, metadata list)
        """
        if shard_path.suffix == ".parquet" and pq is not None:
            df = pd.read_parquet(shard_path)
            spectra = []
            metadata = []

            for _, row in df.iterrows():
                spectrum = {
                    "mz": row.get("mz", []),
                    "intensity": row.get("intensity", []),
                    "precursor_mz": row.get("precursor_mz"),
                    "retention_time": row.get("retention_time"),
                }
                spectra.append(spectrum)

                meta = {k: v for k, v in row.items() if k not in ["mz", "intensity"]}
                metadata.append(meta)

            return spectra, metadata
        else:
            raise ValueError(f"Unsupported shard format: {shard_path.suffix}")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    collection_name: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Process all shards in a directory.

    Args:
        input_dir: Directory containing shard files
        output_dir: Directory to save embeddings
        model_path: Path to model checkpoint
        collection_name: Optional Qdrant collection name
        qdrant_url: Optional Qdrant server URL
        device: Device to run inference on
        batch_size: Batch size for processing

    Returns:
        Dictionary with overall statistics
    """
    processor = BatchProcessor(model_path, device=device, batch_size=batch_size)
    vector_db_client = None

    if qdrant_url:
        vector_db_client = VectorDBClient(url=qdrant_url, collection_name=collection_name or "nexa_spectra")

    shard_files = list(input_dir.glob("*.parquet"))
    total_stats = {"shards_processed": 0, "total_spectra": 0}

    for shard_path in shard_files:
        stats = processor.process_shard(
            shard_path,
            output_dir,
            collection_name=collection_name,
            vector_db_client=vector_db_client,
            save_to_disk=True,
            upload_to_db=qdrant_url is not None,
        )
        total_stats["shards_processed"] += 1
        total_stats["total_spectra"] += stats["processed"]

    return total_stats

