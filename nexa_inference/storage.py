"""Embedding storage utilities for JSONL and Arrow formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

LOGGER = logging.getLogger(__name__)


class EmbeddingStorage:
    """Handles storage and retrieval of embeddings in JSONL or Arrow format."""

    def __init__(self, output_dir: Path, format: str = "jsonl"):
        """Initialize storage handler.

        Args:
            output_dir: Directory to store embeddings
            format: Storage format ('jsonl' or 'arrow')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format.lower()

        if self.format not in ["jsonl", "arrow", "parquet"]:
            raise ValueError(f"Unsupported format: {format}. Use 'jsonl' or 'arrow'")

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        spectrum_ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        filename: Optional[str] = None,
        normalize: bool = True,
    ) -> Path:
        """Save embeddings to disk.

        Args:
            embeddings: Array of shape (N, embedding_dim)
            spectrum_ids: List of spectrum IDs
            metadata: Optional list of metadata dictionaries
            filename: Optional output filename
            normalize: Whether to L2-normalize embeddings before saving

        Returns:
            Path to saved file
        """
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        if metadata is None:
            metadata = [{}] * len(spectrum_ids)

        if filename is None:
            filename = f"embeddings.{self.format}"

        output_path = self.output_dir / filename

        if self.format == "jsonl":
            self._save_jsonl(embeddings, spectrum_ids, metadata, output_path)
        else:
            self._save_arrow(embeddings, spectrum_ids, metadata, output_path)

        LOGGER.info(f"Saved {len(embeddings)} embeddings to {output_path}")
        return output_path

    def _save_jsonl(
        self,
        embeddings: np.ndarray,
        spectrum_ids: List[str],
        metadata: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Save embeddings in JSONL format."""
        with open(output_path, "w") as f:
            for embedding, spectrum_id, meta in zip(embeddings, spectrum_ids, metadata):
                record = {
                    "spectrum_id": spectrum_id,
                    "embedding": embedding.tolist(),
                    **meta,
                }
                f.write(json.dumps(record) + "\n")

    def _save_arrow(
        self,
        embeddings: np.ndarray,
        spectrum_ids: List[str],
        metadata: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Save embeddings in Arrow/Parquet format."""
        if pa is None or pq is None:
            raise ImportError("pyarrow not installed. Install with: pip install pyarrow")

        data = {
            "spectrum_id": spectrum_ids,
            "embedding": embeddings.tolist(),
        }

        for key in metadata[0].keys():
            data[key] = [meta.get(key) for meta in metadata]

        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path)

    def load_embeddings(self, filepath: Path) -> tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Load embeddings from disk.

        Args:
            filepath: Path to embeddings file

        Returns:
            Tuple of (embeddings, spectrum_ids, metadata)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")

        if filepath.suffix == ".jsonl":
            return self._load_jsonl(filepath)
        else:
            return self._load_arrow(filepath)

    def _load_jsonl(self, filepath: Path) -> tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Load embeddings from JSONL format."""
        embeddings = []
        spectrum_ids = []
        metadata = []

        with open(filepath) as f:
            for line in f:
                record = json.loads(line)
                embeddings.append(record["embedding"])
                spectrum_ids.append(record["spectrum_id"])
                meta = {k: v for k, v in record.items() if k not in ["embedding", "spectrum_id"]}
                metadata.append(meta)

        return np.array(embeddings), spectrum_ids, metadata

    def _load_arrow(self, filepath: Path) -> tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """Load embeddings from Arrow/Parquet format."""
        if pa is None or pq is None:
            raise ImportError("pyarrow not installed. Install with: pip install pyarrow")

        table = pq.read_table(filepath)
        df = table.to_pandas()

        embeddings = np.array([np.array(emb) for emb in df["embedding"]])
        spectrum_ids = df["spectrum_id"].tolist()
        metadata = [{k: v for k, v in row.items() if k not in ["embedding", "spectrum_id"]} for _, row in df.iterrows()]

        return embeddings, spectrum_ids, metadata

