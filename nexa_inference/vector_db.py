"""Vector database client for Qdrant/Atlas++ integration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        Filter,
        PointStruct,
        VectorParams,
    )
except ImportError:
    QdrantClient = None
    Distance = None
    Filter = None
    PointStruct = None
    VectorParams = None


class VectorDBClient:
    """Client for interacting with Qdrant vector database."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "nexa_spectra",
        embedding_dim: int = 768,
    ):
        """Initialize Qdrant client.

        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
        """
        if QdrantClient is None:
            raise ImportError("qdrant-client not installed. Install with: pip install qdrant-client")

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                LOGGER.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            LOGGER.warning(f"Error ensuring collection exists: {e}")

    def upload_embeddings(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> None:
        """Upload embeddings to vector database.

        Args:
            embeddings: Array of shape (N, embedding_dim)
            ids: List of spectrum IDs
            metadata: Optional list of metadata dictionaries
        """
        if len(embeddings) != len(ids):
            raise ValueError(f"Embeddings length ({len(embeddings)}) != IDs length ({len(ids)})")

        if metadata is None:
            metadata = [{}] * len(ids)

        points = []
        for i, (embedding, spectrum_id, meta) in enumerate(zip(embeddings, ids, metadata)):
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={"spectrum_id": spectrum_id, **meta},
            )
            points.append(point)

            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        LOGGER.info(f"Uploaded {len(embeddings)} embeddings to {self.collection_name}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar spectra.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_conditions: Optional metadata filters

        Returns:
            List of search results with scores and metadata
        """
        query_filter = None
        if filter_conditions:
            query_filter = Filter(must=[{"key": k, "match": {"value": v}} for k, v in filter_conditions.items()])

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            {
                "spectrum_id": result.payload.get("spectrum_id"),
                "score": result.score,
                "metadata": result.payload,
            }
            for result in results
        ]

    def get_nearest(
        self,
        spectrum_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get nearest neighbors for a spectrum ID or embedding.

        Args:
            spectrum_id: Spectrum ID to search for
            embedding: Optional embedding vector (if spectrum_id not provided)
            top_k: Number of results to return

        Returns:
            List of nearest neighbors with scores
        """
        if spectrum_id:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[{"key": "spectrum_id", "match": {"value": spectrum_id}}]),
                limit=1,
            )
            if not results[0]:
                raise ValueError(f"Spectrum ID not found: {spectrum_id}")
            embedding = np.array(results[0][0].vector)

        if embedding is None:
            raise ValueError("Either spectrum_id or embedding must be provided")

        return self.search(embedding, top_k=top_k)

