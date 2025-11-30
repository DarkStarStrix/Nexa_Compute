"""Spectral inference server for MS/MS spectra embedding and search."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from .caching import EmbeddingCache
from .core import embed_single_spectrum
from .embedding import EmbeddingEngine
from .exporters import export_json_schema, export_mgf
from .feedback import FeedbackCollector
from .federated_search import FederatedSearch
from .graphql_api import build_graphql_app
from .middleware import APIKeyValidator, LicenseTier, RateLimitMiddleware, RateLimiter
from .monitoring import EMBEDDING_LATENCY, SEARCH_LATENCY, observe_latency, record_request
from .multi_tenancy import OrganizationMiddleware
from .preprocessor import SpectrumPreprocessor, SpectrumPreprocessorConfig
from .property_predictor import PropertyPredictor
from .query_expansion import QueryExpander
from .reranker import Reranker
from .snapshot import create_snapshot
from .storage import EmbeddingStorage
from .structure_decoder import StructureDecoder
from .vector_db import VectorDBClient

try:  # optional dependency
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain"
    CollectorRegistry = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]


class SpectrumEmbedRequest(BaseModel):
    """Request schema for spectrum embedding."""

    mz_values: List[float]
    intensity_values: List[float]
    precursor_mz: Optional[float] = None
    retention_time: Optional[float] = None
    spectrum_id: Optional[str] = None


class SpectrumEmbedResponse(BaseModel):
    """Response schema for spectrum embedding."""

    embedding: List[float]
    spectrum_id: Optional[str] = None
    latency_ms: float
    embedding_dim: int


class SearchRequest(BaseModel):
    """Request schema for vector search."""

    embedding: Optional[List[float]] = None
    spectrum_id: Optional[str] = None
    top_k: int = 10
    filter_conditions: Optional[Dict[str, Any]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    query_metadata: Optional[Dict[str, Any]] = None
    use_query_expansion: bool = False
    federated: bool = False


class SearchResponse(BaseModel):
    """Response schema for vector search."""

    results: List[Dict[str, Any]]
    query_latency_ms: float


class NearestRequest(BaseModel):
    """Request schema for nearest neighbors."""

    spectrum_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    top_k: int = 10


class SpectralInferenceServer:
    """FastAPI server for spectral inference and search."""

    def __init__(
        self,
        checkpoint_path: Path,
        config_path: Optional[Path] = None,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "nexa_spectra",
        embedding_dim: int = 768,
        preprocessor_config: Optional[SpectrumPreprocessorConfig] = None,
        api_keys: Optional[Dict[str, str]] = None,
    ):
        """Initialize spectral inference server.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to model config
            qdrant_url: Qdrant server URL
            qdrant_api_key: Optional Qdrant API key
            collection_name: Qdrant collection name
            embedding_dim: Embedding dimension
            preprocessor_config: Optional preprocessor configuration
            api_keys: Optional API key to tier mapping
        """
        self.preprocessor = SpectrumPreprocessor(preprocessor_config)
        self.embedding_engine = EmbeddingEngine(checkpoint_path, config_path, embedding_dim=embedding_dim)
        self.vector_db = VectorDBClient(qdrant_url, qdrant_api_key, collection_name, embedding_dim)
        self.embedding_storage = EmbeddingStorage(Path("./embeddings"))
        self.cache = EmbeddingCache()
        self.property_predictor = PropertyPredictor(input_dim=embedding_dim)
        self.structure_decoder = StructureDecoder(embedding_dim)
        self.reranker = Reranker()
        self.query_expander = QueryExpander()
        self.feedback = FeedbackCollector()
        self.federated_search: Optional[FederatedSearch] = None

        self.graphql_enabled = False

        self.app = FastAPI(title="Nexa-Spec Inference Server")
        self._setup_middleware(api_keys)
        self._setup_routes()

    def _setup_middleware(self, api_keys: Optional[Dict[str, str]]) -> None:
        """Setup API key and rate limiting middleware."""
        rate_limiter = RateLimiter()
        api_validator = APIKeyValidator(api_keys)
        self.app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter, api_validator=api_validator)
        self.app.add_middleware(OrganizationMiddleware, require_org=False)

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self.app.get("/health")
        def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "device": str(self.embedding_engine.device),
                "embedding_dim": self.embedding_engine.embedding_dim,
            }

        @self.app.post("/embed", response_model=SpectrumEmbedResponse)
        def embed(request: SpectrumEmbedRequest, req: Request):
            """Embed a raw spectrum."""
            start_time = time.perf_counter()

            try:
                spectrum_dict = {
                    "mz": request.mz_values,
                    "intensity": request.intensity_values,
                    "precursor_mz": request.precursor_mz,
                    "retention_time": request.retention_time,
                }

                cache_key = request.spectrum_id or str(hash(tuple(request.mz_values) + tuple(request.intensity_values)))
                cached = self.cache.get(cache_key)
                if cached is not None:
                    embedding = cached.tolist()
                else:
                    with observe_latency(EMBEDDING_LATENCY, "/embed"):
                        embedding_tensor = embed_single_spectrum(
                            self.embedding_engine.model,
                            spectrum_dict,
                            self.preprocessor,
                            device=str(self.embedding_engine.device),
                        )
                    embedding = embedding_tensor
                    self.cache.set(cache_key, np.array(embedding))

                latency_ms = (time.perf_counter() - start_time) * 1000

                response = SpectrumEmbedResponse(
                    embedding=embedding,
                    spectrum_id=request.spectrum_id,
                    latency_ms=round(latency_ms, 2),
                    embedding_dim=self.embedding_engine.embedding_dim,
                )
                record_request("/embed", 200)
                return response
            except Exception as e:
                record_request("/embed", 500)
                raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

        @self.app.post("/search", response_model=SearchResponse)
        def search(request: SearchRequest, req: Request):
            """Search for similar spectra."""
            start_time = time.perf_counter()

            try:
                if request.embedding is None and request.spectrum_id is None:
                    raise HTTPException(status_code=400, detail="Either embedding or spectrum_id must be provided")

                if request.spectrum_id and not request.embedding:
                    nearest_results = self.vector_db.get_nearest(
                        spectrum_id=request.spectrum_id,
                        top_k=request.top_k,
                    )
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    record_request("/search", 200)
                    return SearchResponse(results=nearest_results, query_latency_ms=round(latency_ms, 2))

                query_embedding = np.array(request.embedding)

                candidates = []
                embeddings_to_query = [query_embedding]
                if request.use_query_expansion:
                    embeddings_to_query = self.query_expander.expand(query_embedding)

                with observe_latency(SEARCH_LATENCY, "/search"):
                    for embedding_variant in embeddings_to_query:
                        if request.federated and self.federated_search:
                            candidates.extend(self.federated_search.search(embedding_variant, k=request.top_k))
                        else:
                            candidates.extend(
                                self.vector_db.search(
                                    embedding_variant,
                                    top_k=request.top_k,
                                    filter_conditions=request.filter_conditions,
                                )
                            )

                results = self.reranker.rerank(
                    candidates,
                    request.query_metadata or {},
                    metadata_filters=request.metadata_filters,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000
                record_request("/search", 200)
                return SearchResponse(results=results[: request.top_k], query_latency_ms=round(latency_ms, 2))
            except Exception as e:
                record_request("/search", 500)
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

        @self.app.post("/nearest", response_model=SearchResponse)
        def nearest(request: NearestRequest, req: Request):
            """Get nearest neighbors for a spectrum."""
            start_time = time.perf_counter()

            try:
                if request.embedding is None and request.spectrum_id is None:
                    raise HTTPException(status_code=400, detail="Either embedding or spectrum_id must be provided")

                if request.spectrum_id:
                    query_embedding = None
                else:
                    query_embedding = np.array(request.embedding)

                results = self.vector_db.get_nearest(
                    spectrum_id=request.spectrum_id,
                    embedding=query_embedding,
                    top_k=request.top_k,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000

                record_request("/nearest", 200)
                return SearchResponse(results=results, query_latency_ms=round(latency_ms, 2))
            except Exception as e:
                record_request("/nearest", 500)
                raise HTTPException(status_code=500, detail=f"Nearest search failed: {str(e)}")

        @self.app.get("/model/info")
        def model_info():
            """Get model information."""
            return {
                "embedding_dim": self.embedding_engine.embedding_dim,
                "device": str(self.embedding_engine.device),
                "collection_name": self.vector_db.collection_name,
            }

        @self.app.post("/predict/retention_time")
        def predict_retention_time(request: SpectrumEmbedRequest):
            embedding = self._embed_tensor(request)
            value = self.property_predictor.predict_retention_time(torch.tensor(embedding))
            return {"retention_time": value}

        @self.app.post("/predict/ion_mobility")
        def predict_ion_mobility(request: SpectrumEmbedRequest):
            embedding = self._embed_tensor(request)
            value = self.property_predictor.predict_ion_mobility(torch.tensor(embedding))
            return {"ion_mobility": value}

        @self.app.post("/predict/molecular_weight")
        def predict_molecular_weight(request: SpectrumEmbedRequest):
            embedding = self._embed_tensor(request)
            value = self.property_predictor.predict_molecular_weight(torch.tensor(embedding))
            return {"molecular_weight": value}

        @self.app.post("/decode/structure")
        def decode_structure(request: SpectrumEmbedRequest):
            embedding = self._embed_tensor(request)
            tensor = torch.tensor(embedding)
            return self.structure_decoder.decode_structure(tensor)

        @self.app.post("/decode/smiles")
        def decode_smiles(request: SpectrumEmbedRequest):
            embedding = self._embed_tensor(request)
            return {"smiles": self.structure_decoder.decode_smiles(torch.tensor(embedding))}

        @self.app.post("/decode/inchi")
        def decode_inchi(request: SpectrumEmbedRequest):
            embedding = self._embed_tensor(request)
            return {"inchi": self.structure_decoder.decode_inchi(torch.tensor(embedding))}

        @self.app.post("/feedback/positive")
        def feedback_positive(payload: Dict[str, str]):
            record = self.feedback.record(payload["query_id"], payload["result_id"], True, payload.get("user_id"))
            return record

        @self.app.post("/feedback/negative")
        def feedback_negative(payload: Dict[str, str]):
            record = self.feedback.record(payload["query_id"], payload["result_id"], False, payload.get("user_id"))
            return record

        @self.app.get("/feedback/stats")
        def feedback_stats():
            return self.feedback.stats()

        @self.app.post("/export")
        def export_embeddings(payload: Dict[str, Any]):
            embeddings = np.array(payload["embeddings"])
            spectrum_ids = payload.get("spectrum_ids") or [f"spec_{i}" for i in range(len(embeddings))]
            fmt = payload.get("format", "json")
            output_path = Path(payload.get("output_path", "embeddings_export"))
            if fmt == "mgf":
                export_mgf(embeddings, spectrum_ids, output_path.with_suffix(".mgf"))
            else:
                export_json_schema(embeddings, spectrum_ids, output_path.with_suffix(".json"))
            return {"path": str(output_path)}

        @self.app.post("/snapshot")
        def snapshot(payload: Dict[str, Any]):
            source = Path(payload["source"])
            target = Path(payload["target"])
            path = create_snapshot(source, target)
            return {"snapshot_path": str(path)}

        @self.app.get("/metrics")
        def metrics():
            if generate_latest is None:
                raise HTTPException(status_code=503, detail="prometheus_client not installed")
            registry = CollectorRegistry()
            data = generate_latest(registry)
            return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

        @self.app.websocket("/stream/batch")
        async def stream_batch(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_json({"status": "started"})
            # Placeholder streaming loop
            await websocket.send_json({"progress": 1.0, "message": "Completed"})
            await websocket.close()

        try:
            graphql_app = build_graphql_app(lambda mz, intensity: mz[:1] + intensity[:1])
            self.app.add_route("/graphql", graphql_app)
            self.graphql_enabled = True
        except Exception:
            self.graphql_enabled = False

    def _embed_tensor(self, request: SpectrumEmbedRequest) -> List[float]:
        spectrum_dict = {
            "mz": request.mz_values,
            "intensity": request.intensity_values,
            "precursor_mz": request.precursor_mz,
            "retention_time": request.retention_time,
        }
        return embed_single_spectrum(
            self.embedding_engine.model,
            spectrum_dict,
            self.preprocessor,
            device=str(self.embedding_engine.device),
        )


def serve_spectral_model(
    checkpoint_path: Path,
    config_path: Optional[Path] = None,
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: Optional[str] = None,
    collection_name: str = "nexa_spectra",
    embedding_dim: int = 768,
    host: str = "0.0.0.0",
    port: int = 8000,
    api_keys: Optional[Dict[str, str]] = None,
) -> None:
    """Serve spectral inference model via FastAPI."""
    import uvicorn

    server = SpectralInferenceServer(
        checkpoint_path,
        config_path,
        qdrant_url,
        qdrant_api_key,
        collection_name,
        embedding_dim,
        api_keys=api_keys,
    )
    uvicorn.run(server.app, host=host, port=port)

