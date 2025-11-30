"""Unit tests for Nexa inference pipeline feature scaffolds."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from nexa_inference import (
    EmbeddingCache,
    FederatedSearch,
    FeedbackCollector,
    PropertyPredictor,
    QueryExpander,
    Reranker,
    StructureDecoder,
    create_snapshot,
)


class DummyClient:
    """Simple stand-in for VectorDBClient to test FederatedSearch."""

    def __init__(self, score: float) -> None:
        self._score = score

    def search(self, query_vector: np.ndarray, top_k: int = 10, filter_conditions=None):
        return [{"score": self._score, "metadata": {"client": self._score}}]


def test_property_predictor_returns_scalars():
    torch.manual_seed(0)
    predictor = PropertyPredictor(input_dim=4, device="cpu")
    embedding = torch.ones(4)
    rt = predictor.predict_retention_time(embedding)
    im = predictor.predict_ion_mobility(embedding)
    mw = predictor.predict_molecular_weight(embedding)
    assert isinstance(rt, float)
    assert isinstance(im, float)
    assert isinstance(mw, float)


def test_structure_decoder_outputs_expected_fields():
    torch.manual_seed(0)
    decoder = StructureDecoder(embedding_dim=4, device="cpu")
    embedding = torch.zeros(4)
    payload = decoder.decode_structure(embedding)
    assert {"smiles", "inchi", "confidence"} <= payload.keys()
    assert payload["smiles"]
    assert payload["inchi"].startswith("InChI=1S/")


def test_reranker_filters_by_retention_time_and_metadata():
    reranker = Reranker(rt_tolerance=0.05)
    results = [
        {"score": 0.2, "metadata": {"retention_time": 1.00, "instrument": "QTOF"}},
        {"score": 0.9, "metadata": {"retention_time": 1.20, "instrument": "Orbitrap"}},
    ]
    reranked = reranker.rerank(
        results,
        query_metadata={"retention_time": 1.01},
        metadata_filters={"instrument": "QTOF"},
    )
    assert len(reranked) == 1
    assert reranked[0]["metadata"]["instrument"] == "QTOF"


def test_embedding_cache_lru_and_ttl_behavior():
    cache = EmbeddingCache(capacity=1, ttl_seconds=0.01)
    cache.set("a", np.array([0.1]))
    assert cache.get("a") is not None
    cache.set("b", np.array([0.2]))
    assert cache.get("a") is None  # evicted by capacity
    time.sleep(0.02)
    assert cache.get("b") is None  # expired by TTL


def test_query_expander_returns_variants():
    expander = QueryExpander(noise_scale=0.0, variants=2)
    base = np.array([1.0, 0.0, 0.5])
    variants = expander.expand(base)
    assert len(variants) == 3  # base + 2 variants
    assert np.array_equal(variants[0], base)


def test_federated_search_orders_results():
    federated = FederatedSearch([DummyClient(0.5), DummyClient(0.9)])
    results = federated.search(np.zeros(3), k=1)
    assert results[0]["score"] == 0.9


def test_feedback_collector_records_entries():
    collector = FeedbackCollector()
    collector.record("q1", "res1", True, "userA")
    collector.record("q1", "res2", False, "userA")
    stats = collector.stats()
    assert stats["total"] == 2
    assert stats["positives"] == 1
    assert stats["negatives"] == 1


def test_snapshot_creation(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "snap"
    source.mkdir()
    (source / "vectors").mkdir()
    (source / "vectors" / "data.bin").write_bytes(b"nexa")
    snapshot_path = create_snapshot(source, target)
    assert snapshot_path.exists()
    assert (snapshot_path / "vectors" / "data.bin").read_bytes() == b"nexa"

