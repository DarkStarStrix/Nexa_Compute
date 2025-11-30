"""
Integration tests for the Model Registry.
"""

import pytest
from pathlib import Path
from nexa_compute.core.registry import ModelRegistry, RegistryError

@pytest.fixture
def temp_registry(tmp_path):
    db_path = tmp_path / "test_models.db"
    return ModelRegistry(db_path)

def test_register_and_resolve(temp_registry):
    """Test that we can register a model and resolve it back."""
    meta = {"framework": "pytorch", "params": 1000}
    version = temp_registry.register("my-model", "s3://bucket/model.pt", meta)
    
    # Resolve by exact version
    uri = temp_registry.resolve(f"my-model:{version}")
    assert uri == "s3://bucket/model.pt"
    
    # Resolve by latest (implicit)
    uri_latest = temp_registry.resolve("my-model")
    assert uri_latest == "s3://bucket/model.pt"

def test_version_increment(temp_registry):
    """Test that versions increment automatically."""
    meta = {}
    v1 = temp_registry.register("model-a", "uri-1", meta)
    v2 = temp_registry.register("model-a", "uri-2", meta)
    
    assert v1 == "1.0.0"
    assert v2 == "1.0.1"

def test_tags_promotion(temp_registry):
    """Test tagging and promotion flow."""
    meta = {}
    v1 = temp_registry.register("model-b", "uri-v1", meta)
    v2 = temp_registry.register("model-b", "uri-v2", meta)
    
    # Promote v1 to 'prod'
    temp_registry.promote("model-b", v1, "prod")
    assert temp_registry.resolve("model-b:prod") == "uri-v1"
    
    # Update 'prod' to v2
    temp_registry.promote("model-b", v2, "prod")
    assert temp_registry.resolve("model-b:prod") == "uri-v2"

def test_record_run(temp_registry):
    """Test recording pipeline runs."""
    run_id = "run-123"
    spec = {"epochs": 10}
    
    temp_registry.record_run(run_id, spec, "running")
    status = temp_registry.get_run(run_id)
    
    assert status.status == "running"
    assert status.ended_at is None
    
    temp_registry.update_run_status(run_id, "completed", ended=True)
    status_end = temp_registry.get_run(run_id)
    
    assert status_end.status == "completed"
    assert status_end.ended_at is not None

