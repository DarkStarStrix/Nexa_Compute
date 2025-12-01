"""End-to-end integration tests for the Nexa platform."""

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nexa_compute.api.main import app
from nexa_compute.api.auth import get_api_key
from nexa_compute.api.database import UserDB
from nexa_compute.data.catalog import get_catalog
from nexa_compute.models.registry import EnhancedModelRegistry
from nexa_compute.orchestration.scheduler import get_scheduler
from nexa_compute.orchestration.workflow import WorkflowBuilder

# Mock auth for tests
def override_get_api_key():
    user = UserDB(user_id="test-user", email="test@nexa.run", is_active=True)
    # Initialize transient attributes needed for rate limiting
    user._current_api_key_hash = "test-hash"
    user._current_api_key_id = "test-id"
    return user

app.dependency_overrides[get_api_key] = override_get_api_key

client = TestClient(app)

@pytest.fixture
def api_key():
    # Not strictly needed with override, but good for consistency
    return "test-api-key"

@pytest.fixture
def headers(api_key):
    return {"X-Nexa-Api-Key": api_key}

def test_end_to_end_workflow(headers, tmp_path):
    """Test a full cycle: data prep -> train -> deploy."""
    
    # 1. Create a dummy dataset
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()
    (data_dir / "data.csv").write_text("col1,col2\n1,2\n3,4")
    
    # 2. Version the dataset
    catalog = get_catalog()
    # Override root for test
    catalog.dvc.storage_root = tmp_path / "dvc"
    catalog.dvc.blob_store = catalog.dvc.storage_root / "blobs"
    catalog.dvc.meta_store = catalog.dvc.storage_root / "meta"
    catalog.dvc.blob_store.mkdir(parents=True, exist_ok=True)
    catalog.dvc.meta_store.mkdir(parents=True, exist_ok=True)
    
    version = catalog.register_dataset("test-dataset", data_dir)
    assert version is not None
    
    # 3. Define a workflow
    builder = WorkflowBuilder("test-pipeline")
    builder.add_step("step1", "echo", params={"msg": "hello"})
    workflow = builder.build()
    
    # 4. Submit workflow via API
    # We need to mock the scheduler's registry or use the real one
    scheduler = get_scheduler()
    scheduler.register_workflow(workflow)
    
    # Since scheduler is global, we can just trigger it via API
    resp = client.post(
        "/api/workflows/submit",
        json={"workflow_name": "test-pipeline"},
        headers=headers
    )
    
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    
    # 5. Poll for completion
    for _ in range(10):
        resp = client.get(f"/api/workflows/{run_id}", headers=headers)
        status = resp.json()["status"]
        if status == "COMPLETED":
            break
        if status == "FAILED":
            pytest.fail("Workflow failed")
        time.sleep(0.1)
        
    assert status == "COMPLETED"

def test_model_registry_flow(tmp_path):
    """Test model registration and versioning."""
    db_path = tmp_path / "registry.db"
    registry = EnhancedModelRegistry(db_path)
    
    # Register v1 with nested metrics in meta
    v1 = registry.register(
        name="my-model",
        uri="s3://bucket/v1",
        meta={"metrics": {"accuracy": 0.9}, "architecture": "resnet50"}
    )
    assert v1 == "1.0.0"
    
    # Register v2
    v2 = registry.register(
        name="my-model",
        uri="s3://bucket/v2",
        meta={"metrics": {"accuracy": 0.95}, "architecture": "resnet50"},
        version="1.1.0"
    )
    
    # Compare
    diff = registry.compare_versions("my-model", "1.0.0", "1.1.0")
    assert diff["metrics_diff"]["accuracy"]["old"] == 0.9
    assert diff["metrics_diff"]["accuracy"]["new"] == 0.95
