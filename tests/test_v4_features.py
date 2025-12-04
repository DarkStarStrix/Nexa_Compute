"""Tests for V4 specification features."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_run_manifest_v4_compliance():
    """Test that run manifests include all V4 required fields."""
    from nexa_compute.core.manifests import RunManifest
    
    manifest = RunManifest(
        run_id="test_v4_run",
        monorepo_commit="abc123",
        config_snapshot={"model": {"name": "test"}},
        dataset_version="v1.0",
        hardware_specs={"gpu_count": 1},
        shard_list=["shard1.parquet", "shard2.parquet"],
        tokens_processed=1000000,
        metrics={"loss": 0.5, "accuracy": 0.95},
    )
    
    # Verify all V4 fields present
    assert manifest.run_id is not None
    assert manifest.timestamps is not None
    assert manifest.monorepo_commit is not None
    assert manifest.config_snapshot is not None
    assert manifest.dataset_version is not None
    assert manifest.hardware_specs is not None
    assert manifest.shard_list is not None
    assert manifest.tokens_processed is not None
    assert manifest.metrics is not None
    assert manifest.exit_status is not None


def test_manifest_save_location():
    """Test that manifests save to runs/<run_id>.json as per V4 spec."""
    from nexa_compute.core.manifests import RunManifest
    
    manifest = RunManifest(run_id="test_location")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir) / "runs"
        path = manifest.save(runs_dir)
        
        # Verify path structure
        assert path.parent == runs_dir
        assert path.name == f"{manifest.run_id}.json"
        assert path.exists()


def test_manifest_resume_info():
    """Test resume info tracking in manifest."""
    from nexa_compute.core.manifests import RunManifest
    
    manifest = RunManifest(
        run_id="test_resume",
        resume_info={"path": "/path/to/checkpoint", "step": 1000}
    )
    
    assert manifest.resume_info is not None
    assert manifest.resume_info["path"] == "/path/to/checkpoint"
    assert manifest.resume_info["step"] == 1000

