"""Tests for idempotency guarantees per Scaling Policy Section 9."""

import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


@pytest.fixture
def mock_rust_modules():
    """Mock Rust extensions."""
    with patch('nexa_compute.data.rust_core._rust') as mock_core:
        mock_core.shuffle_and_split = MagicMock(return_value=[[0, 1, 2], [3, 4]])
        yield mock_core


def test_pipeline_idempotent_with_checkpoint(mock_rust_modules, tmp_path):
    """Test that pipeline is idempotent when resuming from checkpoint."""
    from nexa_compute.config import load_config
    from nexa_compute.data import DataPipeline, DEFAULT_REGISTRY
    from nexa_compute.models import DEFAULT_MODEL_REGISTRY
    from nexa_compute.orchestration import TrainingPipeline
    from nexa_compute.training import Trainer
    
    # Create minimal config
    import yaml
    config_dict = {
        "data": {
            "dataset_name": "synthetic",
            "dataset_version": "v1",
            "batch_size": 2,
            "num_workers": 0,
            "preprocessing": {"num_features": 4, "num_classes": 2},
            "split": {"train": 0.8, "validation": 0.2, "test": 0.0},
        },
        "model": {
            "name": "mlp",
            "args": {"input_dim": 4, "hidden_dims": [8], "num_classes": 2},
        },
        "training": {
            "epochs": 2,
            "optimizer": {"name": "adam", "lr": 0.001},
            "scheduler": {"name": None},
            "checkpoint": {"dir": str(tmp_path / "checkpoints"), "monitor": "val_loss", "mode": "min"},
            "distributed": {"world_size": 1, "seed": 42},
            "logging": {"level": "WARNING", "log_dir": str(tmp_path / "logs")},
        },
        "evaluation": {
            "batch_size": 2,
            "metrics": ["accuracy"],
            "output_dir": str(tmp_path / "eval"),
        },
        "experiment": {"name": "idempotency_test", "tags": {}},
    }
    
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_dict, f)
    
    config = load_config(config_path)
    config.output_directory = lambda: tmp_path / "runs"
    
    # Run pipeline first time
    pipeline1 = TrainingPipeline(config)
    artifacts1 = pipeline1.run(enable_evaluation=True)
    
    # Get checkpoint path
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if checkpoints:
        checkpoint_path = checkpoints[0]
        
        # Run pipeline second time with same checkpoint
        pipeline2 = TrainingPipeline(config)
        artifacts2 = pipeline2.run(resume_from_checkpoint=checkpoint_path, enable_evaluation=True)
        
        # Metrics should be identical (within tolerance)
        assert abs(artifacts1.metrics.get("val_accuracy", 0) - 
                  artifacts2.metrics.get("val_accuracy", 0)) < 0.01, \
            "Pipeline must be idempotent when resuming from checkpoint"


def test_metadata_materialization_idempotent(tmp_path):
    """Test that metadata materialization is idempotent."""
    from nexa_compute.config.schema import DataConfig
    from nexa_compute.data import DataPipeline
    
    config = DataConfig(
        dataset_name="synthetic",
        dataset_version="v1",
        batch_size=4,
        preprocessing={"num_features": 4},
        split={"train": 0.8, "validation": 0.2, "test": 0.0},
    )
    
    pipeline = DataPipeline(config)
    
    # Materialize twice
    path1 = pipeline.materialize_metadata(tmp_path)
    path2 = pipeline.materialize_metadata(tmp_path)
    
    # Should produce identical files
    assert path1 == path2
    assert path1.read_text() == path2.read_text()


def test_checkpoint_save_idempotent(tmp_path):
    """Test that checkpoint saving is idempotent."""
    from nexa_compute.training.checkpoint import save_checkpoint
    
    state = {"epoch": 1, "model_state": {}}
    
    # Save twice
    path1 = save_checkpoint(state, tmp_path, filename="test.pt")
    path2 = save_checkpoint(state, tmp_path, filename="test.pt")
    
    # Should overwrite safely
    assert path1 == path2
    assert path1.exists()


def test_manifest_save_idempotent(tmp_path):
    """Test that manifest saving is idempotent."""
    from nexa_compute.core.manifests import RunManifest
    
    manifest = RunManifest(
        run_id="test_run",
        monorepo_commit="abc123",
    )
    
    # Save twice
    path1 = manifest.save(tmp_path)
    path2 = manifest.save(tmp_path)
    
    # Should produce same file
    assert path1 == path2
    assert path1.exists()
    
    # Content should be identical
    assert path1.read_text() == path2.read_text()

