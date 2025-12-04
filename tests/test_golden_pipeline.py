"""Golden pipeline test: data → distill → pack → train → eval.

This test verifies the core pipeline executes deterministically and quickly.
If this fails, the repo is "red" per Scaling Policy Section 7.
"""

import warnings
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# Mock Rust modules for CI environments
@pytest.fixture
def mock_rust_modules():
    """Mock Rust extensions for testing."""
    with patch('nexa_compute.data.rust_core._rust') as mock_core, \
         patch('nexa_compute.data.rust_quality.rust_quality') as mock_quality, \
         patch('nexa_compute.data.pack_core._rust') as mock_pack, \
         patch('nexa_compute.data.stats_core._rust') as mock_stats:
        
        # Mock data core
        mock_core.shuffle_and_split = MagicMock(return_value=[[0, 1, 2], [3, 4]])
        mock_core.compute_stats_json = MagicMock(return_value='{"total_rows": 5}')
        
        # Mock quality
        mock_quality.filter_batch = MagicMock(return_value=type('obj', (object,), {
            'total': 5, 'kept': 5, 'rejected_length': 0, 
            'rejected_pattern': 0, 'rejected_dedup': 0,
            'length_histogram': {}
        })())
        
        # Mock pack
        mock_pack.pack_sequences = MagicMock(return_value='{"shards": []}')
        
        # Mock stats
        mock_stats.compute_reductions = MagicMock(return_value='{"mean": 0.5}')
        
        yield {
            'core': mock_core,
            'quality': mock_quality,
            'pack': mock_pack,
            'stats': mock_stats,
        }


def test_golden_pipeline_deterministic(mock_rust_modules, tmp_path):
    """Test that the golden pipeline executes deterministically."""
    from nexa_compute.config import load_config
    from nexa_compute.data import DataPipeline, DEFAULT_REGISTRY
    from nexa_compute.evaluation import Evaluator
    from nexa_compute.models import DEFAULT_MODEL_REGISTRY
    from nexa_compute.orchestration import TrainingPipeline
    from nexa_compute.training import Trainer
    
    # Create minimal config
    config_dict = {
        "data": {
            "dataset_name": "synthetic",
            "dataset_version": "v1",
            "batch_size": 4,
            "num_workers": 0,
            "preprocessing": {"num_features": 8, "num_classes": 2},
            "split": {"train": 0.8, "validation": 0.2, "test": 0.0},
        },
        "model": {
            "name": "mlp",
            "args": {"input_dim": 8, "hidden_dims": [16], "num_classes": 2},
        },
        "training": {
            "epochs": 1,
            "optimizer": {"name": "adam", "lr": 0.001},
            "scheduler": {"name": None},
            "checkpoint": {"dir": str(tmp_path / "checkpoints"), "monitor": "val_loss", "mode": "min"},
            "distributed": {"world_size": 1, "seed": 42},
            "logging": {"level": "INFO", "log_dir": str(tmp_path / "logs")},
        },
        "evaluation": {
            "batch_size": 4,
            "metrics": ["accuracy"],
            "output_dir": str(tmp_path / "eval"),
        },
        "experiment": {"name": "golden_test", "tags": {}},
    }
    
    # Save config
    import yaml
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_dict, f)
    
    # Load config
    config = load_config(config_path)
    
    # Step 1: Data pipeline
    data_pipeline = DataPipeline(config.data, registry=DEFAULT_REGISTRY)
    train_loader = data_pipeline.dataloader("train")
    val_loader = data_pipeline.dataloader("validation")
    
    assert len(train_loader) > 0, "Train loader must have batches"
    assert len(val_loader) > 0, "Val loader must have batches"
    
    # Step 2: Model
    model = DEFAULT_MODEL_REGISTRY.build(config.model)
    assert isinstance(model, nn.Module)
    
    # Step 3: Training (minimal)
    trainer = Trainer(config, callbacks=[])
    trainer.fit(model, train_loader, val_loader)
    
    # Step 4: Evaluation
    evaluator = Evaluator(config.evaluation)
    metrics = evaluator.evaluate(model, val_loader)
    
    assert "val_accuracy" in metrics or "accuracy" in metrics
    assert isinstance(metrics.get("val_accuracy", metrics.get("accuracy")), float)
    
    # Verify determinism: run twice, should get same results
    model2 = DEFAULT_MODEL_REGISTRY.build(config.model)
    trainer2 = Trainer(config, callbacks=[])
    trainer2.fit(model2, train_loader, val_loader)
    metrics2 = evaluator.evaluate(model2, val_loader)
    
    # Metrics should be deterministic (within floating point tolerance)
    acc1 = metrics.get("val_accuracy", metrics.get("accuracy", 0))
    acc2 = metrics2.get("val_accuracy", metrics2.get("accuracy", 0))
    assert abs(acc1 - acc2) < 0.01, "Pipeline must be deterministic"


def test_golden_pipeline_fast(mock_rust_modules, tmp_path):
    """Test that the golden pipeline executes quickly (< 30 seconds)."""
    import time
    
    start = time.time()
    test_golden_pipeline_deterministic(mock_rust_modules, tmp_path)
    elapsed = time.time() - start
    
    assert elapsed < 30.0, f"Golden pipeline must complete quickly, took {elapsed:.2f}s"


def test_golden_pipeline_with_manifest(mock_rust_modules, tmp_path):
    """Test that pipeline produces run manifest."""
    from nexa_compute.config import load_config
    from nexa_compute.core.manifests import RunManifest
    from nexa_compute.orchestration import TrainingPipeline
    
    # Create minimal config
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
            "epochs": 1,
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
        "experiment": {"name": "manifest_test", "tags": {}},
    }
    
    import yaml
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_dict, f)
    
    config = load_config(config_path)
    config.output_directory = lambda: tmp_path / "runs"
    
    pipeline = TrainingPipeline(config)
    artifacts = pipeline.run(enable_evaluation=True)
    
    # Verify manifest was created
    manifest_dir = tmp_path / "runs"
    manifest_files = list(manifest_dir.glob("*.json"))
    assert len(manifest_files) > 0, "Pipeline must create run manifest"
    
    # Verify manifest is valid
    manifest = RunManifest.load(manifest_files[0])
    assert manifest.run_id is not None
    assert manifest.config_snapshot is not None
    assert manifest.exit_status in ["completed", "running", "failed"]

