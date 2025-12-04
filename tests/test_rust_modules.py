"""Tests for Rust modules via Python bindings."""
import warnings
# Suppress pynvml deprecation warning from PyTorch before any imports
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)

import pytest
import json
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# --- Mocks for Rust Modules ---

@pytest.fixture
def mock_rust_data_core():
    """Mocks the low-level nexa_data_core Rust extension."""
    with patch('nexa_compute.data.rust_core._rust') as mock:
        mock.convert_csv_to_parquet = MagicMock(return_value=None)
        mock.shuffle_and_split = MagicMock(return_value=[[0, 1], [2, 3]])
        mock.parallel_process_files = MagicMock(return_value=None)
        mock.compute_stats_json = MagicMock(return_value=json.dumps({
            "total_rows": 100, 
            "columns": {
                "col1": {"count": 100, "min": 0.0, "max": 1.0, "sum": 50.0}
            }
        }))
        yield mock

@pytest.fixture
def mock_rust_data_quality():
    """Mocks the low-level nexa_data_quality Rust extension."""
    with patch('nexa_compute.data.quality_core._rust') as mock:
        # filter_batch returns a JSON string with stats or paths? 
        # Checking wrapper: wrapper returns hardcoded "filtered.parquet", {} for now.
        # But it calls _rust.filter_batch first.
        mock.filter_batch = MagicMock(return_value=json.dumps({
            "output_path": "filtered.parquet",
            "stats": {"kept": 10, "rejected": 2}
        }))
        mock.deduplicate_batch = MagicMock(return_value="deduped.parquet")
        yield mock

@pytest.fixture
def mock_rust_train_pack():
    """Mocks the low-level nexa_train_pack Rust extension."""
    with patch('nexa_compute.data.pack_core._rust') as mock:
        mock.pack_sequences = MagicMock(return_value="packed_shards_manifest.json")
        yield mock

@pytest.fixture
def mock_rust_stats():
    """Mocks the low-level nexa_stats Rust extension."""
    with patch('nexa_compute.data.stats_core._rust', create=True) as mock:
        mock.ks_test = MagicMock(return_value=0.05)
        mock.psi = MagicMock(return_value=0.1)
        mock.compute_histogram = MagicMock(return_value=json.dumps({"bins": [0.0, 1.0, 2.0], "counts": [5, 5]}))
        mock.compute_reductions = MagicMock(return_value=json.dumps({"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0, "p50": 0.5, "p95": 0.95, "p99": 0.99}))
        yield mock

# --- Tests ---

def test_data_core_conversion(mock_rust_data_core):
    from nexa_compute.data.rust_core import rust_core
    
    rust_core.convert_csv_to_parquet("input.csv", "output.parquet", batch_size=2048)
    mock_rust_data_core.convert_csv_to_parquet.assert_called_once_with("input.csv", "output.parquet", 2048)

def test_data_core_shuffle(mock_rust_data_core):
    from nexa_compute.data.rust_core import rust_core
    
    splits = rust_core.shuffle_and_split(100, [0.8, 0.2], seed=123)
    assert len(splits) == 2
    mock_rust_data_core.shuffle_and_split.assert_called_once_with(100, [0.8, 0.2], 123)

def test_data_core_stats(mock_rust_data_core):
    from nexa_compute.data.rust_core import rust_core
    
    stats = rust_core.compute_stats("data.parquet")
    assert stats["total_rows"] == 100
    assert "columns" in stats
    mock_rust_data_core.compute_stats_json.assert_called_once_with("data.parquet")

def test_quality_core_filter(mock_rust_data_quality):
    from nexa_compute.data.quality_core import quality_core
    
    config = {"min_length": 10, "bad_words": ["bad"]}
    out_path, stats = quality_core.filter_batch("input.parquet", config)
    
    # Current placeholder implementation returns fixed values
    assert out_path == "filtered.parquet"
    assert stats == {}
    
    # Verify Rust call
    mock_rust_data_quality.filter_batch.assert_called_once()
    args = mock_rust_data_quality.filter_batch.call_args[0]
    assert args[0] == "input.parquet"
    assert json.loads(args[1]) == config

def test_quality_core_dedup(mock_rust_data_quality):
    from nexa_compute.data.quality_core import quality_core
    
    res = quality_core.deduplicate_batch("input.parquet")
    assert res == "deduped.parquet"
    mock_rust_data_quality.deduplicate_batch.assert_called_once_with("input.parquet")

def test_pack_core_packing(mock_rust_train_pack):
    from nexa_compute.data.pack_core import pack_core
    
    shards = ["shard1.parquet", "shard2.parquet"]
    config = {"context_length": 2048, "seed": 42}
    
    res = pack_core.pack_sequences(shards, config)
    assert res == "packed_shards_manifest.json"
    
    mock_rust_train_pack.pack_sequences.assert_called_once()
    args = mock_rust_train_pack.pack_sequences.call_args[0]
    assert args[0] == shards
    assert json.loads(args[1]) == config

def test_stats_core_ks(mock_rust_stats):
    from nexa_compute.data.stats_core import stats_core
    
    ref_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    cur_data = [1.1, 2.1, 3.1, 4.1, 5.1]
    
    result = stats_core.ks_test(ref_data, cur_data)
    assert isinstance(result, float)
    mock_rust_stats.ks_test.assert_called_once_with(ref_data, cur_data)

def test_stats_core_psi(mock_rust_stats):
    from nexa_compute.data.stats_core import stats_core
    
    ref_data = [1.0, 2.0, 3.0]
    cur_data = [1.1, 2.1, 3.1]
    
    result = stats_core.psi(ref_data, cur_data)
    assert isinstance(result, float)
    mock_rust_stats.psi.assert_called_once_with(ref_data, cur_data)

def test_stats_core_histogram(mock_rust_stats):
    from nexa_compute.data.stats_core import stats_core
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = stats_core.histogram(data, bins=5)
    
    assert "bins" in result
    assert "counts" in result
    mock_rust_stats.compute_histogram.assert_called_once_with(data, 5)

def test_stats_core_reduce(mock_rust_stats):
    from nexa_compute.data.stats_core import stats_core
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = stats_core.reduce(data)
    
    assert "mean" in result
    mock_rust_stats.compute_reductions.assert_called_once_with(data)
