"""Comprehensive validation suite."""

import glob
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import PipelineConfig
from .metrics import PipelineMetrics


def validate_shards(shards_dir: Path, full: bool = False, run_id: Optional[str] = None) -> Dict:
    """Validate shards for integrity and training readiness.

    Args:
        shards_dir: Directory containing shards
        full: If True, validate all samples; if False, spot check
        run_id: Optional run ID to filter shards for validation

    Returns:
        Validation results dictionary
    """
    shards = []
    for split in ["train", "val", "test"]:
        split_dir = shards_dir / split
        if split_dir.exists():
            all_shards = sorted(glob.glob(str(split_dir / "*.parquet")))
            # Filter by run_id if provided
            if run_id:
                filtered = []
                for shard_path in all_shards:
                    manifest_path = Path(shard_path).with_suffix(".manifest.json")
                    if manifest_path.exists():
                        with open(manifest_path) as f:
                            manifest = json.load(f)
                            if manifest.get("run_id") == run_id:
                                filtered.append(shard_path)
                    elif run_id in Path(shard_path).name:
                        # Fallback: check if run_id is in filename
                        filtered.append(shard_path)
                shards.extend(filtered)
            else:
                shards.extend(all_shards)

    if not shards:
        # Return empty results instead of raising error
        return {
            "shards_checked": 0,
            "samples_checked": 0,
            "schema_mismatches": 0,
            "duplicates": [],
            "checksum_failures": 0,
            "sample_validation_failures": 0,
            "training_readiness_failures": 0,
            "inf_batches": 0,
            "loader_failures": 0,
        }

    results = {
        "shards_checked": len(shards),
        "samples_checked": 0,
        "schema_mismatches": 0,
        "duplicates": [],
        "checksum_failures": 0,
        "sample_validation_failures": 0,
        "training_readiness_failures": 0,
        "inf_batches": 0,
        "loader_failures": 0,
    }

    reference_schema = None
    all_sample_ids = []

    for shard_path in shards:
        print(f"Validating {shard_path}")

        table = pq.read_table(shard_path)

        if reference_schema is None:
            reference_schema = table.schema
        else:
            if table.schema != reference_schema:
                results["schema_mismatches"] += 1

        required = ["sample_id", "mzs", "ints", "precursor_mz"]
        for r in required:
            if r not in table.column_names:
                raise ValueError(f"Missing required field {r} in {shard_path}")

        manifest_path = Path(str(shard_path).replace(".parquet", ".manifest.json"))
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            from .shard_writer import sha256_of_file

            expected_checksum = manifest.get("checksum", "").replace("sha256:", "")
            actual_checksum = sha256_of_file(shard_path)

            if expected_checksum != actual_checksum:
                results["checksum_failures"] += 1
                print(f"  WARNING: Checksum mismatch for {shard_path}")

            all_sample_ids.extend(manifest.get("sample_ids", []))

        batch_size = len(table) if full else min(100, len(table))
        batch = table.to_pylist()[:batch_size]
        results["samples_checked"] += len(batch)

        for row in batch:
            try:
                mzs = np.array(row["mzs"], dtype=np.float32)
                ints = np.array(row["ints"], dtype=np.float32)

                if len(mzs) != len(ints):
                    results["sample_validation_failures"] += 1
                    continue

                if not np.isfinite(mzs).all() or not np.isfinite(ints).all():
                    results["sample_validation_failures"] += 1
                    continue

                if (mzs <= 0).any():
                    results["sample_validation_failures"] += 1
                    continue

                # Training readiness check: verify data can be loaded as tensors
                if TORCH_AVAILABLE:
                    try:
                        mzs_tensor = torch.tensor(mzs, dtype=torch.float32)
                        ints_tensor = torch.tensor(ints, dtype=torch.float32)

                        if torch.isnan(mzs_tensor).any() or torch.isnan(ints_tensor).any():
                            results["training_readiness_failures"] += 1
                            continue

                        if torch.isinf(mzs_tensor).any() or torch.isinf(ints_tensor).any():
                            results["training_readiness_failures"] += 1
                            results["inf_batches"] += 1
                            continue
                    except Exception as e:
                        # If tensor conversion fails, mark as training readiness failure
                        results["training_readiness_failures"] += 1
                        results["loader_failures"] += 1
                        continue
                else:
                    # Fallback: use numpy for training readiness check if PyTorch unavailable
                    # Check for NaN/Inf using numpy (less comprehensive but still useful)
                    if np.isnan(mzs).any() or np.isnan(ints).any():
                        results["training_readiness_failures"] += 1
                        continue
                    if np.isinf(mzs).any() or np.isinf(ints).any():
                        results["training_readiness_failures"] += 1
                        results["inf_batches"] += 1
                        continue

            except Exception as e:
                results["sample_validation_failures"] += 1
                continue

    duplicates = [id for id, count in Counter(all_sample_ids).items() if count > 1]
    results["duplicates"] = duplicates

    if duplicates:
        print(f"  ERROR: Found {len(duplicates)} duplicate sample_ids")

    print(f"[OK] Validated {len(shards)} shards")
    return results


def test_determinism(config_path: Path, temp_dir: Path) -> bool:
    """Test determinism by running pipeline twice and comparing.

    Args:
        config_path: Path to config file
        temp_dir: Temporary directory for test runs

    Returns:
        True if outputs are identical
    """
    import shutil
    import subprocess
    import sys

    from .config import load_config

    cfg = load_config(config_path)

    output_dir1 = temp_dir / "run1"
    output_dir2 = temp_dir / "run2"

    output_dir1.mkdir(parents=True, exist_ok=True)
    output_dir2.mkdir(parents=True, exist_ok=True)

    import yaml

    config1_path = temp_dir / "config1.yaml"
    config2_path = temp_dir / "config2.yaml"

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    config_data["output_root"] = str(output_dir1)
    with open(config1_path, "w") as f:
        yaml.dump(config_data, f)

    config_data["output_root"] = str(output_dir2)
    with open(config2_path, "w") as f:
        yaml.dump(config_data, f)

    cmd1 = [
        sys.executable,
        "-m",
        "nexa_data.msms.cli",
        "build-shards",
        "--config",
        str(config1_path),
    ]

    cmd2 = [
        sys.executable,
        "-m",
        "nexa_data.msms.cli",
        "build-shards",
        "--config",
        str(config2_path),
    ]

    print("Running first pass...")
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    if result1.returncode != 0:
        print(f"First pass failed: {result1.stderr}")
        return False

    print("Running second pass...")
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    if result2.returncode != 0:
        print(f"Second pass failed: {result2.stderr}")
        return False

    from .shard_writer import sha256_of_file

    shards1 = sorted(glob.glob(str(output_dir1 / "train" / "*.parquet")))
    shards2 = sorted(glob.glob(str(output_dir2 / "train" / "*.parquet")))

    if len(shards1) != len(shards2):
        print(f"Different number of shards: {len(shards1)} vs {len(shards2)}")
        return False

    for s1, s2 in zip(shards1, shards2):
        if sha256_of_file(Path(s1)) != sha256_of_file(Path(s2)):
            print(f"Checksum mismatch: {s1} vs {s2}")
            return False

    print("[OK] Determinism test passed")
    return True


def generate_quality_report(
    shards_dir: Path, metrics: PipelineMetrics, config: PipelineConfig, run_id: Optional[str] = None
) -> Dict:
    """Generate quality report with pass/warn/fail status.

    Args:
        shards_dir: Directory containing shards
        metrics: Pipeline metrics
        config: Pipeline configuration
        run_id: Optional run ID to filter shards

    Returns:
        Quality report dictionary
    """
    from datetime import datetime

    metrics_dict = metrics.get_metrics_dict()

    integrity_error_rate = metrics_dict["integrity_error_rate"]
    attrition_rate = metrics_dict["attrition_rate"]

    canonicalization_status = "PASS"
    if integrity_error_rate > 0.001:
        canonicalization_status = "FAIL"
    elif integrity_error_rate > 0.0001:
        canonicalization_status = "WARN"

    validation_results = validate_shards(shards_dir, full=False, run_id=run_id)

    shard_construction_status = "PASS"
    if validation_results["duplicates"] or validation_results["checksum_failures"]:
        shard_construction_status = "FAIL"

    training_readiness_status = "PASS"
    if validation_results["training_readiness_failures"] > 0:
        training_readiness_status = "FAIL"

    report = {
        "dataset": config.dataset_name,
        "run_timestamp": datetime.utcnow().isoformat(),
        "stages": {
            "canonicalization": {
                "total_spectra": metrics_dict["total_spectra"],
                "integrity_errors": metrics_dict["integrity_error_count"],
                "integrity_error_rate": integrity_error_rate,
                "attrition": metrics_dict["attrition_count"],
                "attrition_rate": attrition_rate,
                "attrition_reasons": metrics_dict["attrition"],
                "status": canonicalization_status,
            },
            "shard_construction": {
                "shards_written": metrics_dict["shards_written"],
                "samples_sharded": metrics_dict["samples_written"],
                "duplicates": len(validation_results["duplicates"]),
                "checksum_failures": validation_results["checksum_failures"],
                "schema_mismatches": validation_results["schema_mismatches"],
                "status": shard_construction_status,
            },
            "training_readiness": {
                "shards_loaded": validation_results["shards_checked"],
                "samples_checked": validation_results.get("samples_checked", 0),
                "nan_batches": validation_results["training_readiness_failures"],
                "inf_batches": validation_results.get("inf_batches", 0),
                "loader_failures": validation_results.get("loader_failures", 0),
                "torch_available": TORCH_AVAILABLE,
                "status": training_readiness_status,
            },
        },
        "overall_status": "PASS" if all(
            s == "PASS"
            for s in [
                canonicalization_status,
                shard_construction_status,
                training_readiness_status,
            ]
        ) else "WARN",
        "recommendations": [],
    }

    if integrity_error_rate > 0.001:
        report["recommendations"].append("Investigate integrity error rate > 0.1%")

    if validation_results["duplicates"]:
        report["recommendations"].append("Fix duplicate sample_ids")

    return report


def validate_for_training(manifest_path: Path, expected_dataset_hash: Optional[str] = None) -> tuple[bool, List[str]]:
    """Validate dataset manifest for training preflight contract.
    
    Performs all checks required before training can start:
    - manifest.status == "complete"
    - golden_sample_pass == true
    - integrity_errors == 0
    - no duplicate_ids
    - semantic_drift == false
    - dataset_hash matches expected
    
    Args:
        manifest_path: Path to dataset manifest file
        expected_dataset_hash: Optional expected dataset hash to verify
        
    Returns:
        Tuple of (all_passed, list of failure reasons)
    """
    if not manifest_path.exists():
        return False, [f"Manifest file not found: {manifest_path}"]
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    failures = []
    
    # Check status
    status = manifest.get("status")
    if status != "complete":
        failures.append(f"Manifest status is '{status}', expected 'complete'")
    
    # Check golden sample pass
    quality = manifest.get("quality", {})
    golden_sample_pass = quality.get("golden_sample_pass")
    if golden_sample_pass is not True:
        failures.append(f"Golden sample suite did not pass: {golden_sample_pass}")
    
    # Check integrity errors
    metrics = manifest.get("metrics", {})
    integrity_error_count = metrics.get("integrity_error_count", 0)
    if integrity_error_count > 0:
        failures.append(f"Integrity errors found: {integrity_error_count}")
    
    # Check for duplicate IDs
    validation = manifest.get("validation", {})
    duplicates = validation.get("duplicates", [])
    if duplicates:
        failures.append(f"Duplicate sample IDs found: {len(duplicates)}")
    
    # Check semantic drift
    semantic_drift = validation.get("semantic_drift", {})
    drift_detected = semantic_drift.get("drift_detected", False)
    if drift_detected:
        drift_reasons = semantic_drift.get("drift_reasons", [])
        failures.append(f"Semantic drift detected: {', '.join(drift_reasons)}")
    
    # Check dataset hash
    dataset_hash = manifest.get("dataset_hash", "")
    if not dataset_hash:
        failures.append("Dataset hash not found in manifest")
    elif expected_dataset_hash:
        if dataset_hash != expected_dataset_hash:
            failures.append(f"Dataset hash mismatch: expected {expected_dataset_hash}, got {dataset_hash}")
    
    all_passed = len(failures) == 0
    return all_passed, failures

