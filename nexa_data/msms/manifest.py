"""Dataset manifest builder with full execution provenance."""

import glob
import hashlib
import json
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .config import PipelineConfig


def get_hardware_info() -> Dict:
    """Collect hardware information.
    
    Returns:
        Dictionary with hardware information
    """
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
    }
    
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            info["memory_total_gb"] = mem.total / (1024**3)
            info["memory_available_gb"] = mem.available / (1024**3)
        except Exception:
            pass
    
    return info


def compute_config_hash(config_path: Optional[Path]) -> Optional[str]:
    """Compute SHA256 hash of config file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        SHA256 hash as hex string or None
    """
    if not config_path or not config_path.exists():
        return None
    
    h = hashlib.sha256()
    with open(config_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def compute_hdf5_hashes(hdf5_paths: list[str]) -> Dict[str, str]:
    """Compute SHA256 hashes of HDF5 input files.
    
    Args:
        hdf5_paths: List of HDF5 file paths
        
    Returns:
        Dictionary mapping file paths to hashes
    """
    hashes = {}
    for path_str in hdf5_paths:
        path = Path(path_str)
        if path.exists():
            h = hashlib.sha256()
            with open(path, "rb") as f:
                # Read in chunks for large files
                while True:
                    chunk = f.read(1 << 20)  # 1MB chunks
                    if not chunk:
                        break
                    h.update(chunk)
            hashes[path_str] = h.hexdigest()
    return hashes


def compute_dataset_hash(shard_manifests: list) -> str:
    """Compute SHA256 hash of concatenated shard hashes.

    Args:
        shard_manifests: List of shard manifest dictionaries

    Returns:
        SHA256 hash as hex string
    """
    h = hashlib.sha256()
    
    # Sort shards by split and index for deterministic hashing
    sorted_shards = sorted(
        shard_manifests,
        key=lambda m: (m.get("split", ""), m.get("shard_index", 0)),
    )
    
    for manifest in sorted_shards:
        checksum = manifest.get("checksum", "")
        if checksum.startswith("sha256:"):
            checksum = checksum[7:]  # Remove "sha256:" prefix
        h.update(checksum.encode("utf-8"))
    
    return h.hexdigest()


def build_dataset_manifest(
    dataset_name: str,
    output_root: Path,
    cfg: PipelineConfig,
    run_id: Optional[str] = None,
    mode: str = "local",
    profile: Optional[str] = None,
    execution_metadata: Optional[Dict] = None,
) -> Path:
    """Build dataset manifest from shard manifests with full execution provenance.

    Args:
        dataset_name: Dataset name
        output_root: Root output directory
        cfg: Pipeline configuration
        run_id: Optional run ID to filter shards
        mode: Execution mode (local/cloud)
        profile: Profile name used
        execution_metadata: Optional additional execution metadata

    Returns:
        Path to dataset manifest file
    """
    shard_manifests = []

    for split in ["train", "val", "test"]:
        split_dir = output_root / split
        if not split_dir.exists():
            continue

        pattern = str(split_dir / "*.manifest.json")
        for mf_path in sorted(glob.glob(pattern)):
            with open(mf_path) as f:
                manifest = json.load(f)
                # Filter by run_id if provided
                if run_id is None or manifest.get("run_id") == run_id:
                    shard_manifests.append(manifest)

    # Compute full dataset hash from shard hashes
    dataset_hash = compute_dataset_hash(shard_manifests) if shard_manifests else ""

    # Collect execution metadata
    config_hash = compute_config_hash(cfg.config_path)
    hdf5_hashes = compute_hdf5_hashes(cfg.canonical_hdf5)
    hardware_info = get_hardware_info()

    dataset_manifest = {
        "dataset": dataset_name,
        "version": cfg.schema_version,
        "created_at": datetime.utcnow().isoformat(),
        "config_file": str(cfg.config_path) if cfg.config_path else None,
        "config_hash": f"sha256:{config_hash}" if config_hash else None,
        "run_id": run_id,
        "mode": mode,
        "profile": profile,
        "hardware": hardware_info,
        "hdf5_hashes": hdf5_hashes,
        "shards": shard_manifests,
        "dataset_hash": f"sha256:{dataset_hash}",
        "num_shards": len(shard_manifests),
        "status": "complete",
    }

    # Add execution metadata if provided
    if execution_metadata:
        dataset_manifest.update(execution_metadata)

    out = output_root / "dataset_manifest.json"
    with open(out, "w") as f:
        json.dump(dataset_manifest, f, indent=2, default=str)

    print(f"[OK] wrote dataset manifest to {out}")
    print(f"[OK] dataset hash: sha256:{dataset_hash[:16]}...")
    return out

