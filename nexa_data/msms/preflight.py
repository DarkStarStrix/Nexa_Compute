"""Preflight checks before pipeline execution."""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

import psutil


def check_memory(min_gb: float = 4.0, max_gb: float = 20.0) -> Tuple[bool, str]:
    """Check available system memory.

    Args:
        min_gb: Minimum required memory in GB
        max_gb: Maximum recommended memory usage in GB

    Returns:
        Tuple of (success, message)
    """
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)

    if available_gb < min_gb:
        return False, f"Insufficient memory: {available_gb:.1f}GB available, {min_gb}GB required"

    if available_gb > max_gb:
        return True, f"Memory OK: {available_gb:.1f}GB available (recommend <{max_gb}GB for safety)"

    return True, f"Memory OK: {available_gb:.1f}GB available ({total_gb:.1f}GB total)"


def check_disk_space(path: Path, min_gb: float = 10.0) -> Tuple[bool, str]:
    """Check available disk space.

    Args:
        path: Path to check (usually output directory)
        min_gb: Minimum required free space in GB

    Returns:
        Tuple of (success, message)
    """
    # Create directory if it doesn't exist for disk space check
    path.mkdir(parents=True, exist_ok=True)
    
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)

    if free_gb < min_gb:
        return False, f"Insufficient disk space: {free_gb:.1f}GB free, {min_gb}GB required"

    return True, f"Disk space OK: {free_gb:.1f}GB free"


def check_hdf5_file(path: Path) -> Tuple[bool, str]:
    """Ping test for HDF5 file accessibility and complexity analysis.

    Args:
        path: Path to HDF5 file

    Returns:
        Tuple of (success, message)
    """
    if not path.exists():
        return False, f"HDF5 file not found: {path}"

    try:
        import h5py
        import numpy as np

        with h5py.File(path, "r") as f:
            # Quick read test - just check if file opens and has expected structure
            if "name" in f and "spectrum" in f:
                # Table format analysis
                size_gb = path.stat().st_size / (1024**3)
                
                # Sample complexity analysis
                num_samples = min(100, len(f["spectrum"]))
                if num_samples > 0:
                    indices = np.linspace(0, len(f["spectrum"]) - 1, num_samples, dtype=int)
                    spectra = f["spectrum"][indices]
                    
                    total_peaks = 0
                    max_peaks = 0
                    for s in spectra:
                        mzs = s[0, :]
                        valid_peaks = np.count_nonzero(mzs > 0)
                        total_peaks += valid_peaks
                        max_peaks = max(max_peaks, valid_peaks)
                    
                    avg_peaks = total_peaks / num_samples
                    complexity = f"avg_peaks={avg_peaks:.1f}, max_peaks={max_peaks}"
                else:
                    complexity = "empty"

                return True, f"HDF5 file OK: {size_gb:.2f}GB, table format, {complexity}"
            
            elif "spectra" in f:
                # Nested format analysis (lighter check)
                size_gb = path.stat().st_size / (1024**3)
                return True, f"HDF5 file OK: {size_gb:.2f}GB, nested format"
            else:
                return False, f"HDF5 file structure unexpected: missing 'spectrum' or 'spectra' group"
    except Exception as e:
        return False, f"HDF5 file error: {e}"


def run_preflight_checks(
    output_dir: Path,
    hdf5_paths: list[Path],
    min_memory_gb: float = 4.0,
    max_memory_gb: float = 20.0,
    min_disk_gb: float = 10.0,
) -> Tuple[bool, list[str]]:
    """Run all preflight checks.

    Args:
        output_dir: Output directory for shards
        hdf5_paths: List of HDF5 input files
        min_memory_gb: Minimum required memory
        max_memory_gb: Maximum recommended memory
        min_disk_gb: Minimum required disk space

    Returns:
        Tuple of (all_passed, list of messages)
    """
    messages = []
    all_passed = True

    # Memory check
    mem_ok, mem_msg = check_memory(min_memory_gb, max_memory_gb)
    messages.append(f"Memory: {mem_msg}")
    if not mem_ok:
        all_passed = False

    # Disk space check
    disk_ok, disk_msg = check_disk_space(output_dir, min_disk_gb)
    messages.append(f"Disk: {disk_msg}")
    if not disk_ok:
        all_passed = False

    # HDF5 file checks
    for hdf5_path in hdf5_paths:
        hdf5_ok, hdf5_msg = check_hdf5_file(hdf5_path)
        messages.append(f"HDF5 ({hdf5_path.name}): {hdf5_msg}")
        if not hdf5_ok:
            all_passed = False

    return all_passed, messages

