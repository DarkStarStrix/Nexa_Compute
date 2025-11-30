"""Dry-run doctor mode for comprehensive system validation."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import msms_rust
    RUST_VALIDATOR_AVAILABLE = True
except ImportError:
    RUST_VALIDATOR_AVAILABLE = False

from .config import PipelineConfig, load_config

logger = logging.getLogger(__name__)


class DoctorCheck:
    """Single doctor check result."""

    def __init__(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        """Initialize doctor check.

        Args:
            name: Check name
            passed: Whether check passed
            message: Check message
            details: Optional additional details
        """
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


class DoctorMode:
    """Comprehensive system validation in doctor mode."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize doctor mode.

        Args:
            config_path: Optional path to config file
        """
        self.config_path = config_path
        self.checks: List[DoctorCheck] = []

    def check_config_schema(self) -> DoctorCheck:
        """Validate config file schema."""
        if not self.config_path:
            return DoctorCheck(
                "config_schema",
                False,
                "No config file provided",
            )

        if not self.config_path.exists():
            return DoctorCheck(
                "config_schema",
                False,
                f"Config file not found: {self.config_path}",
            )

        try:
            cfg = load_config(self.config_path)
            return DoctorCheck(
                "config_schema",
                True,
                f"Config schema valid: {cfg.dataset_name}",
                {
                    "dataset_name": cfg.dataset_name,
                    "schema_version": cfg.schema_version,
                },
            )
        except Exception as e:
            return DoctorCheck(
                "config_schema",
                False,
                f"Config schema validation failed: {e}",
            )

    def check_machine_compatibility(self) -> DoctorCheck:
        """Check machine compatibility."""
        details = {}
        issues = []

        if not PSUTIL_AVAILABLE:
            issues.append("psutil not available (memory checks disabled)")

        cpu_count = os.cpu_count()
        details["cpu_count"] = cpu_count

        if cpu_count and cpu_count < 2:
            issues.append(f"Insufficient CPU cores: {cpu_count}")

        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                mem_gb = mem.total / (1024**3)
                details["memory_total_gb"] = mem_gb

                if mem_gb < 4.0:
                    issues.append(f"Low memory: {mem_gb:.1f}GB")
            except Exception:
                pass

        platform_info = sys.platform
        details["platform"] = platform_info

        if issues:
            return DoctorCheck(
                "machine_compatibility",
                False,
                f"Compatibility issues: {', '.join(issues)}",
                details,
            )
        else:
            return DoctorCheck(
                "machine_compatibility",
                True,
                "Machine compatibility OK",
                details,
            )

    def check_hdf5_sample_read(self, cfg: Optional[PipelineConfig] = None) -> DoctorCheck:
        """Test HDF5 file reading."""
        if not H5PY_AVAILABLE:
            return DoctorCheck(
                "hdf5_sample_read",
                False,
                "h5py not available",
            )

        if not cfg:
            return DoctorCheck(
                "hdf5_sample_read",
                False,
                "No config provided for HDF5 check",
            )

        if not cfg.canonical_hdf5:
            return DoctorCheck(
                "hdf5_sample_read",
                False,
                "No HDF5 files specified in config",
            )

        try:
            hdf5_path = Path(cfg.canonical_hdf5[0])
            if not hdf5_path.exists():
                return DoctorCheck(
                    "hdf5_sample_read",
                    False,
                    f"HDF5 file not found: {hdf5_path}",
                )

            with h5py.File(hdf5_path, "r") as f:
                if "name" in f and "spectrum" in f:
                    names = f["name"]
                    if len(names) > 0:
                        return DoctorCheck(
                            "hdf5_sample_read",
                            True,
                            f"HDF5 file readable: {len(names)} spectra found",
                            {"file": str(hdf5_path), "num_spectra": len(names)},
                        )
                    else:
                        return DoctorCheck(
                            "hdf5_sample_read",
                            False,
                            "HDF5 file has no spectra",
                        )
                else:
                    return DoctorCheck(
                        "hdf5_sample_read",
                        False,
                        "HDF5 file structure unexpected",
                    )
        except Exception as e:
            return DoctorCheck(
                "hdf5_sample_read",
                False,
                f"HDF5 read test failed: {e}",
            )

    def check_rust_validation(self) -> DoctorCheck:
        """Test Rust validation."""
        if not RUST_VALIDATOR_AVAILABLE:
            return DoctorCheck(
                "rust_validation",
                False,
                "Rust validator not available (msms_rust module not found)",
            )

        try:
            mzs = np.array([100.0, 200.0, 300.0], dtype=np.float32)
            ints = np.array([1000.0, 2000.0, 3000.0], dtype=np.float32)
            precursor_mz = 250.0

            is_valid = msms_rust.validate_spectrum(mzs, ints, precursor_mz)
            if is_valid:
                return DoctorCheck(
                    "rust_validation",
                    True,
                    "Rust validator working",
                )
            else:
                return DoctorCheck(
                    "rust_validation",
                    False,
                    "Rust validator rejected valid test data",
                )
        except Exception as e:
            return DoctorCheck(
                "rust_validation",
                False,
                f"Rust validation test failed: {e}",
            )

    def check_arrow_write(self) -> DoctorCheck:
        """Test Arrow write/rewind."""
        try:
            test_data = [
                {
                    "sample_id": "test_1",
                    "mzs": np.array([100.0, 200.0], dtype=np.float32),
                    "ints": np.array([1000.0, 2000.0], dtype=np.float32),
                    "precursor_mz": np.float32(150.0),
                    "charge": np.int8(1),
                }
            ]

            table = pa.Table.from_pylist(test_data)
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
                pq.write_table(table, tmp.name, compression="zstd")
                
                read_table = pq.read_table(tmp.name)
                if len(read_table) == len(test_data):
                    return DoctorCheck(
                        "arrow_write",
                        True,
                        "Arrow write/rewind test passed",
                    )
                else:
                    return DoctorCheck(
                        "arrow_write",
                        False,
                        f"Arrow read row count mismatch: {len(read_table)} vs {len(test_data)}",
                    )
        except Exception as e:
            return DoctorCheck(
                "arrow_write",
                False,
                f"Arrow write/rewind test failed: {e}",
            )

    def check_disk_ram(self, cfg: Optional[PipelineConfig] = None) -> DoctorCheck:
        """Check disk and RAM availability."""
        details = {}
        issues = []

        if not PSUTIL_AVAILABLE:
            return DoctorCheck(
                "disk_ram",
                False,
                "psutil not available (disk/RAM checks disabled)",
            )

        try:
            output_dir = cfg.output_root if cfg else Path(".")
            stat = os.statvfs(output_dir)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            details["disk_free_gb"] = free_gb

            if free_gb < 10.0:
                issues.append(f"Insufficient disk space: {free_gb:.1f}GB free")

            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            details["ram_available_gb"] = available_gb

            if available_gb < 4.0:
                issues.append(f"Insufficient RAM: {available_gb:.1f}GB available")

        except Exception as e:
            return DoctorCheck(
                "disk_ram",
                False,
                f"Disk/RAM check failed: {e}",
                details,
            )

        if issues:
            return DoctorCheck(
                "disk_ram",
                False,
                f"Availability issues: {', '.join(issues)}",
                details,
            )
        else:
            return DoctorCheck(
                "disk_ram",
                True,
                "Disk and RAM availability OK",
                details,
            )

    def check_dependencies(self) -> DoctorCheck:
        """Check dependency versions."""
        details = {}
        issues = []

        try:
            import pyarrow
            details["pyarrow"] = pyarrow.__version__
        except ImportError:
            issues.append("pyarrow not installed")

        try:
            import h5py
            details["h5py"] = h5py.__version__
        except ImportError:
            issues.append("h5py not installed")

        if PSUTIL_AVAILABLE:
            details["psutil"] = psutil.__version__

        if RUST_VALIDATOR_AVAILABLE:
            details["rust_validator"] = "available"

        if issues:
            return DoctorCheck(
                "dependencies",
                False,
                f"Missing dependencies: {', '.join(issues)}",
                details,
            )
        else:
            return DoctorCheck(
                "dependencies",
                True,
                "All required dependencies available",
                details,
            )

    def run_all_checks(self) -> Tuple[bool, List[DoctorCheck]]:
        """Run all doctor checks.

        Returns:
            Tuple of (all_passed, list of checks)
        """
        self.checks = []

        self.checks.append(self.check_config_schema())
        self.checks.append(self.check_machine_compatibility())
        self.checks.append(self.check_dependencies())

        cfg = None
        if self.config_path and self.config_path.exists():
            try:
                cfg = load_config(self.config_path)
            except Exception:
                pass

        if cfg:
            self.checks.append(self.check_hdf5_sample_read(cfg))
            self.checks.append(self.check_disk_ram(cfg))

        self.checks.append(self.check_rust_validation())
        self.checks.append(self.check_arrow_write())

        all_passed = all(check.passed for check in self.checks)
        return all_passed, self.checks

    def print_report(self) -> None:
        """Print doctor check report."""
        print("\n" + "=" * 60)
        print("NexaData Doctor Mode - System Validation")
        print("=" * 60)

        for check in self.checks:
            status = "✓" if check.passed else "✗"
            print(f"{status} {check.name}: {check.message}")
            if check.details:
                for key, value in check.details.items():
                    print(f"    {key}: {value}")

        print("=" * 60)
        all_passed = all(check.passed for check in self.checks)
        if all_passed:
            print("✓ All checks passed!")
        else:
            print("✗ Some checks failed")
        print("=" * 60)

    def export_json(self, path: Path) -> None:
        """Export check results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "all_passed": all(check.passed for check in self.checks),
            "checks": [check.to_dict() for check in self.checks],
        }
        with open(path, "w") as f:
            json.dump(results, f, indent=2)


def run_doctor_mode(config_path: Optional[Path] = None) -> bool:
    """Run doctor mode validation.

    Args:
        config_path: Optional path to config file

    Returns:
        True if all checks passed, False otherwise
    """
    doctor = DoctorMode(config_path)
    all_passed, checks = doctor.run_all_checks()
    doctor.print_report()
    return all_passed

