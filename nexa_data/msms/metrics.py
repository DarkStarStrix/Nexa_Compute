"""Real-time metrics collection and reporting with time series."""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from dask.distributed import get_client
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


@dataclass
class TimeSeriesPoint:
    """Single time series data point."""

    timestamp: float
    samples_per_second: float
    bytes_per_second: float
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_used_gb: Optional[float] = None
    dask_worker_saturation: Optional[float] = None
    shards_written: int = 0
    integrity_error_rate: float = 0.0


class PipelineMetrics:
    """Track pipeline metrics in real-time with time series."""

    def __init__(self, collect_timeseries: bool = True, timeseries_interval: float = 5.0):
        """Initialize metrics tracker.

        Args:
            collect_timeseries: Whether to collect time series data
            timeseries_interval: Interval between time series samples (seconds)
        """
        self.total_spectra = 0
        self.integrity_errors = defaultdict(int)
        self.attrition = defaultdict(int)
        self.samples_written = 0
        self.shards_written = 0
        self.bytes_written = 0
        self.start_time = time.time()
        
        self.collect_timeseries = collect_timeseries
        self.timeseries_interval = timeseries_interval
        self.timeseries: List[TimeSeriesPoint] = []
        self.last_timeseries_sample = time.time()
        
        self.shard_write_timings: List[float] = []
        self.last_shard_write_time = self.start_time

    def record_integrity_error(self, error_type: str) -> None:
        """Record an integrity error."""
        self.integrity_errors[error_type] += 1
        self.total_spectra += 1

    def record_attrition(self, reason: str) -> None:
        """Record an attrition (soft quality drop)."""
        self.attrition[reason] += 1
        self.total_spectra += 1

    def record_sample_written(self) -> None:
        """Record a successfully written sample."""
        self.samples_written += 1
        self.total_spectra += 1

    def record_samples_written_bulk(self, count: int) -> None:
        """Record multiple successfully written samples."""
        if count <= 0:
            return
        self.samples_written += count
        self.total_spectra += count

    def record_integrity_error_bulk(self, error_type: str, count: int) -> None:
        """Record multiple integrity errors of the same type."""
        if count <= 0:
            return
        self.integrity_errors[error_type] += count
        self.total_spectra += count

    def record_attrition_bulk(self, reason: str, count: int) -> None:
        """Record multiple attrition events of the same reason."""
        if count <= 0:
            return
        self.attrition[reason] += count
        self.total_spectra += count

    def record_shard_written(self, bytes_written: int) -> None:
        """Record a shard write."""
        current_time = time.time()
        write_duration = current_time - self.last_shard_write_time
        self.shard_write_timings.append(write_duration)
        self.last_shard_write_time = current_time
        
        self.shards_written += 1
        self.bytes_written += bytes_written
        
        # Collect time series sample if interval has passed
        if self.collect_timeseries:
            self._maybe_sample_timeseries()

    def get_metrics_dict(self) -> Dict:
        """Get metrics as dictionary."""
        elapsed = time.time() - self.start_time
        integrity_error_count = sum(self.integrity_errors.values())
        attrition_count = sum(self.attrition.values())

        return {
            "total_spectra": self.total_spectra,
            "integrity_errors": dict(self.integrity_errors),
            "integrity_error_count": integrity_error_count,
            "integrity_error_rate": integrity_error_count / self.total_spectra if self.total_spectra > 0 else 0.0,
            "attrition": dict(self.attrition),
            "attrition_count": attrition_count,
            "attrition_rate": attrition_count / self.total_spectra if self.total_spectra > 0 else 0.0,
            "samples_written": self.samples_written,
            "shards_written": self.shards_written,
            "bytes_written": self.bytes_written,
            "elapsed_seconds": elapsed,
            "samples_per_second": self.total_spectra / elapsed if elapsed > 0 else 0.0,
        }

    def print_summary(self) -> None:
        """Print formatted summary."""
        metrics = self.get_metrics_dict()
        print("\n" + "=" * 60)
        print("Pipeline Metrics Summary")
        print("=" * 60)
        print(f"Total spectra processed: {metrics['total_spectra']}")
        print(f"Integrity errors: {metrics['integrity_error_count']} ({metrics['integrity_error_rate']:.4%})")
        print(f"Attrition: {metrics['attrition_count']} ({metrics['attrition_rate']:.4%})")
        print(f"Samples written: {metrics['samples_written']}")
        print(f"Shards written: {metrics['shards_written']}")
        print(f"Bytes written: {metrics['bytes_written']:,}")
        print(f"Processing speed: {metrics['samples_per_second']:.1f} samples/sec")
        print(f"Elapsed time: {metrics['elapsed_seconds']:.1f} seconds")
        print("=" * 60)

    def _maybe_sample_timeseries(self) -> None:
        """Sample time series data if interval has passed."""
        current_time = time.time()
        if current_time - self.last_timeseries_sample < self.timeseries_interval:
            return
        
        elapsed = current_time - self.start_time
        samples_per_sec = self.total_spectra / elapsed if elapsed > 0 else 0.0
        bytes_per_sec = self.bytes_written / elapsed if elapsed > 0 else 0.0
        
        integrity_error_rate = (
            sum(self.integrity_errors.values()) / self.total_spectra
            if self.total_spectra > 0
            else 0.0
        )
        
        cpu_percent = None
        memory_percent = None
        memory_used_gb = None
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                memory_percent = mem.percent
                memory_used_gb = mem.used / (1024**3)
            except Exception:
                pass
        
        dask_saturation = None
        if DASK_AVAILABLE:
            try:
                client = get_client()
                workers = client.scheduler_info().get("workers", {})
                if workers:
                    total_workers = len(workers)
                    active_workers = sum(1 for w in workers.values() if w.get("nthreads", 0) > 0)
                    dask_saturation = active_workers / total_workers if total_workers > 0 else 0.0
            except Exception:
                pass
        
        point = TimeSeriesPoint(
            timestamp=current_time,
            samples_per_second=samples_per_sec,
            bytes_per_second=bytes_per_sec,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            dask_worker_saturation=dask_saturation,
            shards_written=self.shards_written,
            integrity_error_rate=integrity_error_rate,
        )
        
        self.timeseries.append(point)
        self.last_timeseries_sample = current_time
    
    def get_timeseries_dict(self) -> List[Dict]:
        """Get time series data as list of dictionaries."""
        return [
            {
                "timestamp": p.timestamp,
                "samples_per_second": p.samples_per_second,
                "bytes_per_second": p.bytes_per_second,
                "cpu_percent": p.cpu_percent,
                "memory_percent": p.memory_percent,
                "memory_used_gb": p.memory_used_gb,
                "dask_worker_saturation": p.dask_worker_saturation,
                "shards_written": p.shards_written,
                "integrity_error_rate": p.integrity_error_rate,
            }
            for p in self.timeseries
        ]
    
    def export_json(self, path: Path) -> None:
        """Export metrics to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_metrics_dict(), f, indent=2)
    
    def export_timeseries_json(self, path: Path) -> None:
        """Export time series data to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_timeseries_dict(), f, indent=2)

