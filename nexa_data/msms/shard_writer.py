"""Shard writer with checksums and metrics."""

import errno
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import BaseException, Dict, List, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nexa_compute.core.retry import RetryError, RetryPolicy, retry_call

from .metrics import PipelineMetrics


def sha256_of_file(path: Path) -> str:
    """Calculate SHA256 checksum of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_PARQUET_WRITE_POLICY = RetryPolicy(
    max_attempts=4,
    base_delay=0.5,
    max_delay=4.0,
    jitter=0.3,
    retry_exceptions=(pa.ArrowException, OSError, IOError),
)

_RENAME_POLICY = RetryPolicy(
    max_attempts=5,
    base_delay=0.25,
    max_delay=2.0,
    jitter=0.3,
    retry_exceptions=(OSError, IOError),
)


class ShardWriter:
    """Write shards with checksums and duplicate detection.
    
    Uses streaming writes (ParquetWriter) to minimize memory usage.
    """

    def __init__(
        self,
        output_dir: Path,
        max_size: int,
        dataset_name: str,
        schema_version: int,
        split: str = "train",
        metrics: Optional[PipelineMetrics] = None,
        run_id: Optional[str] = None,
    ):
        """Initialize shard writer.

        Args:
            output_dir: Output directory for shards
            max_size: Maximum shard size in bytes
            dataset_name: Dataset name
            schema_version: Schema version
            split: Split name (train/val/test)
            metrics: Optional metrics tracker
            run_id: Optional run identifier for unique shard naming
        """
        self.output_dir = output_dir / split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quarantine directory for failed shards
        self.quarantine_dir = output_dir / "quarantine" / split
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_name = dataset_name
        self.schema_version = schema_version
        self.max_size = max_size
        self.split = split
        self.metrics = metrics
        
        # Generate run_id if not provided (timestamp-based for uniqueness)
        if run_id is None:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id

        self.current_bytes = 0
        self.shard_index = 0
        self.seen_sample_ids: Set[str] = set()
        
        # Streaming state
        self.writer: Optional[pq.ParquetWriter] = None
        self.current_path: Optional[Path] = None
        self.current_temp_path: Optional[Path] = None
        self.current_sample_ids_in_shard: List[str] = []
        self.current_shard_rows_count = 0

        # Define schema explicitly if possible, otherwise inferred from first batch
        self.schema = None

    def _init_writer(self, schema: pa.Schema) -> None:
        """Initialize a new Parquet writer."""
        self.current_shard_rows_count = 0
        self.current_sample_ids_in_shard = []
        
        filename = f"shard_{self.run_id}_{self.shard_index:05d}.parquet"
        self.current_path = self.output_dir / filename
        self.current_temp_path = self.current_path.with_suffix(".parquet.tmp")
        
        self.writer = pq.ParquetWriter(
            self.current_temp_path,
            schema,
            compression="zstd"
        )
        self.schema = schema

    def _close_current_shard(self) -> None:
        """Close the current shard file and finalize."""
        if self.writer:
            self.writer.close()
            self.writer = None
            
            # Atomic rename
            if self.current_temp_path and self.current_temp_path.exists():
                self._rename_with_retry(self.current_temp_path, self.current_path)
                
                # Compute checksum and manifest
                checksum = sha256_of_file(self.current_path)
                file_size = os.path.getsize(self.current_path)
                
                manifest = {
                    "dataset": self.dataset_name,
                    "split": self.split,
                    "run_id": self.run_id,
                    "shard_index": self.shard_index,
                    "num_samples": self.current_shard_rows_count,
                    "sample_ids": self.current_sample_ids_in_shard,
                    "schema_version": self.schema_version,
                    "checksum": f"sha256:{checksum}",
                    "file_size_bytes": file_size,
                    "created_at": datetime.utcnow().isoformat(),
                }
                
                manifest_path = self.current_path.with_suffix(".manifest.json")
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
                
                if self.metrics:
                    self.metrics.record_shard_written(file_size)
                    
                try:
                    from tqdm import tqdm
                    tqdm.write(f"[OK] Wrote shard {self.current_path.name} ({self.current_shard_rows_count} samples, {file_size:,} bytes)")
                except ImportError:
                    print(f"[OK] Wrote shard {self.current_path.name} ({self.current_shard_rows_count} samples, {file_size:,} bytes)")

            self.shard_index += 1
            self.current_bytes = 0

    def _validate_record_batch(self, records: List[Dict]) -> None:
        """Validate a batch of records before writing."""
        for record in records:
            sample_id = record["sample_id"]
            
            # Check required fields
            required_fields = ["mzs", "ints", "precursor_mz", "charge"]
            for field in required_fields:
                if field not in record:
                    raise ValueError(f"Missing required field '{field}' in sample {sample_id}")

            mzs = record["mzs"] if isinstance(record["mzs"], np.ndarray) else np.array(record["mzs"])
            ints = record["ints"] if isinstance(record["ints"], np.ndarray) else np.array(record["ints"])

            # Check shape consistency
            if len(mzs) != len(ints):
                raise ValueError(f"Shape mismatch in sample {sample_id}: mzs={len(mzs)}, ints={len(ints)}")

            # Check for non-finite values
            if not np.isfinite(mzs).all() or not np.isfinite(ints).all():
                raise ValueError(f"Non-finite values in sample {sample_id}")

            # Check for negative m/z
            if (mzs <= 0).any():
                raise ValueError(f"Negative m/z values in sample {sample_id}")

            # Check precursor_mz
            if not np.isfinite(record["precursor_mz"]) or record["precursor_mz"] <= 0:
                raise ValueError(f"Invalid precursor_mz in sample {sample_id}")

    def add(self, record: Dict) -> None:
        """Add a record to current shard."""
        self.add_batch([record])

    def add_batch(self, records: List[Dict]) -> None:
        """Add multiple records efficiently."""
        if not records:
            return

        # deduplication check
        unique_records = []
        for record in records:
            sample_id = record["sample_id"]
            if sample_id in self.seen_sample_ids:
                continue
            
            self.seen_sample_ids.add(sample_id)
            unique_records.append(record)
            self.current_sample_ids_in_shard.append(sample_id)
        
        if not unique_records:
            return

        # Validate data
        self._validate_record_batch(unique_records)

        # Convert to Arrow Table
        try:
            table = pa.Table.from_pylist(unique_records)
        except Exception as e:
            raise ValueError(f"Failed to convert records to Arrow table: {e}") from e

        # Initialize writer if needed
        if self.writer is None:
            self._init_writer(table.schema)
        
        self._write_table_with_retry(table)
        
        # Update stats
        self.current_shard_rows_count += len(unique_records)
        self.current_bytes += table.nbytes
        
        # Rotate if full
        if self.current_bytes >= self.max_size:
            self._close_current_shard()

    def _quarantine_shard(self, shard_name: str, error_msg: str) -> None:
        """Move failed shard to quarantine directory.
        
        Args:
            shard_name: Name of the failed shard
            error_msg: Error message describing the failure
        """
        quarantine_path = self.quarantine_dir / f"{shard_name}.failed"
        try:
            from tqdm import tqdm
            tqdm.write(f"[QUARANTINE] Moving failed shard to {quarantine_path}")
        except ImportError:
            print(f"[QUARANTINE] Moving failed shard to {quarantine_path}")
        
        # Write error information
        error_info = {
            "shard_name": shard_name,
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        error_path = quarantine_path.with_suffix(".error.json")
        with open(error_path, "w") as f:
            json.dump(error_info, f, indent=2)
        
        if self.current_temp_path and self.current_temp_path.exists():
             # Close writer first
            if self.writer:
                self.writer.close()
                self.writer = None
            shutil.move(str(self.current_temp_path), str(quarantine_path))


    def close(self) -> None:
        """Close writer and finalize current shard."""
        if self.writer:
            self._close_current_shard()

    def _write_table_with_retry(self, table: pa.Table) -> None:
        """Write a table to disk with retry/backoff."""

        def _write() -> None:
            if self.writer is None:
                raise RuntimeError("Parquet writer has not been initialized.")
            self.writer.write_table(table)

        try:
            retry_call(
                _write,
                policy=_PARQUET_WRITE_POLICY,
                on_retry=lambda attempt, exc, delay: self._log_retry(
                    "write_table", attempt, delay, exc
                ),
            )
        except RetryError as exc:
            self._handle_write_failure(exc)

    def _rename_with_retry(self, tmp_path: Path, final_path: Path) -> None:
        """Rename tmp shard into final location with retry support."""

        def _rename() -> None:
            tmp_path.replace(final_path)

        try:
            retry_call(
                _rename,
                policy=_RENAME_POLICY,
                on_retry=lambda attempt, exc, delay: self._log_retry(
                    "rename_shard", attempt, delay, exc
                ),
            )
        except RetryError as exc:
            self._handle_write_failure(exc)

    def _log_retry(self, operation: str, attempt: int, delay: float, exc: BaseException) -> None:
        """Emit a retry warning using tqdm when available."""
        message = (
            f"[RETRY] {operation} attempt {attempt} failed ({exc!r}). "
            f"Retrying in {delay:.2f}s."
        )
        try:
            from tqdm import tqdm

            tqdm.write(message)
        except ImportError:
            print(message)

    def _handle_write_failure(self, exc: RetryError) -> None:
        """Quarantine shard and surface meaningful error message."""
        shard_name = self.current_path.name if self.current_path else f"shard_{self.shard_index:05d}"
        if isinstance(exc.last_exception, OSError) and exc.last_exception.errno == errno.ENOSPC:
            error_msg = (
                f"Disk full while writing shard {shard_name}: {exc.last_exception}"
            )
        else:
            error_msg = str(exc.last_exception)
        self._quarantine_shard(shard_name, error_msg)
        raise RuntimeError(f"Failed to finalize shard {shard_name}") from exc.last_exception
