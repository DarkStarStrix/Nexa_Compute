"""Parallel processing utilities for MS/MS pipeline."""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import PreprocessingConfig
from .metrics import PipelineMetrics
from .transforms import clean_and_canonicalize

try:
    from msms_rust import RustBatchProcessor as _RustBatchProcessor  # type: ignore

    RUST_BATCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _RustBatchProcessor = None
    RUST_BATCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process spectra in batches with parallel transforms."""

    def __init__(
        self,
        preprocessing: PreprocessingConfig,
        metrics: PipelineMetrics,
        batch_size: int = 1000,
        num_workers: int = None,
        use_rust_batch: bool = False,
        max_peaks: Optional[int] = None,
        max_input_peaks: Optional[int] = None,
    ):
        """Initialize batch processor.

        Args:
            preprocessing: Preprocessing configuration
            metrics: Metrics tracker
            batch_size: Number of spectra per batch
            num_workers: Number of worker threads (default: CPU count)
        """
        import os

        self.preprocessing = preprocessing
        self.metrics = metrics
        self.batch_size = batch_size
        self.num_workers = num_workers or min(os.cpu_count() or 4, 8)
        self._lock = Lock()
        self.max_peaks = max_peaks
        self.max_input_peaks = max_input_peaks

        self.use_rust_batch = bool(use_rust_batch and RUST_BATCH_AVAILABLE)
        self.rust_processor = None
        if self.use_rust_batch:
            try:
                rust_max_peaks = max_peaks or getattr(preprocessing, "max_peaks", 4096)
                input_guard = (
                    max_input_peaks
                    or getattr(preprocessing, "max_input_peaks", None)
                    or rust_max_peaks * 4
                )
                self.rust_processor = _RustBatchProcessor(
                    normalize=preprocessing.normalize_intensities,
                    sort_mz=preprocessing.sort_mz,
                    min_peaks=preprocessing.min_peaks,
                    max_peaks=rust_max_peaks,
                    max_precursor_mz=preprocessing.max_precursor_mz,
                    filter_nonfinite=preprocessing.filter_nonfinite,
                    max_input_peaks=input_guard,
                )
                logger.info(
                    "Rust batch processor enabled (max_peaks=%s, max_input_peaks=%s)",
                    rust_max_peaks,
                    input_guard,
                )
            except Exception as exc:  # pragma: no cover - optional path
                logger.warning(
                    "Failed to initialize Rust batch processor, falling back to Python: %s",
                    exc,
                )
                self.use_rust_batch = False

        if use_rust_batch and not self.use_rust_batch:
            logger.warning(
                "Rust batch processing requested but Rust extension is not available."
            )

    def process_batch(
        self, batch: List[Tuple[str, Dict]]
    ) -> List[Tuple[str, Dict]]:
        """Process a batch of spectra in parallel."""
        if self.use_rust_batch and self.rust_processor is not None:
            return self._process_batch_rust(batch)
        return self._process_batch_python(batch)

    def _process_batch_python(
        self, batch: List[Tuple[str, Dict]]
    ) -> List[Tuple[str, Dict]]:
        results = []

        def process_one(item: Tuple[str, Dict]) -> Optional[Tuple[str, Dict]]:
            sample_id, raw_record = item
            with self._lock:
                self.metrics.total_spectra += 1

            cleaned = clean_and_canonicalize(
                sample_id, raw_record, self.preprocessing, self.metrics
            )

            if cleaned:
                with self._lock:
                    self.metrics.record_sample_written()
                return (sample_id, cleaned)
            return None

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_one, item) for item in batch]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Error processing spectrum: {e}")
                finally:
                    del future

        del futures
        # GC handled by caller (cli.py) to avoid frequent stops
        return results

    def _process_batch_rust(
        self, batch: List[Tuple[str, Dict]]
    ) -> List[Tuple[str, Dict]]:
        if self.rust_processor is None:
            return self._process_batch_python(batch)

        py_batch: List[Dict] = []
        for sample_id, raw_record in batch:
            prepared = dict(raw_record)
            prepared["sample_id"] = sample_id
            if "ints" not in prepared and "intensities" in prepared:
                prepared["ints"] = prepared["intensities"]
            py_batch.append(prepared)

        records, summary = self.rust_processor.process(py_batch)  # type: ignore[arg-type]
        summary_dict = dict(summary)  # PyDict -> Python dict

        self._apply_rust_metrics(summary_dict, len(batch), len(records))

        cleaned_batch: List[Tuple[str, Dict]] = []
        for record in records:
            sample_id = record["sample_id"]
            cleaned_batch.append((sample_id, record))

        # GC handled by caller (cli.py)
        return cleaned_batch

    def _apply_rust_metrics(
        self, summary: Dict, processed: int, retained: int
    ) -> None:
        integrity = summary.get("integrity_errors", {}) or {}
        attrition = summary.get("attrition", {}) or {}

        for error_type, count in integrity.items():
            self.metrics.record_integrity_error_bulk(str(error_type), int(count))

        for reason, count in attrition.items():
            self.metrics.record_attrition_bulk(str(reason), int(count))

        self.metrics.record_samples_written_bulk(retained)

        dropped = summary.get("dropped_peaks", 0)
        if dropped:
            logger.debug(
                "Rust batch processor dropped %s low-priority peaks (%s processed -> %s retained)",
                dropped,
                processed,
                retained,
            )

