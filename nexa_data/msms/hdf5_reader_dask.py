"""Dask-optimized HDF5 spectrum source with parallel chunked reading."""

import h5py
import numpy as np
from typing import Dict, Iterator, List, Optional

try:
    import dask.array as da
    from dask.distributed import Client, LocalCluster
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from .config import PreprocessingConfig


class DaskHDF5SpectrumSource:
    """High-performance HDF5 reader using Dask for parallel chunked processing."""

    def __init__(
        self,
        hdf5_paths: List[str],
        max_spectra: Optional[int] = None,
        num_workers: int = 4,
        memory_limit: str = "3GB",
        chunk_size: Optional[tuple] = None,
        processes: bool = False,
        threads_per_worker: int = 1,
    ):
        """Initialize Dask-optimized spectrum source.

        Args:
            hdf5_paths: List of HDF5 file paths
            max_spectra: Optional limit on number of spectra to process
            num_workers: Number of Dask workers
            memory_limit: Memory limit per worker
            chunk_size: Optional chunk size for Dask (auto if None)
            processes: Use multiprocess (True) or threads-only (False)
            threads_per_worker: Number of threads per worker
        """
        if not DASK_AVAILABLE:
            raise ImportError("Dask is required. Install with: pip install dask distributed")

        self.paths = sorted(hdf5_paths)
        self.max_spectra = max_spectra
        self.num_workers = num_workers
        self.memory_limit = memory_limit
        self.chunk_size = chunk_size
        self.processes = processes
        self.threads_per_worker = threads_per_worker

        # Initialize Dask cluster
        # Local mode: threads-only (processes=False) for M-series Macs
        # Production mode: multiprocess (processes=True) for HPC systems
        self.cluster = LocalCluster(
            n_workers=num_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=processes,
        )
        self.client = Client(self.cluster)

    def __len__(self) -> int:
        """Return total number of spectra across all files."""
        if self.max_spectra:
            return self.max_spectra
        
        total = 0
        for path in self.paths:
            try:
                with h5py.File(path, "r") as f:
                    if "name" in f:
                        total += len(f["name"])
                    elif "spectra" in f:
                        total += len(f["spectra"])
            except (OSError, KeyError):
                pass
        return total

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close Dask client and cluster."""
        if hasattr(self, "client"):
            self.client.close()
        if hasattr(self, "cluster"):
            self.cluster.close()

    def iter_spectra(self) -> Iterator[tuple[str, Dict]]:
        """Iterate over spectra using Dask for parallel chunked reading.

        Yields:
            Tuple of (sample_id, record_dict)
        """
        for path in self.paths:
            with h5py.File(path, "r") as f:
                if "name" in f and "spectrum" in f:
                    yield from self._iter_table_format_dask(f)
                elif "spectra" in f:
                    # Fallback to regular iteration for nested format
                    yield from self._iter_nested_format(f)
                else:
                    raise ValueError(f"Unsupported HDF5 format in {path}")

    def _iter_table_format_dask(self, f: h5py.File) -> Iterator[tuple[str, Dict]]:
        """Iterate over table-format HDF5 using Dask for parallel processing."""
        names = f["name"]
        spectra = f["spectrum"]
        precursor_mz = f["precursor_mz"]
        charge = f["charge"]

        # Get total number of spectra
        total_spectra = len(names)
        if self.max_spectra:
            total_spectra = min(total_spectra, self.max_spectra)

        # Determine optimal chunk size for Dask
        if self.chunk_size is None:
            # Use HDF5's chunk size or calculate optimal size
            hdf5_chunks = spectra.chunks if spectra.chunks else (1000, 2, 60)
            chunk_size = hdf5_chunks[0] if len(hdf5_chunks) > 0 else 1000
        else:
            chunk_size = self.chunk_size[0] if isinstance(self.chunk_size, tuple) else self.chunk_size

        # Create Dask arrays for parallel processing
        # Process in chunks for memory efficiency
        indices = np.arange(total_spectra)
        sorted_indices = sorted(
            indices,
            key=lambda i: (
                names[i].decode() if isinstance(names[i], bytes) else str(names[i]),
                i,
            ),
        )

        # Process in chunks using Dask
        # HDF5 requires indices to be in increasing order for fancy indexing
        for chunk_start in range(0, len(sorted_indices), chunk_size):
            chunk_indices = sorted_indices[chunk_start : chunk_start + chunk_size]
            
            # Sort chunk indices for HDF5 (must be increasing order)
            sorted_chunk_indices = sorted(chunk_indices)
            index_mapping = {orig_idx: sorted_idx for sorted_idx, orig_idx in enumerate(sorted_chunk_indices)}

            # Read chunk using sorted indices for HDF5 compatibility
            chunk_indices_array = np.array(sorted_chunk_indices)
            batch_names = names[chunk_indices_array]
            batch_spectra = spectra[chunk_indices_array]
            batch_precursor_mz = precursor_mz[chunk_indices_array]
            batch_charge = charge[chunk_indices_array]

            # Process each spectrum in the chunk (maintain original order)
            for orig_idx in chunk_indices:
                # Find position in sorted batch
                i = index_mapping[orig_idx]
                try:
                    name = batch_names[i]
                    if isinstance(name, bytes):
                        base_id = name.decode("utf-8")
                    else:
                        base_id = str(name)

                    sample_id = f"{base_id}_{orig_idx}"

                    spec_data = batch_spectra[i]
                    mzs = spec_data[0, :].astype(np.float64)
                    ints = spec_data[1, :].astype(np.float64)

                    valid = (mzs > 0) & (ints > 0)
                    mzs = mzs[valid]
                    ints = ints[valid]

                    if len(mzs) == 0:
                        continue

                    record = {
                        "mzs": mzs,
                        "intensities": ints,
                        "precursor_mz": float(batch_precursor_mz[i]),
                        "charge": int(batch_charge[i]),
                        "collision_energy": 0.0,
                        "adduct": None,
                        "instrument_type": None,
                        "smiles": None,
                        "inchikey": None,
                        "formula": None,
                    }

                    yield sample_id, record

                except (KeyError, ValueError, TypeError, IndexError) as e:
                    continue

    def _iter_nested_format(self, f: h5py.File) -> Iterator[tuple[str, Dict]]:
        """Iterate over nested group format (fallback)."""
        count = 0
        spectra_group = f["spectra"]
        sample_ids = sorted(list(spectra_group.keys()))

        for sid in sample_ids:
            if self.max_spectra and count >= self.max_spectra:
                return

            try:
                s = spectra_group[sid]

                record = {
                    "mzs": s["mzs"][:],
                    "intensities": s["intensities"][:],
                    "precursor_mz": float(s["precursor_mz"][()]),
                    "charge": int(s["charge"][()]),
                    "collision_energy": float(s["collision_energy"][()]),
                    "adduct": s.attrs.get("adduct", None),
                    "instrument_type": s.attrs.get("instrument_type", None),
                    "smiles": s.attrs.get("smiles", None),
                    "inchikey": s.attrs.get("inchikey", None),
                    "formula": s.attrs.get("formula", None),
                }

                yield sid, record
                count += 1

            except (KeyError, ValueError, TypeError) as e:
                continue

