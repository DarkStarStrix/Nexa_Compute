"""HDF5 spectrum source with deterministic iteration and batch reading."""

import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional


class HDF5SpectrumSource:
    """Deterministic iterator over one or more canonical HDF5 datasets with batch reading."""

    def __init__(
        self,
        hdf5_paths: List[str],
        max_spectra: Optional[int] = None,
        batch_size: int = 1000,
        num_workers: int = 4,
    ):
        """Initialize spectrum source.

        Args:
            hdf5_paths: List of HDF5 file paths (will be sorted for determinism)
            max_spectra: Optional limit on number of spectra to process
            batch_size: Number of spectra to read in each batch
            num_workers: Number of worker threads for parallel reading
        """
        self.paths = sorted(hdf5_paths)
        self.max_spectra = max_spectra
        self.batch_size = batch_size
        self.num_workers = num_workers

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

    def iter_spectra(self) -> Iterator[tuple[str, Dict]]:
        """Iterate over spectra in deterministic order with batch processing.

        Supports two HDF5 formats:
        1. Nested group format: f["spectra"][sample_id]["mzs"], etc.
        2. Table format: f["name"], f["spectrum"], f["precursor_mz"], etc.

        Yields:
            Tuple of (sample_id, record_dict)
        """
        for path in self.paths:
            try:
                with h5py.File(path, "r") as f:
                    # Check if it's table format (has 'name' and 'spectrum' at top level)
                    if "name" in f and "spectrum" in f:
                        yield from self._iter_table_format_batched(f)
                    elif "spectra" in f:
                        yield from self._iter_nested_format(f)
                    else:
                        raise ValueError(f"Unsupported HDF5 format in {path}")

            except (OSError, ValueError) as e:
                continue

    def _iter_table_format_batched(self, f: h5py.File) -> Iterator[tuple[str, Dict]]:
        """Iterate over table-format HDF5 file with batch reading."""
        names = f["name"]
        spectra = f["spectrum"]
        precursor_mz = f["precursor_mz"]
        charge = f["charge"]

        total = len(names)
        chunk_size = max(self.batch_size * 10, 5000)
        emitted = 0
        chunk_start = 0

        def make_sort_key(idx: int) -> tuple[str, int]:
            raw_name = names[idx]
            base = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            return (base, idx)

        chunk_start = 0
        sorted_indices_buffer: List[int] = []

        def emit_buffer(buffer: List[int]):
            buffer.sort(key=make_sort_key)
            return buffer.copy()

        while chunk_start < total:
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_indices = list(range(chunk_start, chunk_end))
            sorted_chunk = sorted(chunk_indices, key=make_sort_key)
            sorted_indices_buffer.extend(sorted_chunk)
            chunk_start = chunk_end

            if len(sorted_indices_buffer) < self.batch_size * 4 and chunk_end < total:
                continue

            sorted_indices = sorted_indices_buffer
            sorted_indices_buffer = []

            for local_start in range(0, len(sorted_indices), self.batch_size):
                batch_indices = sorted_indices[
                    local_start : local_start + self.batch_size
                ]

            for local_start in range(0, len(chunk_indices), self.batch_size):
                batch_indices = chunk_indices[
                    local_start : local_start + self.batch_size
                ]
                if self.max_spectra and emitted >= self.max_spectra:
                    return

                batch_indices_array = np.array(batch_indices, dtype=np.int64)
                read_order = np.argsort(batch_indices_array, kind="mergesort")
                read_indices = batch_indices_array[read_order]

                batch_names_raw = names[read_indices]
                batch_spectra_raw = spectra[read_indices]
                batch_precursor_mz_raw = precursor_mz[read_indices]
                batch_charge_raw = charge[read_indices]

                index_to_position = {
                    int(idx): pos for pos, idx in enumerate(read_indices.tolist())
                }

                for idx in batch_indices:
                    if self.max_spectra and emitted >= self.max_spectra:
                        return
                    read_pos = index_to_position[int(idx)]
                    try:
                        name = batch_names_raw[read_pos]
                        base_id = name.decode("utf-8") if isinstance(name, bytes) else str(name)
                        sample_id = f"{base_id}_{idx}"

                        spec_data = batch_spectra_raw[read_pos]
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
                            "precursor_mz": float(batch_precursor_mz_raw[read_pos]),
                            "charge": int(batch_charge_raw[read_pos]),
                            "collision_energy": 0.0,
                            "adduct": None,
                            "instrument_type": None,
                            "smiles": None,
                            "inchikey": None,
                            "formula": None,
                        }

                        emitted += 1
                        yield sample_id, record

                    except (KeyError, ValueError, TypeError, IndexError):
                        continue

            chunk_start += chunk_size

    def _iter_nested_format(self, f: h5py.File) -> Iterator[tuple[str, Dict]]:
        """Iterate over nested group format HDF5 file."""
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
