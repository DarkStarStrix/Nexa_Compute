# MS/MS Rust Extension

Fast HDF5 reading, validation, and batch sanitization routines implemented in Rust for maximum performance and memory safety.

## Building

First, install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then install maturin (Python-Rust bridge):
```bash
pip install maturin
```

Build the extension:
```bash
cd nexa_data/msms/rust_ext
maturin develop --release
```

Or install in development mode:
```bash
maturin develop
```

## Usage

The Rust extension is automatically used if available. `transforms.py` and `processor.py` will detect it and enable:

- **RustBatchProcessor**: deterministic batch cleaner that enforces memory limits, clamps/normalizes peaks, and trims spectra using safe Rust allocators.
- **Validator**: SIMD-fast validation for spectra before writing.

Enable the batch processor via config (`use_rust_batch: true`) or CLI (`--use-rust-batch`).

## Features

- Fast HDF5 spectrum reading with zero-copy where possible
- SIMD-optimized validation routines
- Parallel batch processing (Rayon) with deterministic ordering
- Memory fragmentation resistant transforms (peak capping + normalization)

