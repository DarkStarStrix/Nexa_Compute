# ADR-002: Data Versioning with Content-Addressable Storage

## Status
Accepted

## Context
Reproducibility in ML requires exact versioning of datasets. Storing full copies of large datasets for every version is storage-inefficient.

## Decision
We will implement a custom **Content-Addressable Storage (CAS)** layer, inspired by Git and DVC.

### Rationale
1. **Deduplication**: Files with identical content are stored only once, regardless of filename or dataset version.
2. **Immutability**: Blobs are immutable; versions are just metadata pointing to blobs.
3. **Simplicity**: Avoids the complexity of running a full DVC server or Git-LFS for the core platform logic.

## Implementation
- **Blob Store**: Flat directory structure (sharded by hash prefix) storing files by SHA-256 hash.
- **Meta Store**: JSON files describing dataset versions (file path -> blob hash mapping).
- **API**: `DataVersionControl` class handles commit/checkout operations.

## Consequences
- **Positive**: Efficient storage, instant deduplication, simplified versioning logic.
- **Negative**: Requires custom tooling for GC (garbage collection) of unreferenced blobs.

