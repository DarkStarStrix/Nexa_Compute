"""Deduplication helpers (MinHash stubs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "DeduplicationReport",
    "minhash_deduplicate",
]


@dataclass(frozen=True)
class DeduplicationReport:
    """Summary information for a deduplication pass."""

    total_records: int
    duplicates_removed: int
    duplicates: Sequence[int]


def minhash_deduplicate(
    records: Iterable[Mapping[str, object]],
    *,
    key_fn: Callable[[Mapping[str, object]], str],
    threshold: float = 0.9,
) -> DeduplicationReport:
    """Placeholder implementation that flags no duplicates.

    The MinHash implementation will be provided in a future revision. For now
    the stub returns a report showing zero duplicates removed so that pipeline
    steps can depend on a stable interface.
    """

    total = sum(1 for _ in records)
    return DeduplicationReport(total_records=total, duplicates_removed=0, duplicates=())
