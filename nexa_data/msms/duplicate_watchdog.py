"""Global duplicate ID watchdog with Bloom filter support."""

import logging
import mmap
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

try:
    from pybloom_live import BloomFilter
    BLOOM_FILTER_AVAILABLE = True
except ImportError:
    BLOOM_FILTER_AVAILABLE = False
    BloomFilter = None


class DuplicateWatchdog:
    """Global duplicate ID watchdog for cross-shard duplicate detection."""

    def __init__(
        self,
        use_bloom_filter: bool = False,
        capacity: int = 10_000_000,
        error_rate: float = 0.001,
        memory_mapped_path: Optional[Path] = None,
    ):
        """Initialize duplicate watchdog.

        Args:
            use_bloom_filter: Use Bloom filter instead of hash set (memory-efficient)
            capacity: Expected number of unique sample IDs
            error_rate: False positive rate for Bloom filter
            memory_mapped_path: Optional path for memory-mapped storage
        """
        self.use_bloom_filter = use_bloom_filter and BLOOM_FILTER_AVAILABLE
        self.capacity = capacity
        self.error_rate = error_rate
        self.memory_mapped_path = memory_mapped_path

        if self.use_bloom_filter:
            if not BLOOM_FILTER_AVAILABLE:
                logger.warning("Bloom filter not available, falling back to hash set")
                self.use_bloom_filter = False

        if self.use_bloom_filter:
            self.bloom_filter: Optional[BloomFilter] = BloomFilter(
                capacity=capacity,
                error_rate=error_rate,
            )
            self.seen_set: Optional[Set[str]] = None
        else:
            self.seen_set: Set[str] = set()
            self.bloom_filter: Optional[BloomFilter] = None

        self.duplicate_count = 0
        self.total_checked = 0

    def check(self, sample_id: str) -> bool:
        """Check if sample_id has been seen before.

        Args:
            sample_id: Sample identifier to check

        Returns:
            True if duplicate, False if new
        """
        self.total_checked += 1

        if self.use_bloom_filter and self.bloom_filter:
            if sample_id in self.bloom_filter:
                self.duplicate_count += 1
                return True
            self.bloom_filter.add(sample_id)
            return False
        else:
            if sample_id in self.seen_set:
                self.duplicate_count += 1
                return True
            self.seen_set.add(sample_id)
            return False

    def add(self, sample_id: str) -> None:
        """Add sample_id to tracking (without checking).

        Args:
            sample_id: Sample identifier to add
        """
        if self.use_bloom_filter and self.bloom_filter:
            self.bloom_filter.add(sample_id)
        else:
            self.seen_set.add(sample_id)

    def get_stats(self) -> dict:
        """Get watchdog statistics."""
        return {
            "duplicate_count": self.duplicate_count,
            "total_checked": self.total_checked,
            "unique_count": len(self.seen_set) if self.seen_set else (
                self.capacity if self.bloom_filter else 0
            ),
            "using_bloom_filter": self.use_bloom_filter,
        }

    def save(self, path: Path) -> None:
        """Save watchdog state to disk.

        Args:
            path: Path to save state
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_bloom_filter and self.bloom_filter:
            with open(path, "wb") as f:
                self.bloom_filter.tofile(f)
            logger.info(f"Saved Bloom filter to {path}")
        else:
            import json
            with open(path, "w") as f:
                json.dump(list(self.seen_set), f)
            logger.info(f"Saved {len(self.seen_set)} sample IDs to {path}")

    def load(self, path: Path) -> None:
        """Load watchdog state from disk.

        Args:
            path: Path to load state from
        """
        if not path.exists():
            logger.warning(f"Watchdog state file not found: {path}")
            return

        if self.use_bloom_filter and self.bloom_filter and BLOOM_FILTER_AVAILABLE:
            with open(path, "rb") as f:
                self.bloom_filter = BloomFilter.fromfile(f)
            logger.info(f"Loaded Bloom filter from {path}")
        else:
            import json
            with open(path) as f:
                ids = json.load(f)
            self.seen_set = set(ids)
            logger.info(f"Loaded {len(self.seen_set)} sample IDs from {path}")

