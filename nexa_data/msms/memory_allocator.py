"""Dynamic memory allocation for Dask workers to optimize throughput."""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def calculate_optimal_memory_allocation(
    num_workers: int,
    system_reserve_pct: float = 25.0,
    target_utilization_pct: float = 75.0,
    min_memory_gb: float = 1.5,  # Raised from 1.0 for better stability
    max_memory_gb: float = 8.0,
) -> str:
    """Calculate optimal memory limit per worker based on available system memory.
    
    This function dynamically calculates memory allocation to:
    - Maximize pipeline throughput
    - Avoid worker pausing (stays below 80% threshold)
    - Reserve memory for system processes
    
    Args:
        num_workers: Number of Dask workers
        system_reserve_pct: Percentage of total RAM to reserve for system (default: 25%)
        target_utilization_pct: Target memory utilization per worker (default: 75%, below 80% pause threshold)
        min_memory_gb: Minimum memory per worker in GB (default: 1.5GB)
        max_memory_gb: Maximum memory per worker in GB (default: 8GB)
    
    Returns:
        Memory limit string in format "XGB" suitable for Dask LocalCluster
    """
    if not PSUTIL_AVAILABLE:
        logger.warning(
            "psutil not available, using conservative default: 3GB per worker. "
            "Install psutil for dynamic memory allocation."
        )
        return "3GB"
    
    try:
        # Get system memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024 ** 3)  # Convert bytes to GB
        available_memory_gb = memory.available / (1024 ** 3)
        used_memory_gb = memory.used / (1024 ** 3)
        
        # Calculate reserved memory for system processes
        system_reserve_gb = total_memory_gb * (system_reserve_pct / 100.0)
        
        # Calculate worker memory pool
        # Strategy: Use total memory minus system reserve, but be conservative if
        # available memory is very low (suggests high current usage)
        calculated_pool = total_memory_gb - system_reserve_gb
        
        # If available memory is very low (< 20% of total), be more conservative
        # Otherwise, use the calculated pool (which allows using currently used memory
        # that may be freed up during processing)
        if available_memory_gb < (total_memory_gb * 0.2):
            # Very low available memory - use only a portion of available
            worker_memory_pool = available_memory_gb * 0.8
            logger.warning(
                f"Low available memory ({available_memory_gb:.1f}GB, {used_memory_gb:.1f}GB used). "
                f"Using conservative allocation based on available memory only."
            )
        else:
            # Normal case - use calculated pool (total - system reserve)
            worker_memory_pool = calculated_pool
        
        # Calculate optimal memory per worker
        # Account for target utilization to stay below 80% pause threshold
        # Also account for memory fragmentation (reduce effective pool by 15%)
        fragmentation_factor = 0.85  # Assume 15% overhead from fragmentation
        effective_pool_after_fragmentation = worker_memory_pool * fragmentation_factor
        
        effective_memory_per_worker = (effective_pool_after_fragmentation / num_workers) * (target_utilization_pct / 100.0)
        
        # CRITICAL FIX: If calculated memory is too low (< min_memory_gb),
        # warn the user that they have too many workers for this RAM.
        # We enforce the minimum to prevent constant pausing.
        if effective_memory_per_worker < min_memory_gb:
            logger.warning(
                f"Calculated memory per worker ({effective_memory_per_worker:.2f}GB) is below minimum ({min_memory_gb}GB). "
                f"Forcing minimum memory limit. This may cause Dask warnings or spills. "
                f"Consider reducing worker count."
            )
            effective_memory_per_worker = min_memory_gb

        # Apply min/max bounds
        memory_per_worker_gb = max(
            min_memory_gb,
            min(effective_memory_per_worker, max_memory_gb)
        )
        
        # Round to 2 decimal places for cleaner output
        memory_per_worker_gb = round(memory_per_worker_gb, 2)
        
        # Convert to string format for Dask
        memory_limit_str = f"{memory_per_worker_gb}GB"
        
        logger.info(
            f"Dynamic memory allocation: "
            f"Total RAM: {total_memory_gb:.1f}GB, "
            f"Available: {available_memory_gb:.1f}GB, "
            f"System reserve: {system_reserve_gb:.1f}GB ({system_reserve_pct}%), "
            f"Worker pool: {worker_memory_pool:.1f}GB (after fragmentation: {effective_pool_after_fragmentation:.1f}GB), "
            f"Per worker: {memory_limit_str} (target {target_utilization_pct}% utilization, "
            f"accounts for fragmentation, stays below 80% pause threshold)"
        )
        
        return memory_limit_str
        
    except Exception as e:
        logger.warning(
            f"Failed to calculate dynamic memory allocation: {e}. "
            f"Using conservative default: 3GB per worker."
        )
        return "3GB"


def get_system_memory_info() -> Tuple[float, float, float]:
    """Get system memory information.
    
    Returns:
        Tuple of (total_gb, available_gb, used_gb)
    """
    if not PSUTIL_AVAILABLE:
        return (0.0, 0.0, 0.0)
    
    try:
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024 ** 3)
        available_gb = memory.available / (1024 ** 3)
        used_gb = memory.used / (1024 ** 3)
        return (total_gb, available_gb, used_gb)
    except Exception:
        return (0.0, 0.0, 0.0)
