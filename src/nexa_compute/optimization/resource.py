"""Resource optimization recommendations."""

from __future__ import annotations

import logging
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)


class ResourceOptimizer:
    """Analyzes metrics to provide optimization advice."""

    def analyze_batch_size(
        self,
        gpu_util: float,
        gpu_mem_mb: float,
        gpu_mem_capacity_mb: float,
        batch_size: int,
    ) -> Optional[Dict[str, str]]:
        """Suggest batch size adjustments."""
        mem_util = gpu_mem_mb / gpu_mem_capacity_mb
        
        if mem_util < 0.5 and gpu_util < 80:
            new_bs = batch_size * 2
            return {
                "action": "increase_batch_size",
                "reason": f"Low memory utilization ({mem_util:.1%})",
                "suggestion": f"Try increasing batch size to {new_bs}",
            }
            
        if mem_util > 0.95:
            new_bs = max(1, batch_size // 2)
            return {
                "action": "decrease_batch_size",
                "reason": f"Memory pressure critical ({mem_util:.1%})",
                "suggestion": f"Decrease batch size to {new_bs}",
            }
            
        return None

    def recommend_instance_type(
        self,
        avg_gpu_util: float,
        avg_cpu_util: float,
        current_instance: str,
    ) -> Optional[Dict[str, str]]:
        """Suggest instance type changes."""
        if avg_gpu_util < 30:
            return {
                "action": "downgrade_gpu",
                "reason": f"Low GPU utilization ({avg_gpu_util:.1%})",
                "suggestion": "Consider switching to a smaller GPU instance (e.g., g4dn.xlarge -> g4dn.large)",
            }
            
        if avg_cpu_util > 90 and avg_gpu_util < 60:
            return {
                "action": "upgrade_cpu",
                "reason": "CPU bottleneck detected (GPU starving)",
                "suggestion": "Switch to an instance with more CPU cores to feed the GPU faster",
            }
            
        return None

