"""Loss smoothing utilities for training."""

from __future__ import annotations

from collections import deque
from typing import Optional


class ExponentialMovingAverage:
    """Exponential moving average for smoothing values."""
    
    def __init__(self, alpha: float = 0.95):
        """
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Higher = more smoothing.
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.value: Optional[float] = None
    
    def update(self, new_value: float) -> float:
        """Update the EMA with a new value."""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value
    
    def get(self) -> Optional[float]:
        """Get the current EMA value."""
        return self.value
    
    def reset(self) -> None:
        """Reset the EMA."""
        self.value = None


class RollingAverage:
    """Rolling window average for smoothing values."""
    
    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Number of values to average over.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def update(self, new_value: float) -> float:
        """Update the rolling average with a new value."""
        self.values.append(new_value)
        return sum(self.values) / len(self.values)
    
    def get(self) -> Optional[float]:
        """Get the current rolling average."""
        if not self.values:
            return None
        return sum(self.values) / len(self.values)
    
    def reset(self) -> None:
        """Reset the rolling average."""
        self.values.clear()


class LossTracker:
    """Tracks and smooths loss values."""
    
    def __init__(self, smoothing_method: str = "ema", **kwargs):
        """
        Args:
            smoothing_method: "ema" or "rolling"
            **kwargs: Arguments for the smoothing method
                - For EMA: alpha (default 0.95)
                - For rolling: window_size (default 10)
        """
        if smoothing_method == "ema":
            alpha = kwargs.get("alpha", 0.95)
            self.smoother = ExponentialMovingAverage(alpha=alpha)
        elif smoothing_method == "rolling":
            window_size = kwargs.get("window_size", 10)
            self.smoother = RollingAverage(window_size=window_size)
        else:
            raise ValueError(f"Unknown smoothing method: {smoothing_method}")
        
        self.raw_values = []
        self.smoothed_values = []
    
    def update(self, loss: float) -> dict[str, float]:
        """Update with a new loss value."""
        self.raw_values.append(loss)
        smoothed = self.smoother.update(loss)
        self.smoothed_values.append(smoothed)
        
        return {
            "loss": loss,
            "loss_smoothed": smoothed,
        }
    
    def get_stats(self) -> dict[str, float]:
        """Get statistics about the loss."""
        if not self.raw_values:
            return {}
        
        recent_raw = self.raw_values[-10:] if len(self.raw_values) >= 10 else self.raw_values
        recent_smoothed = self.smoothed_values[-10:] if len(self.smoothed_values) >= 10 else self.smoothed_values
        
        return {
            "loss_current": self.raw_values[-1],
            "loss_smoothed_current": self.smoothed_values[-1],
            "loss_mean_10": sum(recent_raw) / len(recent_raw),
            "loss_smoothed_mean_10": sum(recent_smoothed) / len(recent_smoothed),
            "loss_min": min(self.raw_values),
            "loss_max": max(self.raw_values),
        }
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.raw_values.clear()
        self.smoothed_values.clear()
        self.smoother.reset()

