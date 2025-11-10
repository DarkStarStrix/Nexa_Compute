"""Stubs for budget and retention policy enforcement.

The v2 spec calls for budgets and retention to be enforced across pipeline
steps. This module provides light-weight placeholders so that downstream
components can depend on a stable API today while the full implementation is
designed. Policies currently act as pass-through validators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

__all__ = [
    "BudgetPolicy",
    "RetentionPolicy",
    "PolicyBundle",
    "load_policies",
    "BudgetExceededError",
]


class BudgetExceededError(RuntimeError):
    """Raised when a step exceeds its configured budget."""


@dataclass(frozen=True)
class BudgetPolicy:
    """Budget thresholds applied to a pipeline step or run."""

    max_wallclock_hours: Optional[float] = None
    max_cost_usd: Optional[float] = None
    max_gpu_hours: Optional[float] = None

    def evaluate(self, *, wallclock_hours: float | None = None, cost_usd: float | None = None, gpu_hours: float | None = None) -> bool:
        """Return ``True`` if the supplied metrics satisfy the policy."""

        if self.max_wallclock_hours is not None and wallclock_hours is not None:
            if wallclock_hours > self.max_wallclock_hours:
                return False
        if self.max_cost_usd is not None and cost_usd is not None:
            if cost_usd > self.max_cost_usd:
                return False
        if self.max_gpu_hours is not None and gpu_hours is not None:
            if gpu_hours > self.max_gpu_hours:
                return False
        return True


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention parameters for artifact housekeeping."""

    keep_versions: int | None = None
    keep_days: int | None = None

    def should_reclaim(self, *, version_count: int | None = None) -> bool:
        """Return ``True`` if old artifacts should be reclaimed (stub)."""

        if self.keep_versions is not None and version_count is not None:
            return version_count > self.keep_versions
        return False


@dataclass(frozen=True)
class PolicyBundle:
    """Convenience wrapper for budget and retention policies."""

    budget: BudgetPolicy = BudgetPolicy()
    retention: RetentionPolicy = RetentionPolicy()


def load_policies(config: Mapping[str, object] | None) -> PolicyBundle:
    """Construct policies from a configuration mapping.

    Parameters
    ----------
    config:
        Mapping containing optional ``budget`` and ``retention`` sections.
        Values outside the known keys are ignored for forwards compatibility.
    """

    config = config or {}
    budget_cfg = config.get("budget", {}) if isinstance(config, Mapping) else {}
    retention_cfg = config.get("retention", {}) if isinstance(config, Mapping) else {}

    budget_policy = BudgetPolicy(
        max_wallclock_hours=_coerce_float(budget_cfg, "max_wallclock_hours"),
        max_cost_usd=_coerce_float(budget_cfg, "max_cost_usd"),
        max_gpu_hours=_coerce_float(budget_cfg, "max_gpu_hours"),
    )
    retention_policy = RetentionPolicy(
        keep_versions=_coerce_int(retention_cfg, "keep_versions"),
        keep_days=_coerce_int(retention_cfg, "keep_days"),
    )
    return PolicyBundle(budget=budget_policy, retention=retention_policy)


def _coerce_float(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _coerce_int(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None

