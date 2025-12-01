"""Cost monitoring and alerting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from nexa_compute.monitoring.alerts import AlertSeverity, send_alert

LOGGER = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    project_id: str
    monthly_limit_usd: float
    alert_threshold_percent: float = 80.0


class CostMonitor:
    """Tracks spending against budgets."""

    def __init__(self) -> None:
        self.budgets: Dict[str, BudgetConfig] = {}
        self.current_spend: Dict[str, float] = {}

    def set_budget(self, config: BudgetConfig) -> None:
        self.budgets[config.project_id] = config
        if config.project_id not in self.current_spend:
            self.current_spend[config.project_id] = 0.0

    def track_cost(self, project_id: str, amount: float) -> None:
        """Record cost accumulation."""
        if project_id not in self.current_spend:
            self.current_spend[project_id] = 0.0
            
        self.current_spend[project_id] += amount
        self._check_budget(project_id)

    def _check_budget(self, project_id: str) -> None:
        if project_id not in self.budgets:
            return
            
        budget = self.budgets[project_id]
        spend = self.current_spend[project_id]
        percent = (spend / budget.monthly_limit_usd) * 100
        
        if percent >= 100:
            self._alert(
                project_id,
                f"Budget Exceeded: {percent:.1f}% used (${spend:.2f} / ${budget.monthly_limit_usd:.2f})",
                AlertSeverity.CRITICAL,
            )
        elif percent >= budget.alert_threshold_percent:
            self._alert(
                project_id,
                f"Budget Warning: {percent:.1f}% used (${spend:.2f} / ${budget.monthly_limit_usd:.2f})",
                AlertSeverity.WARNING,
            )

    def _alert(self, project_id: str, message: str, severity: AlertSeverity) -> None:
        # Deduplicate alerts in production (e.g. only send once per day)
        send_alert(
            title=f"Cost Alert: {project_id}",
            message=message,
            severity=severity,
            source="cost_monitor",
            metadata={"project": project_id, "spend": self.current_spend[project_id]},
        )

# Global instance
_COST_MONITOR = CostMonitor()

def track_project_cost(project_id: str, amount: float) -> None:
    _COST_MONITOR.track_cost(project_id, amount)

