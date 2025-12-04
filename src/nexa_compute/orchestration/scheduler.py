"""Workflow scheduler and execution engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from nexa_compute.core.logging import get_logger
from nexa_compute.orchestration.workflow import WorkflowDefinition

LOGGER = get_logger(__name__)


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class WorkflowRun:
    run_id: str
    workflow_name: str
    status: WorkflowStatus
    start_time: float
    end_time: Optional[float] = None
    step_status: Dict[str, str] = None # type: ignore

    def __post_init__(self) -> None:
        if self.step_status is None:
            self.step_status = {}


class WorkflowScheduler:
    """Manages workflow execution and scheduling."""

    def __init__(self) -> None:
        self.runs: Dict[str, WorkflowRun] = {}
        self.definitions: Dict[str, WorkflowDefinition] = {}

    def register_workflow(self, definition: WorkflowDefinition) -> None:
        self.definitions[definition.name] = definition
        LOGGER.info(f"Workflow registered: {definition.name}")

    def trigger_workflow(
        self,
        workflow_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a workflow execution."""
        if workflow_name not in self.definitions:
            raise ValueError(f"Workflow {workflow_name} not found")

        import uuid
        run_id = uuid.uuid4().hex
        
        run = WorkflowRun(
            run_id=run_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.PENDING,
            start_time=time.time(),
        )
        self.runs[run_id] = run
        
        # In a real system, this would submit to a task queue (Celery/Ray)
        # For now, we'll simulate async execution
        LOGGER.info(f"Workflow triggered: {workflow_name} (ID: {run_id})")
        self._execute_run(run_id, parameters or {})
        
        return run_id

    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        return self.runs.get(run_id)

    def cancel_run(self, run_id: str) -> bool:
        if run_id not in self.runs:
            return False
        
        run = self.runs[run_id]
        if run.status in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING}:
            run.status = WorkflowStatus.CANCELLED
            run.end_time = time.time()
            return True
        return False

    def _execute_run(self, run_id: str, params: Dict[str, Any]) -> None:
        # Mock execution logic
        run = self.runs[run_id]
        defn = self.definitions[run.workflow_name]
        
        run.status = WorkflowStatus.RUNNING
        
        # Simple topological sort execution
        # In production, this would be handled by the DAG engine
        try:
            for step in defn.steps:
                run.step_status[step.step_id] = "RUNNING"
                # Simulate work
                time.sleep(0.1) 
                run.step_status[step.step_id] = "COMPLETED"
                
            run.status = WorkflowStatus.COMPLETED
        except Exception as e:
            LOGGER.error(f"Workflow failed: {e}")
            run.status = WorkflowStatus.FAILED
        finally:
            run.end_time = time.time()

# Global instance
_SCHEDULER = WorkflowScheduler()

def get_scheduler() -> WorkflowScheduler:
    return _SCHEDULER

