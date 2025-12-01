"""Declarative workflow definition DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexa_compute.core.dag import PipelineStep


@dataclass
class WorkflowDefinition:
    name: str
    steps: List[PipelineStep]
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 3600
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {"max_attempts": 3})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "steps": [
                {
                    "id": s.step_id,
                    "uses": s.uses,
                    "in": s.inputs,
                    "out": s.outputs,
                    "backend": s.backend,
                    "scheduler": s.scheduler,
                    "params": s.params,
                    "cache": s.cache_hint,
                    "after": list(s.after),
                }
                for s in self.steps
            ],
            "parameters": self.parameters,
            "timeout_seconds": self.timeout_seconds,
            "retry_policy": self.retry_policy,
        }


class WorkflowBuilder:
    """Builder for creating workflow definitions programmatically."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.steps: List[PipelineStep] = []
        self.parameters: Dict[str, Any] = {}

    def add_step(
        self,
        step_id: str,
        uses: str,
        inputs: Optional[Dict[str, str]] = None,
        outputs: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        backend: Optional[str] = None,
    ) -> "WorkflowBuilder":
        step = PipelineStep(
            step_id=step_id,
            uses=uses,
            inputs=inputs or {},
            outputs=outputs or {},
            params=params or {},
            backend=backend,
            after=tuple(depends_on or []),
        )
        self.steps.append(step)
        return self

    def set_parameter(self, key: str, value: Any) -> "WorkflowBuilder":
        self.parameters[key] = value
        return self

    def build(self) -> WorkflowDefinition:
        return WorkflowDefinition(
            name=self.name,
            steps=self.steps,
            parameters=self.parameters,
        )

