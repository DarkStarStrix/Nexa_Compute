"""Minimal DAG engine with caching, resume, and failure handling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

from .artifacts import is_complete

STATE_VERSION = 1
STATE_FILENAME = "pipeline_state.json"

DEFAULT_INPUT_KEY = "__default__"


class StepStatus(str, Enum):
    """Lifecycle state for a pipeline step."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


def _normalize_io(value: Optional[object]) -> Dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {DEFAULT_INPUT_KEY: value}
    if isinstance(value, Mapping):
        return {str(key): str(val) for key, val in value.items()}
    raise TypeError(f"unsupported IO declaration: {value!r}")


def _normalize_params(value: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if not value:
        return {}
    return {str(k): v for k, v in value.items()}


def _hash_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_step_dependencies(value: Optional[Iterable[str]]) -> List[str]:
    if not value:
        return []
    return [str(dep) for dep in value]


@dataclass(frozen=True)
class PipelineStep:
    """Declarative representation of a pipeline step."""

    step_id: str
    uses: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    backend: Optional[str] = None
    scheduler: Optional[str] = None
    params: Dict[str, object] = field(default_factory=dict)
    cache_hint: Optional[str] = None
    after: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PipelineStep":
        return cls(
            step_id=str(payload["id"]),
            uses=str(payload["uses"]),
            inputs=_normalize_io(payload.get("in")),
            outputs=_normalize_io(payload.get("out")),
            backend=str(payload["backend"]) if "backend" in payload else None,
            scheduler=str(payload["scheduler"]) if "scheduler" in payload else None,
            params=_normalize_params(payload.get("params")),
            cache_hint=str(payload["cache"]) if "cache" in payload else None,
            after=tuple(_parse_step_dependencies(payload.get("after"))),
        )

    @property
    def cache_key(self) -> str:
        payload = {
            "uses": self.uses,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "backend": self.backend,
            "scheduler": self.scheduler,
            "params": self.params,
            "cache_hint": self.cache_hint,
        }
        return _hash_payload(payload)

    @property
    def output_targets(self) -> List[str]:
        return list(self.outputs.values())


@dataclass
class StepState:
    """Persistence record for a single step."""

    status: StepStatus = StepStatus.PENDING
    cache_key: Optional[str] = None
    updated_at: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status.value,
            "cache_key": self.cache_key,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StepState":
        return cls(
            status=StepStatus(payload.get("status", StepStatus.PENDING.value)),
            cache_key=payload.get("cache_key"),
            updated_at=str(payload.get("updated_at", "")),
        )


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class PipelineState:
    """Persistent representation of DAG execution state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._steps: Dict[str, StepState] = {}

    @classmethod
    def load(cls, directory: Path) -> "PipelineState":
        path = directory / STATE_FILENAME
        state = cls(path)
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("version") != STATE_VERSION:
                return state
            for step_id, step_payload in payload.get("steps", {}).items():
                state._steps[step_id] = StepState.from_dict(step_payload)
        return state

    def save(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "steps": {step_id: state.to_dict() for step_id, state in self._steps.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def get(self, step_id: str) -> StepState:
        return self._steps.setdefault(step_id, StepState())

    def update(self, step_id: str, status: StepStatus, cache_key: Optional[str]) -> None:
        record = self.get(step_id)
        record.status = status
        record.cache_key = cache_key
        record.updated_at = _now()
        self._steps[step_id] = record
        self.save()


def _is_local_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme in ("", "file")


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(uri)


def _has_complete_marker(uri: str) -> bool:
    if not uri:
        return False
    if not _is_local_uri(uri):
        # Remote URIs cannot be checked; assume they are valid when state matches.
        return True
    path = _uri_to_path(uri)
    try:
        return is_complete(path)
    except FileNotFoundError:
        return False


class PipelineGraph:
    """Directed acyclic graph describing pipeline step ordering."""

    def __init__(self, steps: Sequence[PipelineStep]) -> None:
        self.steps = list(steps)
        self._index = {step.step_id: step for step in steps}
        self._validate()

    def _validate(self) -> None:
        seen = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError(f"duplicate step id detected: {step.step_id}")
            seen.add(step.step_id)
            for dependency in step.after:
                if dependency not in self._index:
                    raise ValueError(f"step {step.step_id} depends on unknown step {dependency}")

    def topological_order(self) -> List[PipelineStep]:
        ordered: List[PipelineStep] = []
        temporary_marks: set[str] = set()
        permanent_marks: set[str] = set()

        def visit(step: PipelineStep) -> None:
            if step.step_id in permanent_marks:
                return
            if step.step_id in temporary_marks:
                raise ValueError(f"cycle detected involving {step.step_id}")
            temporary_marks.add(step.step_id)
            for dep in step.after:
                visit(self._index[dep])
            permanent_marks.add(step.step_id)
            temporary_marks.remove(step.step_id)
            ordered.append(step)

        for step in self.steps:
            visit(step)
        return ordered


class PipelineExecutor:
    """Executes pipeline steps sequentially with caching and resume support."""

    def __init__(self, graph: PipelineGraph, state_dir: Path) -> None:
        self.graph = graph
        self.state = PipelineState.load(state_dir)
        self.state_dir = state_dir

    def should_run(self, step: PipelineStep) -> bool:
        record = self.state.get(step.step_id)
        if record.status == StepStatus.FAILED:
            return True
        if record.status in (StepStatus.PENDING, StepStatus.RUNNING):
            return True
        if record.cache_key != step.cache_key:
            return True
        if record.status != StepStatus.COMPLETE:
            return True
        if not step.output_targets:
            return False
        for uri in step.output_targets:
            if not _has_complete_marker(uri):
                return True
        return False

    def mark_running(self, step: PipelineStep) -> None:
        self.state.update(step.step_id, StepStatus.RUNNING, step.cache_key)

    def mark_failed(self, step: PipelineStep) -> None:
        self.state.update(step.step_id, StepStatus.FAILED, step.cache_key)

    def mark_complete(self, step: PipelineStep) -> None:
        self.state.update(step.step_id, StepStatus.COMPLETE, step.cache_key)

    def mark_skipped(self, step: PipelineStep) -> None:
        self.state.update(step.step_id, StepStatus.SKIPPED, step.cache_key)

    def iter_steps(self) -> Iterable[PipelineStep]:
        for step in self.graph.topological_order():
            yield step

    def run(self, handler: Callable[[PipelineStep], None]) -> None:
        """Run all steps, invoking ``handler`` for those that require execution."""

        for step in self.iter_steps():
            if not self.should_run(step):
                self.mark_skipped(step)
                continue

            unmet = [dep for dep in step.after if self.state.get(dep).status != StepStatus.COMPLETE]
            if unmet:
                raise RuntimeError(f"step {step.step_id} waiting on incomplete dependencies: {', '.join(unmet)}")

            self.mark_running(step)
            try:
                handler(step)
            except Exception:
                self.mark_failed(step)
                raise
            else:
                self.mark_complete(step)

