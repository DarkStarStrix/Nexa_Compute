"""Runtime controller that executes tool calls on behalf of the model."""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Protocol, Sequence, Set

from nexa_data.schemas import (
    FINALREPORT_JSON_SCHEMA,
    FinalReport,
    ToolCall,
    ToolName,
    ToolResult,
    TOOLCALL_JSON_SCHEMA,
)
from nexa_tools import PaperFetcher, PaperSearcher, SandboxRunner, UnitConverter
from nexa_tools.toolresult_schema import ToolResultMeta

LOGGER = logging.getLogger(__name__)

_TOOLCALL_PATTERN = re.compile(r"~~~toolcall\s*(\{.*?\})\s*~~~", re.DOTALL | re.IGNORECASE)
_TOOLRESULT_WRAPPER = "~~~toolresult\n{json}\n~~~"
_FINAL_PATTERN = re.compile(r"~~~final\s*(\{.*?\})\s*~~~", re.DOTALL | re.IGNORECASE)


class ModelClient(Protocol):
    """Interface for language model backends."""

    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        """Return the assistant response for the provided conversation."""


class ToolClient(Protocol):
    """Interface abstracting tool execution backends."""

    def call(self, tool_call: ToolCall) -> ToolResult:
        """Execute the requested tool and return structured results."""


@dataclass
class ControllerRun:
    """Structured output from executing the controller loop."""

    messages: List[Dict[str, str]]
    tool_calls: List[ToolCall] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    fetched_dois: Set[str] = field(default_factory=set)


class LocalToolClient:
    """Tool client that executes requests inside the current process."""

    def __init__(self) -> None:
        self._sandbox = SandboxRunner()
        self._paper_searcher = PaperSearcher()
        self._paper_fetcher = PaperFetcher()
        self._unit_converter = UnitConverter()

    def call(self, tool_call: ToolCall) -> ToolResult:
        start = time.perf_counter()
        if tool_call.tool == ToolName.PYTHON_RUN:
            result = self._sandbox.run(tool_call.args.get("code", ""), timeout_s=int(tool_call.args.get("timeout_s", 10)))
            payload = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "artifacts": result.artifacts,
            }
        elif tool_call.tool == ToolName.PAPERS_SEARCH:
            results = self._paper_searcher.search(
                tool_call.args.get("query", ""),
                top_k=int(tool_call.args.get("top_k", 5)),
                corpus=str(tool_call.args.get("corpus", "crossref")),
            )
            payload = {"results": results}
        elif tool_call.tool == ToolName.PAPERS_FETCH:
            payload = self._paper_fetcher.fetch(tool_call.args.get("doi", ""))
        elif tool_call.tool == ToolName.UNITS_CONVERT:
            payload = self._unit_converter.convert(
                tool_call.args.get("value", 1.0),
                from_unit=str(tool_call.args.get("from_unit", "mol")),
                to_unit=str(tool_call.args.get("to_unit", "mmol")),
            )
        elif tool_call.tool == ToolName.THINK:
            payload = {
                "notes": tool_call.args.get("goal", ""),
                "budget_tokens": tool_call.args.get("budget_tokens", 256),
            }
        else:  # pragma: no cover - guarded by Enum
            raise ValueError(f"Unsupported tool: {tool_call.tool}")

        latency_ms = int((time.perf_counter() - start) * 1000)
        tool_result = ToolResult(
            id=tool_call.id,
            ok=True,
            data=payload,
            error=None,
            meta=ToolResultMeta(ms=latency_ms),
        )
        return tool_result


class ToolController:
    """Execute model-tool interactions with guardrails."""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        tool_client: ToolClient | None = None,
        max_rounds: int = 5,
        allow_repair_attempts: int = 1,
    ) -> None:
        self._model = model_client
        self._tool_client = tool_client or LocalToolClient()
        self._max_rounds = max_rounds
        self._repair_attempts = allow_repair_attempts

    def run(self, initial_messages: Sequence[Dict[str, str]]) -> ControllerRun:
        """Drive a full turn-taking loop until the model emits a FinalReport."""

        messages = list(initial_messages)
        tool_calls: List[ToolCall] = []
        artifacts: List[str] = []
        fetched_dois: Set[str] = set()
        repair_budget = self._repair_attempts

        for round_idx in range(self._max_rounds):
            assistant_reply = self._model.generate(messages)
            messages.append({"role": "assistant", "content": assistant_reply})

            final_payload = self._extract_block(assistant_reply, _FINAL_PATTERN)
            if final_payload is not None:
                LOGGER.info("Final report detected on round %s.", round_idx + 1)
                final_message = self._post_process_final(final_payload, fetched_dois)
                messages[-1]["content"] = final_message
                break

            tool_payload = self._extract_block(assistant_reply, _TOOLCALL_PATTERN)
            if tool_payload is None:
                LOGGER.info("No tool call found; terminating loop.")
                break

            try:
                tool_call = self._parse_toolcall(tool_payload)
            except ValueError as exc:
                if repair_budget <= 0:
                    raise
                LOGGER.warning("Tool call parsing failed (%s). Attempting repair.", exc)
                repaired = self._attempt_repair(tool_payload)
                tool_call = self._parse_toolcall(repaired)
                repair_budget -= 1

            tool_calls.append(tool_call)
            tool_result = self._tool_client.call(tool_call)
            messages.append({"role": "tool", "content": self._format_toolresult(tool_result)})

            if tool_call.tool == ToolName.PAPERS_FETCH and isinstance(tool_result.data, dict):
                doi = tool_result.data.get("doi")
                if doi:
                    fetched_dois.add(str(doi))

            artifacts.extend(_extract_artifacts(tool_result))

        return ControllerRun(messages=messages, tool_calls=tool_calls, artifacts=artifacts, fetched_dois=fetched_dois)

    def _parse_toolcall(self, payload: str) -> ToolCall:
        candidate = json.loads(payload)
        ToolCall.model_validate_json(json.dumps(candidate))  # Validation side-effect
        return ToolCall.model_validate(candidate)

    def _attempt_repair(self, payload: str) -> str:
        try:
            python_obj = ast.literal_eval(payload)
            return json.dumps(python_obj)
        except (SyntaxError, ValueError):
            raise ValueError("Unable to repair malformed toolcall payload.")

    def _extract_block(self, content: str, pattern: re.Pattern[str]) -> str | None:
        match = pattern.search(content)
        if not match:
            return None
        return match.group(1).strip()

    def _post_process_final(self, payload: str, fetched_dois: Set[str]) -> str:
        try:
            final_data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid final report JSON: {exc}") from exc

        FinalReport.model_validate(final_data)
        cited_dois = {entry.get("doi") for entry in final_data.get("evidence", []) if entry.get("doi")}
        missing_fetch = cited_dois - fetched_dois
        if missing_fetch:
            LOGGER.warning("Final report cites DOIs without papers.fetch: %s", missing_fetch)
            final_data.setdefault("limitations", []).append(
                f"Controller inserted limitation: verify citation(s) {sorted(missing_fetch)} with papers.fetch."
            )
        return _wrap_final(final_data)

    def _format_toolresult(self, result: ToolResult) -> str:
        json_payload = result.model_dump(mode="json")
        return _TOOLRESULT_WRAPPER.format(json=json.dumps(json_payload, indent=2, ensure_ascii=False))


def _wrap_final(payload: Dict[str, Any]) -> str:
    return f"~~~final\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n~~~"


def _extract_artifacts(result: ToolResult) -> List[str]:
    if not isinstance(result.data, dict):
        return []
    artifacts = result.data.get("artifacts", [])
    if isinstance(artifacts, list):
        return [str(item) for item in artifacts]
    return []


__all__ = ["ModelClient", "ToolClient", "ToolController", "ControllerRun", "LocalToolClient"]

