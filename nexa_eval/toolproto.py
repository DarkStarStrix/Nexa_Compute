"""Evaluation utilities for tool-augmented conversations."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import typer

from nexa_data.schemas import FinalReport, ToolCall, ToolResult
from nexa_tools.units import UnitConverter

import re

_TOOLCALL_PATTERN = re.compile(r"~~~toolcall\s*(\{.*?\})\s*~~~", re.DOTALL | re.IGNORECASE)
_TOOLRESULT_PATTERN = re.compile(r"~~~toolresult\s*(\{.*?\})\s*~~~", re.DOTALL | re.IGNORECASE)
_FINAL_PATTERN = re.compile(r"~~~final\s*(\{.*?\})\s*~~~", re.DOTALL | re.IGNORECASE)


@dataclass
class ConversationMetrics:
    """Per-conversation evaluation summary."""

    tool_calls: int = 0
    tool_calls_valid: int = 0
    tool_results_ok: int = 0
    tool_results: int = 0
    final_valid: bool = False
    citation_valid: bool = False
    numeric_sanity: bool = True
    latency_ms: List[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.latency_ms = self.latency_ms or []


@dataclass
class EvaluationSummary:
    """Aggregate evaluation metrics."""

    conversations: int
    final_valid_rate: float
    tool_call_valid_rate: float
    tool_success_rate: float
    citation_valid_rate: float
    numeric_sanity_rate: float
    average_tool_calls: float
    average_latency_ms: float


class ToolProtoEvaluator:
    """Evaluate generated conversations against tool protocol requirements."""

    def __init__(self) -> None:
        self._unit_converter = UnitConverter()

    def evaluate(self, conversations: Sequence[Dict[str, Any]]) -> EvaluationSummary:
        metrics: List[ConversationMetrics] = []
        for conversation in conversations:
            metrics.append(self._evaluate_conversation(conversation))
        return self._summarise(metrics)

    def _evaluate_conversation(self, conversation: Dict[str, Any]) -> ConversationMetrics:
        record = ConversationMetrics()
        messages = conversation.get("messages", [])
        executed_fetch_dois = set()
        cited_dois = set()
        tool_map: Dict[str, ToolCall] = {}

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "assistant":
                for block in _TOOLCALL_PATTERN.findall(content):
                    record.tool_calls += 1
                    try:
                        tool_call = ToolCall.model_validate_json(block)
                        tool_map[tool_call.id] = tool_call
                        record.tool_calls_valid += 1
                    except Exception:  # noqa: BLE001
                        continue
                final_block = _FINAL_PATTERN.search(content)
                if final_block and not record.final_valid:
                    try:
                        final_payload = json.loads(final_block.group(1))
                        FinalReport.model_validate(final_payload)
                        record.final_valid = True
                        for evidence in final_payload.get("evidence", []):
                            doi = evidence.get("doi")
                            if doi:
                                cited_dois.add(str(doi))
                        if final_payload.get("limitations"):
                            if any("Controller inserted limitation" in item for item in final_payload["limitations"]):
                                record.citation_valid = False
                    except Exception:  # noqa: BLE001
                        continue
            elif role == "tool":
                for block in _TOOLRESULT_PATTERN.findall(content):
                    try:
                        tool_result = ToolResult.model_validate_json(block)
                    except Exception:  # noqa: BLE001
                        continue
                    record.tool_results += 1
                    if tool_result.ok:
                        record.tool_results_ok += 1
                    if tool_result.meta:
                        record.latency_ms.append(tool_result.meta.ms)
                    call = tool_map.get(tool_result.id)
                    if call is not None:
                        if call.tool == "units.convert":
                            record.numeric_sanity &= self._check_unit_conversion(call, tool_result)
                        if call.tool == "python.run":
                            record.numeric_sanity &= self._check_python_result(tool_result)
                        if call.tool == "papers.fetch":
                            doi = None
                            if isinstance(tool_result.data, dict):
                                doi = tool_result.data.get("doi")
                            if doi:
                                executed_fetch_dois.add(str(doi))

        if record.final_valid and not record.citation_valid:
            record.citation_valid = cited_dois.issubset(executed_fetch_dois) if cited_dois else True
        elif not cited_dois:
            record.citation_valid = True
        return record

    def _summarise(self, metrics: Sequence[ConversationMetrics]) -> EvaluationSummary:
        total = len(metrics)
        if total == 0:
            return EvaluationSummary(
                conversations=0,
                final_valid_rate=0.0,
                tool_call_valid_rate=0.0,
                tool_success_rate=0.0,
                citation_valid_rate=0.0,
                numeric_sanity_rate=0.0,
                average_tool_calls=0.0,
                average_latency_ms=0.0,
            )

        final_valid = sum(1 for m in metrics if m.final_valid)
        tool_calls = sum(m.tool_calls for m in metrics)
        tool_calls_valid = sum(m.tool_calls_valid for m in metrics)
        tool_results = sum(m.tool_results for m in metrics)
        tool_success = sum(m.tool_results_ok for m in metrics)
        citation_valid = sum(1 for m in metrics if m.citation_valid)
        numeric_sanity = sum(1 for m in metrics if m.numeric_sanity)
        avg_calls = tool_calls / total if total else 0.0
        latencies = [lat for m in metrics for lat in m.latency_ms]
        avg_latency = statistics.mean(latencies) if latencies else 0.0

        return EvaluationSummary(
            conversations=total,
            final_valid_rate=final_valid / total,
            tool_call_valid_rate=(tool_calls_valid / tool_calls) if tool_calls else 0.0,
            tool_success_rate=(tool_success / tool_results) if tool_results else 0.0,
            citation_valid_rate=citation_valid / total,
            numeric_sanity_rate=numeric_sanity / total,
            average_tool_calls=avg_calls,
            average_latency_ms=avg_latency,
        )

    def _check_unit_conversion(self, tool_call: ToolCall, tool_result: ToolResult) -> bool:
        try:
            expected = self._unit_converter.convert(
                tool_call.args.get("value"),
                from_unit=tool_call.args.get("from_unit"),
                to_unit=tool_call.args.get("to_unit"),
            )
        except Exception:  # noqa: BLE001
            return False
        if not isinstance(tool_result.data, dict):
            return False
        observed = tool_result.data.get("value")
        return math.isclose(float(observed), float(expected["value"]), rel_tol=1e-3, abs_tol=1e-3)

    def _check_python_result(self, tool_result: ToolResult) -> bool:
        if not isinstance(tool_result.data, dict):
            return False
        stdout = tool_result.data.get("stdout", "")
        if not isinstance(stdout, str):
            return False
        numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", stdout)]
        return all(number > -1e3 for number in numbers)


def _load_conversations(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _format_summary(summary: EvaluationSummary) -> str:
    return json.dumps(
        {
            "conversations": summary.conversations,
            "final_valid_rate": summary.final_valid_rate,
            "tool_call_valid_rate": summary.tool_call_valid_rate,
            "tool_success_rate": summary.tool_success_rate,
            "citation_valid_rate": summary.citation_valid_rate,
            "numeric_sanity_rate": summary.numeric_sanity_rate,
            "average_tool_calls": summary.average_tool_calls,
            "average_latency_ms": summary.average_latency_ms,
        },
        indent=2,
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    conversations_path: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Evaluate a JSONL file of tool conversations."""

    conversations = _load_conversations(conversations_path)
    evaluator = ToolProtoEvaluator()
    summary = evaluator.evaluate(conversations)
    typer.echo(_format_summary(summary))


if __name__ == "__main__":
    app()

