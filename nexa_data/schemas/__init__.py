"""Convenience exports for tool protocol JSON schemas."""

from __future__ import annotations

from .finalreport_schema import (
    EvidenceEntry,
    FINALREPORT_JSON_SCHEMA,
    FinalReport,
    Figure,
    JsonScalar,
    Method,
    Metric,
    PlanStep,
    ResultsBlock,
    Table,
)
from .toolcall_schema import ReturnFormat, ToolCall, ToolName, TOOLCALL_JSON_SCHEMA
from .toolresult_schema import ToolError, ToolResult, ToolResultMeta, TOOLRESULT_JSON_SCHEMA
from .training_config import TrainingConfig

__all__ = [
    "EvidenceEntry",
    "FINALREPORT_JSON_SCHEMA",
    "FinalReport",
    "Figure",
    "JsonScalar",
    "Method",
    "Metric",
    "PlanStep",
    "ResultsBlock",
    "ReturnFormat",
    "Table",
    "ToolCall",
    "ToolError",
    "ToolName",
    "ToolResult",
    "ToolResultMeta",
    "TOOLCALL_JSON_SCHEMA",
    "TOOLRESULT_JSON_SCHEMA",
    "TrainingConfig",
]

