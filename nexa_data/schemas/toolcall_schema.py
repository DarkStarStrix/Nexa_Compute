"""JSON schema definition for tool invocation payloads."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class ToolName(str, Enum):
    """Enumeration of supported tool identifiers."""

    PYTHON_RUN = "python.run"
    PAPERS_SEARCH = "papers.search"
    PAPERS_FETCH = "papers.fetch"
    UNITS_CONVERT = "units.convert"
    THINK = "think"


class ReturnFormat(str, Enum):
    """Enumeration of supported tool return payload formats."""

    JSON = "json"
    TABLE = "table"
    TEXT = "text"


class ToolCall(BaseModel):
    """Pydantic model that mirrors the tool invocation JSON schema."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique identifier for correlating tool calls and results.")
    tool: ToolName = Field(..., description="Name of the tool to invoke.")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary keyword arguments specific to the selected tool.",
    )
    return_format: ReturnFormat = Field(
        ...,
        description="Preferred formatting for the tool result payload.",
    )


TOOLCALL_JSON_SCHEMA: Dict[str, Any] = ToolCall.model_json_schema()

__all__ = ["ToolName", "ReturnFormat", "ToolCall", "TOOLCALL_JSON_SCHEMA"]

