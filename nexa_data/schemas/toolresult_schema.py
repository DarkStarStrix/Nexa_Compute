"""JSON schema definition for tool execution results."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolResultMeta(BaseModel):
    """Auxiliary metadata emitted alongside tool results."""

    model_config = ConfigDict(extra="forbid")

    ms: int = Field(..., ge=0, description="Execution latency in milliseconds.")


class ToolError(BaseModel):
    """Structured error payload for failed tool executions."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., description="Human-readable error message.")
    code: Optional[str] = Field(
        default=None,
        description="Stable machine-readable error code when available.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional diagnostic metadata returned by the tool server.",
    )


class ToolResult(BaseModel):
    """Pydantic model mirroring the tool result JSON schema."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Identifier linking this result to the originating ToolCall.")
    ok: bool = Field(..., description="Flag indicating whether the tool invocation succeeded.")
    data: Any = Field(
        default=None,
        description="Return payload produced by the tool when execution succeeds.",
    )
    error: Optional[ToolError] = Field(
        default=None,
        description="Structured error information when the tool invocation fails.",
    )
    meta: ToolResultMeta = Field(
        ...,
        description="Metadata emitted by the controller about the tool invocation.",
    )

    @model_validator(mode="after")
    def ensure_error_state_is_consistent(self) -> "ToolResult":
        """Guarantee that success and error states cannot conflict."""

        if self.ok and self.error is not None:
            raise ValueError("Successful tool results must not include an error payload.")
        if not self.ok and self.error is None:
            raise ValueError("Failed tool results must include an error payload.")
        return self


TOOLRESULT_JSON_SCHEMA: Dict[str, Any] = ToolResult.model_json_schema()

__all__ = ["ToolResultMeta", "ToolError", "ToolResult", "TOOLRESULT_JSON_SCHEMA"]

