"""JSON schema definition for the structured FinalReport payload."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


JsonScalar = Union[str, int, float, bool, None]


class PlanStep(BaseModel):
    """Single step in the execution plan summarised for the user."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(..., ge=1, description="Sequential step number in the proposed plan.")
    desc: str = Field(..., min_length=1, description="Short natural-language description of the step.")


class Method(BaseModel):
    """Entry describing a method, script, or artifact used in the workflow."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Human-readable label for the method.")
    code_ref: str = Field(
        ...,
        min_length=1,
        description="URI or artifact reference that points to the executable asset.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional remarks clarifying how the method was applied.",
    )


class Metric(BaseModel):
    """Key quantitative result produced by the tool workflow."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Identifier for the reported metric.")
    value: Union[int, float, str] = Field(..., description="Measured value for the metric.")
    unit: Optional[str] = Field(default=None, description="Measurement unit associated with the value.")
    notes: Optional[str] = Field(default=None, description="Optional clarification about the metric.")


class Figure(BaseModel):
    """Artifact reference for visual outputs."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1, description="Artifact URI pointing to the stored figure.")
    caption: str = Field(..., min_length=1, description="Concise caption describing the figure.")


class Table(BaseModel):
    """Tabular artefact summarising structured data."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Identifier for the table.")
    rows: List[List[JsonScalar]] = Field(
        default_factory=list,
        description="Row-major table values expressed as JSON scalars.",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Optional column headers corresponding to the row values.",
    )
    notes: Optional[str] = Field(default=None, description="Supplementary detail about how to read the table.")


class ResultsBlock(BaseModel):
    """Container for the quantitative outputs in the FinalReport."""

    model_config = ConfigDict(extra="forbid")

    metrics: List[Metric] = Field(default_factory=list, description="List of metrics captured during evaluation.")
    figures: List[Figure] = Field(default_factory=list, description="List of generated figures or plots.")
    tables: List[Table] = Field(default_factory=list, description="List of structured data tables generated.")


class EvidenceEntry(BaseModel):
    """External evidence or citation supporting report claims."""

    model_config = ConfigDict(extra="forbid")

    doi: str = Field(..., min_length=1, description="Digital object identifier for the cited work.")
    title: str = Field(..., min_length=1, description="Title of the cited work.")
    where_used: str = Field(
        ...,
        min_length=1,
        description="Explanation of where in the analysis the evidence informed decisions.",
    )


class FinalReport(BaseModel):
    """Structured payload communicated back to the user after executing tools."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(..., min_length=1, description="Identifier describing the overall task or evaluation goal.")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Canonicalised inputs used in the workflow.")
    plan: List[PlanStep] = Field(default_factory=list, description="Ordered list of planned execution steps.")
    methods: List[Method] = Field(default_factory=list, description="Methods or scripts executed during the run.")
    results: ResultsBlock = Field(default_factory=ResultsBlock, description="Quantitative outputs gathered.")
    evidence: List[EvidenceEntry] = Field(
        default_factory=list,
        description="Citations or external evidence supporting the conclusions.",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations, failure modes, or caveats surfaced during execution.",
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Recommended follow-up actions to continue the investigation.",
    )


FINALREPORT_JSON_SCHEMA: Dict[str, Any] = FinalReport.model_json_schema()

__all__ = [
    "JsonScalar",
    "PlanStep",
    "Method",
    "Metric",
    "Figure",
    "Table",
    "ResultsBlock",
    "EvidenceEntry",
    "FinalReport",
    "FINALREPORT_JSON_SCHEMA",
]

