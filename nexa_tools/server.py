"""FastAPI application exposing tool endpoints for the controller runtime."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .papers import PaperFetcher, PaperSearcher
from .sandbox import SandboxRunner
from .units import UnitConverter

LOGGER = logging.getLogger(__name__)


class PythonRunRequest(BaseModel):
    """Request payload for executing Python snippets."""

    code: str = Field(..., min_length=1, description="Python source code to execute.")
    timeout_s: int = Field(
        10,
        ge=1,
        le=60,
        description="Wall-clock timeout in seconds for script execution.",
    )


class PythonRunResponse(BaseModel):
    """Response payload returned after executing Python code."""

    stdout: str
    stderr: str
    artifacts: List[str]


class PaperSearchRequest(BaseModel):
    """Request model for paper search."""

    query: str = Field(..., min_length=1, description="Search query string.")
    top_k: int = Field(5, ge=1, le=25, description="Maximum number of papers to return.")
    corpus: str = Field(..., description="Target corpus identifier (e.g., crossref).")


class PaperSearchResponse(BaseModel):
    """Search results returned from the requested corpus."""

    results: List[Dict[str, Any]]


class PaperFetchRequest(BaseModel):
    """Request model for fetching a paper by DOI."""

    doi: str = Field(..., min_length=1, description="Digital Object Identifier string.")


class PaperFetchResponse(BaseModel):
    """Structured metadata for the requested DOI."""

    title: str
    abstract: Optional[str]
    year: Optional[int]
    bibtex: Optional[str]
    authors: List[str]
    doi: Optional[str]
    url: Optional[str]
    source: str


class UnitConvertRequest(BaseModel):
    """Request payload for unit conversion."""

    value: float = Field(..., description="Numeric magnitude to convert.")
    from_unit: str = Field(..., min_length=1, description="Unit expression describing the input value.")
    to_unit: str = Field(..., min_length=1, description="Unit expression describing the output value.")


class UnitConvertResponse(BaseModel):
    """Response confirming conversion output."""

    value: float
    unit: str


class ThinkRequest(BaseModel):
    """Request payload for the think tool."""

    goal: str = Field(..., min_length=1, description="Purpose of the deliberation step.")
    budget_tokens: int = Field(
        256,
        ge=1,
        le=2048,
        description="Maximum number of scratch tokens granted to the model.",
    )


class ThinkResponse(BaseModel):
    """Echoes back the scratch space allocated to the model."""

    notes: str
    budget_tokens: int


class ToolServer:
    """Wrapper around FastAPI registering tool endpoints."""

    def __init__(
        self,
        *,
        sandbox: SandboxRunner | None = None,
        paper_searcher: PaperSearcher | None = None,
        paper_fetcher: PaperFetcher | None = None,
        unit_converter: UnitConverter | None = None,
    ) -> None:
        self._sandbox = sandbox or SandboxRunner()
        self._paper_searcher = paper_searcher or PaperSearcher()
        self._paper_fetcher = paper_fetcher or PaperFetcher()
        self._unit_converter = unit_converter or UnitConverter()

        self.app = FastAPI(title="Nexa Tool Server", version="0.1.0")
        self._register_routes()

    def _register_routes(self) -> None:
        """Register FastAPI routes for tool endpoints."""

        @self.app.post("/python/run", response_model=PythonRunResponse)
        def run_python(request: PythonRunRequest) -> PythonRunResponse:
            LOGGER.info("python.run invoked with timeout=%s", request.timeout_s)
            result = self._sandbox.run(request.code, timeout_s=request.timeout_s)
            return PythonRunResponse(**result.__dict__)

        @self.app.post("/papers/search", response_model=PaperSearchResponse)
        def search_papers(request: PaperSearchRequest) -> PaperSearchResponse:
            LOGGER.info("papers.search invoked corpus=%s top_k=%s", request.corpus, request.top_k)
            try:
                results = self._paper_searcher.search(
                    request.query,
                    top_k=request.top_k,
                    corpus=request.corpus,
                )
            except NotImplementedError as exc:
                raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
            return PaperSearchResponse(results=results)

        @self.app.post("/papers/fetch", response_model=PaperFetchResponse)
        def fetch_paper(request: PaperFetchRequest) -> PaperFetchResponse:
            LOGGER.info("papers.fetch invoked doi=%s", request.doi)
            try:
                payload = self._paper_fetcher.fetch(request.doi)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
            return PaperFetchResponse(**payload)

        @self.app.post("/units/convert", response_model=UnitConvertResponse)
        def convert_units(request: UnitConvertRequest) -> UnitConvertResponse:
            LOGGER.info("units.convert invoked from=%s to=%s", request.from_unit, request.to_unit)
            try:
                payload = self._unit_converter.convert(
                    request.value,
                    from_unit=request.from_unit,
                    to_unit=request.to_unit,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Unit conversion failed.")
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
            return UnitConvertResponse(**payload)

        @self.app.post("/think", response_model=ThinkResponse)
        def think(request: ThinkRequest) -> ThinkResponse:
            LOGGER.info("think invoked goal=%s", request.goal)
            notes = (
                f"Think step requested with budget {request.budget_tokens} tokens.\n"
                f"Goal: {request.goal}"
            )
            return ThinkResponse(notes=notes, budget_tokens=request.budget_tokens)


__all__ = [
    "PythonRunRequest",
    "PythonRunResponse",
    "PaperSearchRequest",
    "PaperSearchResponse",
    "PaperFetchRequest",
    "PaperFetchResponse",
    "UnitConvertRequest",
    "UnitConvertResponse",
    "ThinkRequest",
    "ThinkResponse",
    "ToolServer",
]

