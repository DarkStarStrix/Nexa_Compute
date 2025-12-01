# Nexa Tools

> 📚 **Full Documentation**: [docs/pipelines/TOOLS.md](../../docs/pipelines/TOOLS.md)

## Overview

The `nexa_tools` module provides a suite of executable utilities designed for **Agentic Workflows**. These tools allow models to interact with the external world—executing code, searching literature, and performing calculations—in a structured and controlled environment. The tools are typically exposed via a FastAPI server (`nexa_tools.server`) that the `nexa_inference` controller calls.

## Key Components

### `server.py`
The HTTP interface for the toolset. It wraps the individual tool implementations into standard API endpoints.

#### Classes
*   `ToolServer`
    *   Initializes the FastAPI app and registers routes for `python/run`, `papers/search`, `papers/fetch`, `units/convert`, and `think`.

### `sandbox.py`
Provides an isolated environment for executing model-generated Python code.

#### Classes
*   `SandboxRunner`
    *   `run(code: str, timeout_s: int) -> SandboxResult`
        *   Executes Python code in a temporary directory using a subprocess. Captures `stdout`, `stderr`, and any generated file artifacts.

### `papers.py`
 integrations with scholarly databases (currently Crossref) to retrieve scientific literature.

#### Classes
*   `PaperSearcher`
    *   `search(query: str, top_k: int, corpus: str) -> List[Dict]`
        *   Queries the Crossref API for papers matching the search terms.
*   `PaperFetcher`
    *   `fetch(doi: str) -> Dict`
        *   Retrieves detailed metadata (abstract, authors, BibTeX) for a specific Digital Object Identifier (DOI).

### `units.py`
A wrapper around the `Pint` library for handling physical unit conversions.

#### Classes
*   `UnitConverter`
    *   `convert(value: float, from_unit: str, to_unit: str) -> Dict`
        *   Converts a numeric value from one unit to another (e.g., "meters" to "feet").
