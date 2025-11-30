# Nexa Tool Server

> **Scope**: Sandboxed Execution and External APIs.
> **Modules**: `nexa_tools`

The Tool Server provides a secure environment for agents to execute code and access external knowledge. It is designed to be called by the `nexa_inference` controller or during synthetic data generation.

## Core Tools

### 1. Python Sandbox (`nexa_tools/sandbox.py`)
*   **Capabilities**: Executes Python code in an isolated environment.
*   **Security**: Limits network access and filesystem operations.
*   **Returns**: `stdout`, `stderr`, and generated artifacts (e.g., plots).

### 2. Paper Search (`nexa_tools/papers.py`)
*   **Search**: Queries academic databases (Arxiv, Crossref) for relevant literature.
*   **Fetch**: Retrieves abstracts and full text (where available).

### 3. Unit Conversion (`nexa_tools/units.py`)
*   **Capabilities**: Precise physical unit conversion using the `Pint` library.
*   **Purpose**: Prevents LLM hallucination on math/physics constants.

## Usage

```python
from nexa_tools.server import ToolServer

server = ToolServer()
result = server.execute_tool("python.run", {"code": "print('Hello World')"})
print(result)
```

