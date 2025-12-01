# Nexa Inference Engine

> **Scope**: Model Serving, Spectral Analysis, and Tool Control.
> **Modules**: `nexa_inference`

The Inference Engine is a high-performance serving layer designed for two distinct workloads: **Spectral Inference** (mass spectrometry embedding/search) and **Agentic Generation** (LLM tool use). It leverages **FastAPI** for the control plane and **vLLM** (or native PyTorch) for the compute plane.

## 1. Spectral Inference Server (`nexa_inference/spectral_server.py`)

A specialized server for embedding and searching Mass Spectrometry (MS/MS) data.

### Architecture
1.  **Preprocessor (`nexa_inference/preprocessor.py`)**:
    *   Converts raw sparse arrays (m/z, intensity) into dense tensors.
    *   Applies **Entropy Filtering** to remove noise peaks.
    *   Performs intensity normalization (0-1 scaling) and binning.
2.  **Embedding Engine (`nexa_inference/embedding.py`)**:
    *   Wraps a 7B-parameter Transformer Encoder.
    *   **`embed_single_spectrum`**: Runs forward pass, extracts the `[CLS]` token, and applies L2 normalization.
    *   **Batching**: Optimized `embed_batch` for high-throughput indexing.
3.  **Vector Database (`nexa_inference/vector_db.py`)**:
    *   Connects to a **Qdrant** cluster.
    *   Manages collections (`nexa_spectra`) and HNSW indices.
    *   Handles metadata storage (Precursor m/z, Retention Time) alongside vectors.
4.  **Reranker (`nexa_inference/reranker.py`)**:
    *   Post-processes vector search results.
    *   Applies hard filters (e.g., "Retention Time delta < 30s").
    *   Re-sorts candidates based on metadata alignment.

### API Endpoints
*   `POST /embed`: Transform raw spectrum -> 768d vector.
*   `POST /search`: Vector similarity search with metadata filtering.
*   `POST /nearest`: Find nearest neighbors by Spectrum ID.
*   `POST /predict/{property}`: Predict molecular properties (RT, Ion Mobility) from embeddings.
*   `POST /decode/structure`: Decode embedding to SMILES/InChI (experimental).

## 2. Agentic Tool Controller (`nexa_inference/controller.py`)

The runtime environment for models that need to execute actions.

### The Control Loop
The `ToolController` implements a ReAct-style loop:
1.  **Generate**: Calls the model with the current conversation history.
2.  **Parse**: Scans output for the `~~~toolcall JSON~~~` pattern.
    *   *Self-Correction*: If JSON is malformed, it attempts to repair it or prompts the model to fix it.
3.  **Execute**: Dispatches the call to the `nexa_tools` server.
4.  **Feedback**: Formats the result as `~~~toolresult JSON~~~` and appends it to history.
5.  **Repeat**: Continues until the model emits a `~~~final JSON~~~` block.

### Integration
*   **`ModelClient` Protocol**: Abstracts the underlying LLM (local vLLM or OpenRouter).
*   **`ToolClient` Protocol**: Abstracts the execution backend (Local process or HTTP Tool Server).

## 3. CLI & Utilities

*   **`nexa-inference serve-spectral`**: Launches the MS/MS server.
    *   Args: `--checkpoint`, `--qdrant-url`, `--embedding-dim`.
*   **`nexa-inference process-batch`**: Offline batch processing.
    *   Reads a directory of Parquet shards.
    *   Generates embeddings in bulk.
    *   Uploads to Qdrant.

## Usage

**Serving Spectral Model:**
```bash
python -m nexa_inference.cli serve-spectral \
  --checkpoint artifacts/checkpoints/spectral_encoder_v1.pt \
  --port 8000
```

**Running the Agent Controller:**
```python
from nexa_inference.controller import ToolController, LocalToolClient
from nexa_eval.clients import LocalInferenceClient

model = LocalInferenceClient(base_url="http://localhost:8000")
tools = LocalToolClient()
controller = ToolController(model, tool_client=tools)

result = controller.run([{"role": "user", "content": "Find papers on CRISPR."}])
print(result.messages[-1])
```
