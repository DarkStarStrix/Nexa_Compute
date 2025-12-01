# Nexa Inference

> 📚 **Full Documentation**:
> - [Implementation Spec](../../docs/Spec's/Nexa_inference_spec.md)
> - [Pipeline Design](../../docs/Spec's/inference_pipeline.md)
> - [General Inference Docs](../../docs/pipelines/INFERENCE.md)

## Overview

The `nexa_inference` module provides a high-performance serving layer for Nexa models. Its primary focus is on **Spectral Inference** (MS/MS embedding and retrieval) but also supports generic model serving and agentic tool execution.

It integrates with **Qdrant** for vector search and **FastAPI** for the REST interface.

## Key Components

### `spectral_server.py`
The specialized server for Mass Spectrometry data. It exposes endpoints for embedding raw spectra, searching against a vector database, and predicting molecular properties.

#### Classes
*   `SpectralInferenceServer`
    *   Manages the lifecycle of the embedding model, vector DB client, and preprocessor.
    *   **Endpoints:**
        *   `POST /embed`: Converts raw m/z and intensity arrays into a vector.
        *   `POST /search`: Finds similar spectra using cosine similarity + reranking.
        *   `POST /nearest`: Retrieves nearest neighbors for a given spectrum ID.
        *   `POST /predict/*`: Predicts properties like Retention Time or Ion Mobility.

### `embedding.py`
The core inference engine that wraps PyTorch encoder models.

#### Classes
*   `EmbeddingEngine`
    *   `embed(spectrum_tensor: Tensor) -> Tensor`:
        *   Runs the model forward pass and extracts the [CLS] token or pooled embedding.
    *   `embed_batch(...)`: optimized batch processing.

### `preprocessor.py`
Transforms raw spectral data into the tensor format expected by the model.

#### Classes
*   `SpectrumPreprocessor`
    *   Handles binning, intensity normalization, and entropy filtering to remove noise.
    *   `preprocess(mz, intensity, ...) -> Tensor`: Converts arrays to dense tensors.

### `vector_db.py`
A client wrapper for the Qdrant vector database.

#### Classes
*   `VectorDBClient`
    *   `upload_embeddings(...)`: Batch inserts vectors with metadata.
    *   `search(...)`: Performs approximate nearest neighbor (ANN) search with filtering.

### `controller.py`
The runtime environment for "Agentic" models that use tools.

#### Classes
*   `ToolController`
    *   `run(messages) -> ControllerRun`:
        *   Executes a conversation loop, detecting tool calls emitted by the model, executing them (via `nexa_tools`), and feeding the results back until completion.

### `cli.py`
Command-line entry points for the module.

#### Commands
*   `nexa-inference serve-spectral`: Launches the spectral inference server.
*   `nexa-inference process-batch`: Runs batch embedding generation on a directory of data shards.

### `reranker.py`
Refines vector search results using domain-specific metadata.

#### Classes
*   `Reranker`
    *   `rerank(...)`:
        *   Filters results based on retention time tolerance and other metadata before sorting by similarity score.
