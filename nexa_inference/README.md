# Nexa Inference

This module handles model serving and inference for Nexa-Spec MS/MS spectral analysis, including embedding generation, vector search, and real-time querying.

> 📚 **Full Documentation**: 
> - [Implementation Spec](../../docs/Spec's/Nexa_inference_spec.md)
> - [Pipeline Design](../../docs/Spec's/inference_pipeline.md)
> - [General Inference Docs](../../docs/pipelines/INFERENCE.md)

---

## 🚀 Quick Start

### Serve Spectral Inference Model

```bash
nexa-inference serve-spectral \
  --checkpoint path/to/checkpoint.pt \
  --config path/to/config.yaml \
  --qdrant-url http://localhost:6333 \
  --collection-name nexa_spectra \
  --embedding-dim 768 \
  --port 8000
```

### Embed a Spectrum (Python)

```python
from nexa_inference import SpectrumPreprocessor, EmbeddingEngine
import numpy as np

preprocessor = SpectrumPreprocessor()
engine = EmbeddingEngine(checkpoint_path=Path("model.pt"))

# Preprocess spectrum
mz_values = np.array([100.0, 200.0, 300.0])
intensity_values = np.array([0.5, 0.8, 0.3])
spectrum_tensor = preprocessor.preprocess(mz_values, intensity_values)

# Generate embedding
embedding = engine.embed(spectrum_tensor)
```

### Query via API

```bash
curl -X POST http://localhost:8000/embed \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "mz_values": [100.0, 200.0],
    "intensity_values": [0.5, 0.8],
    "precursor_mz": 500.0
  }'
```

---

## 📦 Key Components

### Core Modules

* **`preprocessor.py`** - Spectral preprocessing with entropy filtering, m/z binning, and normalization
* **`embedding.py`** - Embedding generation engine for 7B encoder models
* **`storage.py`** - Embedding storage in JSONL/Arrow formats
* **`vector_db.py`** - Qdrant/Atlas++ vector database client
* **`spectral_server.py`** - FastAPI server for spectral inference endpoints
* **`middleware.py`** - API key authentication and rate limiting
* **`server.py`** - Generic inference server (legacy)
* **`controller.py`** - Tool controller for model-tool interactions
* **`cli.py`** - Command-line interface
* **`property_predictor.py`** - Property heads (RT/IM/MW)
* **`structure_decoder.py`** - Structure decoding (SMILES/InChI)
* **`reranker.py`** - Metadata + RT-aware reranking
* **`caching.py`** - LRU/TTL embedding cache
* **`monitoring.py`** - Prometheus instrumentation helpers
* **`feedback.py`** - In-memory feedback collector
* **`federated_search.py`** - Multi-cluster search orchestration
* **`query_expansion.py`** - Diverse query generation
* **`batch_processor.py`** - Arrow/HDF5 batch ingestion utilities
* **`snapshot.py`** - Local snapshotting for offline search
* **`exporters.py`** - Export embeddings to MGF / JSON schema
* **`graphql_api.py`** - Optional GraphQL interface

---

## 🔧 API Endpoints

### `POST /embed`
Embed a raw MS/MS spectrum into vector space.

**Request:**
```json
{
  "mz_values": [100.0, 200.0, ...],
  "intensity_values": [0.5, 0.8, ...],
  "precursor_mz": 500.0,
  "retention_time": 120.5,
  "spectrum_id": "spec_123"
}
```

**Response:**
```json
{
  "embedding": [0.1, 0.2, ...],
  "spectrum_id": "spec_123",
  "latency_ms": 150.5,
  "embedding_dim": 768
}
```

**Performance Target:** < 200ms latency

---

### `POST /search`
Search for similar spectra by embedding vector.

**Request:**
```json
{
  "embedding": [0.1, 0.2, ...],
  "top_k": 10,
  "filter_conditions": {"compound_id": "C123"}
}
```

**Response:**
```json
{
  "results": [
    {
      "spectrum_id": "spec_456",
      "score": 0.95,
      "metadata": {...}
    }
  ],
  "query_latency_ms": 8.2
}
```

**Performance Target:** < 10ms latency

---

### `POST /nearest`
Get nearest neighbors for a spectrum ID or embedding.

**Request:**
```json
{
  "spectrum_id": "spec_123",
  "top_k": 10
}
```

**Response:** Same format as `/search`

---

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "embedding_dim": 768
}
```

---

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "embedding_dim": 768
}
```

---

### `POST /predict/{property}`
Predict molecular properties directly from embeddings. Supported properties:

- `retention_time`
- `ion_mobility`
- `molecular_weight`

```bash
curl -X POST http://localhost:8000/predict/retention_time \
  -H "Content-Type: application/json" \
  -d '{"mz_values":[100.0],"intensity_values":[0.5]}'
```

---

### `POST /decode/*`
Structure decoding endpoints:

- `/decode/structure` → JSON payload with SMILES/InChI + confidence
- `/decode/smiles`
- `/decode/inchi`

---

### `POST /feedback/*`
Capture user feedback for search quality:

- `/feedback/positive`
- `/feedback/negative`
- `/feedback/stats`

---

### `GET /metrics`
Prometheus metrics endpoint (requires `prometheus_client`).

---

### `GET /graphql`
Optional GraphQL endpoint (requires `graphene`). Provides lightweight embedding queries.

---

### `WEBSOCKET /stream/batch`
Streaming progress updates for long-running batch jobs.

---

## 🔒 Authentication & Rate Limiting

### License Tiers

* **Free**: 100 requests/hour
* **Pro**: 1000 requests/hour
* **Academic**: Unlimited (logged)
* **Commercial**: Unlimited (logged)

### API Key Usage

API keys are passed via the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/embed ...
```

---

## 📊 Pipeline Overview

1. **Spectral Ingestion** - Load Arrow/HDF5 shards (2GB per shard)
2. **Preprocessing** - Entropy filtering, m/z binning, padding
3. **Inference** - 7B encoder-only Transformer → 512-1024 dim embedding
4. **Storage** - JSONL/Arrow format with metadata
5. **Vector DB** - Qdrant HNSW index for fast search
6. **Query Gateway** - FastAPI endpoints with rate limiting
7. **Postprocessing** - Optional property heads (RT, IM) and structure decoder

---

## 🎯 Performance Targets

| Component         | Target             |
|------------------|--------------------|
| Embedding latency| < 200ms / sample   |
| Search latency   | < 10ms / query     |
| Full round trip  | < 300ms per call   |
| DB scale         | 1M+ spectra vectors|

---

## 💻 Usage Examples

### Batch Processing

```python
from nexa_inference import (
    SpectrumPreprocessor,
    EmbeddingEngine,
    EmbeddingStorage,
    VectorDBClient
)

# Initialize components
preprocessor = SpectrumPreprocessor()
engine = EmbeddingEngine(checkpoint_path=Path("model.pt"))
storage = EmbeddingStorage(output_dir=Path("./embeddings"))
vector_db = VectorDBClient(url="http://localhost:6333")

# Process batch
spectra = [
    (np.array([100.0, 200.0]), np.array([0.5, 0.8])),
    (np.array([150.0, 250.0]), np.array([0.6, 0.9])),
]
spectrum_ids = ["spec_1", "spec_2"]

# Preprocess
tensors = preprocessor.batch_preprocess(spectra)

# Generate embeddings
embeddings = engine.embed_batch(tensors)

# Save to disk
storage.save_embeddings(
    embeddings=embeddings.numpy(),
    spectrum_ids=spectrum_ids,
    metadata=[{"precursor_mz": 500.0}, {"precursor_mz": 550.0}]
)

# Upload to vector DB
vector_db.upload_embeddings(
    embeddings=embeddings.numpy(),
    ids=spectrum_ids,
    metadata=[{"precursor_mz": 500.0}, {"precursor_mz": 550.0}]
)
```

### Vector Search

```python
from nexa_inference import VectorDBClient
import numpy as np

client = VectorDBClient(url="http://localhost:6333")

# Search by embedding
query_embedding = np.array([0.1, 0.2, ...])
results = client.search(query_embedding, top_k=10)

# Search by spectrum ID
results = client.get_nearest(spectrum_id="spec_123", top_k=10)

# Filtered search
results = client.search(
    query_embedding,
    top_k=10,
    filter_conditions={"compound_id": "C123"}
)
```

---

## 🛠️ Configuration

### Preprocessor Configuration

```python
from nexa_inference import SpectrumPreprocessorConfig, SpectrumPreprocessor

config = SpectrumPreprocessorConfig(
    mz_min=0.0,
    mz_max=2000.0,
    mz_bin_size=0.1,
    max_peaks=1000,
    entropy_threshold=0.5,
    normalize_intensity=True,
    apply_augmentation=False  # Disabled in inference
)

preprocessor = SpectrumPreprocessor(config)
```

### Server Configuration

```python
from nexa_inference import serve_spectral_model

serve_spectral_model(
    checkpoint_path=Path("model.pt"),
    config_path=Path("config.yaml"),
    qdrant_url="http://localhost:6333",
    collection_name="nexa_spectra",
    embedding_dim=768,
    api_keys={
        "key-123": "pro",
        "key-456": "academic"
    }
)
```

---

## 📚 Dependencies

### Required
- `torch>=2.1.0` - PyTorch for model inference
- `fastapi>=0.115` - Web framework
- `uvicorn>=0.30` - ASGI server
- `numpy>=1.26` - Numerical operations
- `scipy>=1.11.0` - Scientific computing

### Optional
- `qdrant-client` - Vector database client
- `pyarrow>=14.0.0` - Arrow/Parquet support
- `pandas>=2.1` - Data manipulation

---

## 🔗 Integration Points

* **Model Registry**: Integrates with `nexa_train.models.DEFAULT_MODEL_REGISTRY`
* **Artifacts**: Compatible with `nexa_compute.core.artifacts` for checkpoint resolution
* **Data Pipeline**: Can integrate with `nexa_data` loaders for batch processing

---

## 🧪 Future Enhancements

* Batch processing CLI for Arrow/HDF5 shards
* Property heads (RT/IM prediction endpoints)
* Structure decoder for molecular structure generation
* Reranking with RT delta and metadata filters
* Prometheus metrics integration
* Multi-GPU distributed inference via vLLM

---

## 📖 Additional Resources

* [Implementation Specification](../../docs/Spec's/Nexa_inference_spec.md) - Detailed technical documentation
* [Pipeline Design](../../docs/Spec's/inference_pipeline.md) - Architecture overview
* [General Inference Docs](../../docs/pipelines/INFERENCE.md) - General inference pipeline documentation
