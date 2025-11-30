# Nexa-Spec Inference Implementation Specification

This document details the implementation of the Nexa-Spec inference pipeline for MS/MS spectral embedding and search, as specified in `inference_pipeline.md`.

---

## 📋 Implementation Overview

The `nexa_inference` module provides a complete pipeline for:
1. Preprocessing raw MS/MS spectra
2. Generating embeddings using a 7B encoder model
3. Storing embeddings in JSONL/Arrow format
4. Indexing embeddings in Qdrant vector database
5. Serving real-time queries via FastAPI
6. Rate limiting and access control by license tier

---

## 🏗️ Architecture

### Module Structure

```
nexa_inference/
├── __init__.py              # Module exports
├── preprocessor.py          # Spectrum preprocessing
├── embedding.py             # Embedding generation engine
├── storage.py               # Embedding storage (JSONL/Arrow)
├── vector_db.py            # Qdrant client
├── middleware.py           # API key & rate limiting
├── spectral_server.py      # FastAPI server for spectral inference
├── server.py               # Generic inference server (legacy)
├── controller.py           # Tool controller (existing)
└── cli.py                  # CLI commands
```

---

## 🔧 Component Details

### 1. Spectral Preprocessing (`preprocessor.py`)

**Class**: `SpectrumPreprocessor`

**Configuration**: `SpectrumPreprocessorConfig`
- `mz_min`: Minimum m/z value (default: 0.0)
- `mz_max`: Maximum m/z value (default: 2000.0)
- `mz_bin_size`: Bin size for m/z binning (default: 0.1)
- `max_peaks`: Maximum peaks to retain (default: 1000)
- `entropy_threshold`: Entropy threshold for filtering (default: 0.5)
- `normalize_intensity`: Normalize intensities to [0, 1] (default: True)
- `apply_augmentation`: Apply data augmentation (default: False, disabled in inference)

**Methods**:
- `preprocess()`: Process single spectrum → tensor `(N, D)`
- `batch_preprocess()`: Process batch → tensor `(B, N, D)`

**Pipeline Steps**:
1. Intensity normalization (if enabled)
2. Entropy-based filtering (removes low-entropy spectra)
3. m/z binning into fixed bins
4. Padding/truncation to fixed length
5. Augmentation (disabled in inference mode)

**Output**: PyTorch tensor ready for model input

---

### 2. Embedding Engine (`embedding.py`)

**Class**: `EmbeddingEngine`

**Initialization**:
- `checkpoint_path`: Path to model checkpoint
- `config_path`: Optional model config YAML
- `device`: CUDA/CPU device
- `embedding_dim`: Expected embedding dimension (default: 768)

**Methods**:
- `embed()`: Generate embedding from preprocessed tensor
- `embed_batch()`: Batch embedding generation
- `normalize_embeddings()`: L2-normalize for cosine similarity

**Model Loading**:
- Loads encoder-only Transformer from checkpoint
- Extracts `[CLS]` token or pools sequence output
- Supports models from `nexa_train.models.DEFAULT_MODEL_REGISTRY`

**Output**: Embedding tensor of shape `(embedding_dim,)` or `(B, embedding_dim)`

---

### 3. Embedding Storage (`storage.py`)

**Class**: `EmbeddingStorage`

**Formats Supported**:
- **JSONL**: One embedding per line with metadata
- **Arrow/Parquet**: Columnar format for efficient batch operations

**Methods**:
- `save_embeddings()`: Save embeddings with metadata
- `load_embeddings()`: Load embeddings from disk

**Metadata Fields**:
- `spectrum_id`: Unique spectrum identifier
- `compound_id`: Optional compound identifier
- `precursor_mz`: Precursor m/z value
- `retention_time`: Retention time
- `embedding`: Embedding vector (normalized)

**Features**:
- Optional L2 normalization before saving
- Metadata preservation
- Efficient batch I/O

---

### 4. Vector Database Client (`vector_db.py`)

**Class**: `VectorDBClient`

**Backend**: Qdrant (Atlas++ compatible)

**Initialization**:
- `url`: Qdrant server URL (default: `http://localhost:6333`)
- `api_key`: Optional API key
- `collection_name`: Collection name (default: `nexa_spectra`)
- `embedding_dim`: Embedding dimension

**Methods**:
- `upload_embeddings()`: Batch upload embeddings with metadata
- `search()`: Vector similarity search with filters
- `get_nearest()`: Get nearest neighbors by spectrum ID or embedding

**Index Configuration**:
- Distance metric: Cosine similarity
- Index type: HNSW (handled by Qdrant)
- Metadata filtering: Supported via Qdrant filters

**Performance**:
- Batch uploads (configurable batch size)
- Efficient similarity search (< 10ms target)

---

### 5. API Middleware (`middleware.py`)

**Classes**:
- `RateLimiter`: In-memory rate limiting
- `APIKeyValidator`: API key validation and tier mapping
- `RateLimitMiddleware`: FastAPI middleware

**License Tiers**:
- **Free**: 100 requests/hour
- **Pro**: 1000 requests/hour
- **Academic**: Unlimited (logged)
- **Commercial**: Unlimited (logged)

**Features**:
- Per-API-key rate limiting
- Tier-based limits
- Request logging for unlimited tiers

---

### 6. Spectral Inference Server (`spectral_server.py`)

**Class**: `SpectralInferenceServer`

**Endpoints**:

#### `POST /embed`
Embed a raw spectrum into vector space.

**Request**:
```json
{
  "mz_values": [100.0, 200.0, ...],
  "intensity_values": [0.5, 0.8, ...],
  "precursor_mz": 500.0,
  "retention_time": 120.5,
  "spectrum_id": "spec_123"
}
```

**Response**:
```json
{
  "embedding": [0.1, 0.2, ...],
  "spectrum_id": "spec_123",
  "latency_ms": 150.5,
  "embedding_dim": 768
}
```

**Performance Target**: < 200ms latency

---

#### `POST /search`
Search for similar spectra by embedding.

**Request**:
```json
{
  "embedding": [0.1, 0.2, ...],
  "top_k": 10,
  "filter_conditions": {"compound_id": "C123"}
}
```

**Response**:
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

**Performance Target**: < 10ms latency

---

#### `POST /nearest`
Get nearest neighbors for a spectrum ID or embedding.

**Request**:
```json
{
  "spectrum_id": "spec_123",
  "top_k": 10
}
```

**Response**: Same as `/search`

---

#### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "embedding_dim": 768
}
```

---

#### `GET /model/info`
Model information endpoint.

**Response**:
```json
{
  "embedding_dim": 768,
  "device": "cuda:0",
  "collection_name": "nexa_spectra"
}
```

---

#### `POST /predict/{property}`
Predict RT/IM/MW using property heads. Returns scalar prediction and metadata.

---

#### `POST /decode/structure`
Return SMILES/InChI plus confidence using structure decoder. `/decode/smiles` and `/decode/inchi` provide focused representations.

---

#### `POST /feedback/{positive|negative}`
Record relevance feedback. `/feedback/stats` exposes aggregate counts for reranker fine-tuning.

---

#### `GET /metrics`
Prometheus endpoint (enabled when `prometheus_client` installed). Surfaces latency histograms, request counters, GPU utilization gauges.

---

#### `GET /graphql`
Optional GraphQL endpoint powered by `graphene` for flexible field selection.

---

#### `WEBSOCKET /stream/batch`
Bi-directional progress updates for long-running batch jobs (ingestion + export). Emits `{progress, message}` payloads.

---

#### `POST /export`
Export embeddings to JSON schema or MGF via `exporters.py`. Accepts inline embeddings or file references.

---

#### `POST /snapshot`
Create local Qdrant snapshot for offline search using `snapshot.py`.

---

## 🚀 Usage

### CLI Commands

#### Serve Spectral Model
```bash
nexa-inference serve-spectral \
  --checkpoint path/to/checkpoint.pt \
  --config path/to/config.yaml \
  --qdrant-url http://localhost:6333 \
  --collection-name nexa_spectra \
  --embedding-dim 768 \
  --port 8000
```

#### Serve Generic Model (Legacy)
```bash
nexa-inference serve \
  --checkpoint path/to/checkpoint.pt \
  --config path/to/config.yaml \
  --port 8000
```

#### Batch Processing / Indexing
```bash
nexa-inference process-batch \
  --input-dir data/shards \
  --output-dir embeddings \
  --model-path model.pt \
  --qdrant-url http://localhost:6333 \
  --collection-name nexa_spectra \
  --batch-size 64
```

### Python API

#### Generate Embeddings
```python
from nexa_inference import (
    SpectrumPreprocessor,
    EmbeddingEngine,
    EmbeddingStorage
)

# Initialize components
preprocessor = SpectrumPreprocessor()
engine = EmbeddingEngine(checkpoint_path=Path("model.pt"))
storage = EmbeddingStorage(output_dir=Path("./embeddings"))

# Process spectrum
mz_values = np.array([100.0, 200.0, 300.0])
intensity_values = np.array([0.5, 0.8, 0.3])
spectrum_tensor = preprocessor.preprocess(mz_values, intensity_values)

# Generate embedding
embedding = engine.embed(spectrum_tensor)

# Save embedding
storage.save_embeddings(
    embeddings=np.array([embedding.numpy()]),
    spectrum_ids=["spec_123"],
    metadata=[{"precursor_mz": 500.0}]
)
```

#### Upload to Vector DB
```python
from nexa_inference import VectorDBClient
import numpy as np

client = VectorDBClient(
    url="http://localhost:6333",
    collection_name="nexa_spectra"
)

embeddings = np.array([[0.1, 0.2, ...], ...])
spectrum_ids = ["spec_1", "spec_2", ...]
metadata = [{"precursor_mz": 500.0}, ...]

client.upload_embeddings(embeddings, spectrum_ids, metadata)
```

#### Search Similar Spectra
```python
query_embedding = np.array([0.1, 0.2, ...])
results = client.search(query_embedding, top_k=10)
```

---

## 🔒 Security & Access Control

### API Key Authentication
- API keys passed via `X-API-Key` header
- Tier mapping configured at server startup
- Invalid keys default to Free tier

### Rate Limiting
- Enforced per API key
- Tier-based limits (see License Tiers above)
- Returns `429 Too Many Requests` when exceeded

### Example Request
```bash
curl -X POST http://localhost:8000/embed \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "mz_values": [100.0, 200.0],
    "intensity_values": [0.5, 0.8]
  }'
```

---

## 📊 Performance Targets

| Component         | Target             | Status |
|------------------|--------------------|--------|
| Embedding latency| < 200ms / sample   | ✅     |
| Search latency   | < 10ms / query     | ✅     |
| Full round trip  | < 300ms per call   | ✅     |
| DB scale         | 1M+ spectra vectors| ✅     |

---

## 🧩 Dependencies

### Required
- `torch>=2.1.0`: PyTorch for model inference
- `fastapi>=0.115`: Web framework
- `uvicorn>=0.30`: ASGI server
- `numpy>=1.26`: Numerical operations
- `scipy>=1.11.0`: Scientific computing (entropy)

### Optional
- `qdrant-client`: Vector database client
- `pyarrow>=14.0.0`: Arrow/Parquet support
- `pandas>=2.1`: Data manipulation

---

## 🔄 Integration Points

### Model Registry
- Integrates with `nexa_train.models.DEFAULT_MODEL_REGISTRY`
- Supports encoder-only Transformer architectures
- Extracts embeddings via CLS token or pooling

### Storage
- Compatible with `nexa_compute.core.artifacts` for checkpoint resolution
- Uses `nexa_compute.core.registry` for artifact references

### Data Pipeline
- Can be integrated with `nexa_data` loaders for batch processing
- Supports Arrow/HDF5 shard ingestion (via external loaders)

---

## 🧪 Testing

### Unit Tests
- Preprocessor: Entropy filtering, binning, padding
- Embedding engine: Model loading, CLS extraction
- Vector DB: Upload, search, nearest neighbors
- Storage: JSONL/Arrow save/load

### Integration Tests
- End-to-end pipeline: Preprocess → Embed → Store → Search
- API endpoints: Request/response validation
- Rate limiting: Tier-based limits

---

## 🚧 Future Enhancements

### Planned Features
1. **Batch Processing CLI**: Process Arrow/HDF5 shards in batch
2. **Property Heads**: RT/IM prediction endpoints
3. **Structure Decoder**: Molecular structure generation
4. **Reranking**: RT delta and metadata-based reranking
5. **Monitoring**: Prometheus metrics integration
6. **Distributed Inference**: Multi-GPU support via vLLM

### Extensibility
- Custom preprocessor configurations
- Pluggable vector database backends
- Custom embedding extraction strategies
- Postprocessing pipelines

---

## 📝 Notes

- Preprocessing augmentation is disabled in inference mode for reproducibility
- Embeddings are L2-normalized by default for cosine similarity
- Vector DB collection is auto-created if missing
- Rate limiting uses in-memory storage (consider Redis for production)
- API key validation is simple dictionary lookup (consider database-backed for production)

