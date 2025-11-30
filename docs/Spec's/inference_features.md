# Additional Features for Nexa-Spec Inference Pipeline

This document outlines suggested additional features to enhance the inference pipeline beyond the core functionality. **Status:** All features below are now scaffolded in code and wired into the FastAPI server/CLI.

---

## 🚀 High Priority Features

### 1. **Property Prediction Heads** ✅
Add endpoints for predicting molecular properties directly from embeddings.

**Endpoints:**
- `POST /predict/retention_time` - Predict retention time
- `POST /predict/ion_mobility` - Predict ion mobility
- `POST /predict/molecular_weight` - Predict molecular weight

**Implementation:**
```python
class PropertyPredictor:
    def __init__(self, model_path: Path):
        self.rt_head = load_property_head(model_path, "retention_time")
        self.im_head = load_property_head(model_path, "ion_mobility")
    
    def predict_rt(self, embedding: torch.Tensor) -> float:
        return self.rt_head(embedding)
```

**Use Cases:**
- Filter search results by predicted RT
- Validate compound identification
- Guide experimental design

---

### 2. **Structure Decoder Integration** ✅
Add molecular structure generation from embeddings.

**Endpoints:**
- `POST /decode/structure` - Generate molecular structure
- `POST /decode/smiles` - Generate SMILES string
- `POST /decode/inchi` - Generate InChI identifier

**Implementation:**
```python
class StructureDecoder:
    def __init__(self, decoder_model_path: Path):
        self.decoder = load_decoder(decoder_model_path)
    
    def decode_smiles(self, embedding: torch.Tensor) -> str:
        tokens = self.decoder.generate(embedding)
        return tokens_to_smiles(tokens)
```

**Use Cases:**
- De novo structure elucidation
- Compound identification validation
- Structure-based search refinement

---

### 3. **Reranking Pipeline** ✅
Implement multi-stage reranking with metadata and property filters.

**Features:**
- RT delta filtering (remove matches with large RT differences)
- Metadata-based filtering (instrument type, collision energy)
- Learned reranking model (fine-tuned on user feedback)

**Implementation:**
```python
class Reranker:
    def rerank(
        self,
        results: List[Dict],
        query_metadata: Dict,
        rt_tolerance: float = 0.5,
        metadata_filters: Optional[Dict] = None
    ) -> List[Dict]:
        filtered = self._filter_by_rt_delta(results, query_metadata, rt_tolerance)
        filtered = self._filter_by_metadata(filtered, metadata_filters)
        return self._learned_rerank(filtered, query_metadata)
```

---

### 4. **Batch Processing CLI** ✅
Enhanced CLI for processing large datasets.

**Commands:**
```bash
# Process directory of shards
nexa-inference process-batch \
  --input-dir data/shards \
  --output-dir embeddings \
  --model-path model.pt \
  --qdrant-url http://localhost:6333 \
  --collection-name nexa_spectra \
  --batch-size 64 \
  --num-workers 4

# Process single shard
nexa-inference process-shard \
  --shard data/shards/shard_001.parquet \
  --output embeddings/shard_001_embeddings.jsonl \
  --model-path model.pt

# Stream processing from stdin
cat spectra.jsonl | nexa-inference process-stream \
  --output embeddings/stream_embeddings.jsonl \
  --model-path model.pt
```

**Features:**
- Progress bars with ETA
- Resume capability (checkpointing)
- Parallel processing
- Memory-efficient streaming

---

### 5. **Monitoring & Observability** ✅
Add comprehensive monitoring for production deployments.

**Metrics:**
- Embedding latency (p50, p95, p99)
- Search latency (p50, p95, p99)
- Throughput (requests/second)
- Error rates by endpoint
- GPU utilization
- Memory usage

**Implementation:**
```python
from prometheus_client import Counter, Histogram, Gauge

embedding_latency = Histogram('embedding_latency_seconds', 'Embedding latency')
search_latency = Histogram('search_latency_seconds', 'Search latency')
request_count = Counter('requests_total', 'Total requests', ['endpoint', 'status'])
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization')
```

**Dashboards:**
- Grafana dashboards for real-time monitoring
- Alerting rules for SLA violations
- Cost tracking per API key tier

---

### 6. **Caching Layer** ✅
Implement intelligent caching for frequently accessed embeddings.

**Features:**
- LRU cache for recent embeddings
- Redis-backed distributed cache
- Cache warming for popular spectra
- Cache invalidation strategies

**Implementation:**
```python
class EmbeddingCache:
    def __init__(self, redis_client: Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def get(self, spectrum_id: str) -> Optional[np.ndarray]:
        cached = self.redis.get(f"embedding:{spectrum_id}")
        return np.frombuffer(cached) if cached else None
    
    def set(self, spectrum_id: str, embedding: np.ndarray):
        self.redis.setex(
            f"embedding:{spectrum_id}",
            self.ttl,
            embedding.tobytes()
        )
```

---

## 🔧 Medium Priority Features

### 7. **Multi-Model Ensemble** ✅
Support ensemble predictions from multiple model checkpoints.

**Use Cases:**
- Improved accuracy through model averaging
- A/B testing new model versions
- Confidence scoring via ensemble variance

**Implementation:**
```python
class EnsembleEmbedder:
    def __init__(self, model_paths: List[Path]):
        self.models = [load_nexa_model(p) for p in model_paths]
    
    def embed(self, spectrum: Dict) -> np.ndarray:
        embeddings = [model.embed(spectrum) for model in self.models]
        return np.mean(embeddings, axis=0)
```

---

### 8. **Federated Search** ✅
Enable search across multiple vector database instances.

**Features:**
- Query multiple Qdrant collections
- Merge and rerank results
- Geographic distribution support

**Implementation:**
```python
class FederatedSearch:
    def __init__(self, clients: List[VectorDBClient]):
        self.clients = clients
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Dict]:
        all_results = []
        for client in self.clients:
            results = client.search(query, top_k=k)
            all_results.extend(results)
        return self._merge_and_rerank(all_results, k)
```

---

### 9. **Feedback Loop Integration** ✅
Collect user feedback for model improvement.

**Endpoints:**
- `POST /feedback/positive` - Mark result as correct
- `POST /feedback/negative` - Mark result as incorrect
- `GET /feedback/stats` - Get feedback statistics

**Implementation:**
```python
class FeedbackCollector:
    def record_feedback(
        self,
        query_id: str,
        result_id: str,
        is_correct: bool,
        user_id: Optional[str] = None
    ):
        feedback = {
            "query_id": query_id,
            "result_id": result_id,
            "is_correct": is_correct,
            "user_id": user_id,
            "timestamp": datetime.utcnow()
        }
        self.storage.save(feedback)
```

**Use Cases:**
- Fine-tune reranking models
- Identify problematic query patterns
- Improve search quality over time

---

### 10. **Query Expansion** ✅
Enhance search with query expansion techniques.

**Features:**
- Generate multiple query variants
- Search with each variant
- Merge results with weighted scoring

**Implementation:**
```python
class QueryExpander:
    def expand(self, query_embedding: np.ndarray) -> List[np.ndarray]:
        # Add noise for diversity
        variants = [
            query_embedding + np.random.normal(0, 0.1, size=query_embedding.shape)
            for _ in range(3)
        ]
        return [query_embedding] + variants
```

---

## 🎯 Low Priority Features

### 11. **Local Snapshotting** ✅
Enable offline search with local vector database snapshots.

**Features:**
- Download collection snapshots
- Local Qdrant instance for offline search
- Periodic sync with main database

---

### 12. **GraphQL API** ✅
Add GraphQL endpoint for flexible querying.

**Benefits:**
- Client-side field selection
- Reduced over-fetching
- Complex nested queries

---

### 13. **WebSocket Streaming** ✅
Real-time streaming for long-running batch jobs.

**Features:**
- Progress updates via WebSocket
- Streaming results as they're generated
- Cancellation support

---

### 14. **Multi-Tenancy** ✅
Support multiple organizations with isolated data.

**Features:**
- Organization-scoped collections
- Per-organization rate limits
- Data isolation guarantees

---

### 15. **Export Formats** ✅
Support additional export formats.

**Formats:**
- MGF (Mascot Generic Format)
- mzML
- JSON Schema validation
- OpenAPI/Swagger specs

---

## 📊 Implementation Priority Matrix

| Feature | Priority | Effort | Impact | Dependencies |
|---------|----------|--------|--------|--------------|
| Property Prediction | High | Medium | High | Trained property heads |
| Structure Decoder | High | High | High | Decoder model |
| Reranking Pipeline | High | Medium | High | None |
| Batch Processing CLI | High | Low | High | None |
| Monitoring | High | Medium | Medium | Prometheus/Grafana |
| Caching Layer | Medium | Low | Medium | Redis |
| Multi-Model Ensemble | Medium | Low | Medium | Multiple checkpoints |
| Federated Search | Medium | Medium | Low | Multiple DB instances |
| Feedback Loop | Medium | Medium | Medium | Storage backend |
| Query Expansion | Medium | Low | Low | None |
| Local Snapshotting | Low | Medium | Low | Local Qdrant |
| GraphQL API | Low | High | Low | GraphQL library |
| WebSocket Streaming | Low | Medium | Low | WebSocket support |
| Multi-Tenancy | Low | High | Medium | Auth system |
| Export Formats | Low | Low | Low | Format libraries |

---

## 🎓 Recommended Implementation Order

1. **Batch Processing CLI** (Quick win, high impact)
2. **Property Prediction Heads** (Core functionality extension)
3. **Reranking Pipeline** (Improves search quality)
4. **Monitoring & Observability** (Production readiness)
5. **Caching Layer** (Performance optimization)
6. **Structure Decoder** (Advanced feature)
7. **Feedback Loop** (Long-term improvement)
8. **Remaining features** (As needed)

---

## 💡 Additional Considerations

### Performance Optimizations
- **TensorRT Integration**: Convert PyTorch models to TensorRT for faster inference
- **ONNX Runtime**: Cross-platform inference optimization
- **Quantization**: INT8 quantization for 2-4x speedup
- **Batching Optimization**: Dynamic batching for variable-length inputs

### Security Enhancements
- **Input Validation**: Strict schema validation for all inputs
- **Rate Limiting**: Per-endpoint rate limits
- **Audit Logging**: Comprehensive audit trail
- **Encryption**: Encrypt embeddings at rest

### Developer Experience
- **SDK**: Python SDK for easy integration
- **Examples**: Comprehensive example notebooks
- **Documentation**: API documentation with examples
- **Testing**: Comprehensive test suite

