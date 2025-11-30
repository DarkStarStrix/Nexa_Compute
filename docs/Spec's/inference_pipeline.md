# Nexa-Spec Inference and Query System Design

This document outlines the full technical architecture for the Nexa-Spec inference and molecular query system, from embedding generation to real-time querying through the Atlas++ database.

---

## 🧠 Core Objective
Enable high-throughput, low-latency molecular inference from raw MS/MS spectra using a 7B parameter foundation model. The system supports real-time queries, batch processing, structure decoding, and large-scale vector database search.

---

## 🔧 Pipeline Overview

### 1. **Spectral Ingestion**
- Source: Arrow/HDF5 shards
- Format: Standardized spectra input, 2GB per shard
- Ingestion handled by a `spectra_loader` module with preprocessing

### 2. **Preprocessing & Tokenization**
- `SpectrumPreprocessor` applies:
  - Entropy filtering
  - m/z binning & padding
  - Augmentations (optional, disabled in inference)
- Output: Tensor batch `(B, N, D)` ready for model

### 3. **Inference Engine**
- Hosted 7B encoder-only Transformer (Nexa-Spec)
- Deployed via:
  - PyTorch Lightning for baseline
  - Optional vLLM or FasterTransformer for optimization
- Output: `[CLS]` token (512–1024 dim embedding)

### 4. **Embedding Storage**
- Format: JSONL or Arrow
- Fields:
  - `spectrum_id`, `compound_id`, `precursor_mz`, `retention_time`, `embedding`, etc.
- Optionally: L2-normalize and compress

### 5. **Vector DB Ingestion (Qdrant)**
- Embeddings uploaded to Atlas++ via batch writer
- Index: HNSW (cosine or inner product)
- Metadata stored with each vector (filterable)

### 6. **FastAPI Query Gateway**
- Endpoints:
  - `/embed`: embed raw spectrum
  - `/search`: return top-k matches
  - `/nearest`: nearest neighbors to spectrum ID or embedding
- Middleware routes based on license tier (free, pro, academic, commercial)

### 7. **Postprocessing Layer**
- Optional modules:
  - Property heads (e.g., RT, IM)
  - Structure decoder (transformer or graph model)
- Reranking possible with RT delta or metadata filters

### 8. **Access Control & Rate Limiting**
- API key enforcement middleware
- Rate limits for Free & Pro tiers
- Unlimited for Academic/Commercial (with logs)

---

## 🚀 Performance Targets
| Component         | Target             |
|------------------|--------------------|
| Embedding latency| < 200ms / sample   |
| Search latency   | < 10ms / query     |
| Full round trip  | < 300ms per call   |
| DB scale         | 1M+ spectra vectors|

---

## 🧩 Deployment Stack
- Inference Pods: PyTorch or vLLM + CUDA
- Vector DB: Qdrant (HNSW, sharded)
- Gateway: FastAPI + Gunicorn + NGINX
- Storage: Wasabi (HDF5, Arrow), local NVMe for speed
- Monitoring: Prometheus + Grafana
- Auth: API keys, license metadata

---

## 🧪 Future Extensions
- 🔁 Real-time structure generation
- 🧬 Spectral-to-molecule decoder
- 💡 Feedback loop for model retraining
- 🔍 Relevance scoring + learned reranking
- 📦 Local snapshotting + federated search

