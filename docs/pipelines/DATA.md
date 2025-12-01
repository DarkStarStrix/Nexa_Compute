# NexaData Engineering Architecture

> **Scope**: Data Ingestion, Validation, Synthetic Generation, and Distillation.
> **Modules**: `nexa_data`

NexaData is the production-grade data refinery for NexaCompute. It operates upstream of training, ensuring that all datasets are structurally valid, semantically consistent, and reproducible before they reach the compute cluster.

The system consists of three specialized pipelines:
1.  **Scientific Data Refinery (MS/MS)**: High-performance ingestion of mass spectrometry data.
2.  **Tool Use Protocol**: Synthetic data generation for agentic reasoning.
3.  **Distillation Pipeline**: Teacher-student knowledge transfer.

---

# 1. Scientific Data Refinery (MS/MS)

> **Codename: Atheron Pipeline**
> **Location**: `nexa_data/msms/`

This pipeline handles the massive-scale processing of molecular MS/MS datasets (100GB–500GB+). It guarantees that no samples enter training unless they are structurally and semantically validated.

## 1.1 Architecture
The pipeline uses a "Rust-Sandwich" architecture: Python handles I/O and orchestration (Dask), while Rust handles the compute-intensive validation and transformation loops.

```text
[Raw HDF5] 
    ↓ 
[HDF5 Reader (Dask)] 
    ↓ 
[Rust Batch Processor] <--- (SIMD Validation, Peak Picking, Normalization)
    ↓ 
[Duplicate Watchdog] 
    ↓ 
[Shard Writer (Arrow)] 
    ↓ 
[Shard Sentinel] 
    ↓ 
[Final Parquet Shards]
```

## 1.2 Core Components

### Rust Extension (`nexa_data/msms/rust_ext`)
A compiled Python extension aimed at performance and safety:
*   **Zero-Copy Validation**: Validates floating-point constraints (NaN/Inf, negative m/z) without python object overhead.
*   **Memory Safety**: Enforces strict memory limits during batch processing to prevent OOM kills on shared nodes.
*   **Determinism**: Implements stable sort and fixed-precision rounding for reproducible outputs.

### Dask Processor (`nexa_data/msms/processor.py`)
Orchestrates parallel execution:
*   **Local Mode**: Runs on a single workstation with limited workers.
*   **Cloud Mode**: Scales to 48-core nodes, managing worker saturation and spill-to-disk.

### Reliability Features
1.  **Golden Sample Suite**: A regression test suite that runs before every large job. It transforms a set of known inputs and compares the bit-exact output against a "golden" reference.
2.  **Shard Sentinel**: An atomic commit protocol for data shards.
    *   Step 1: Write `shard_X.parquet.tmp`
    *   Step 2: Re-open and validate schema/rows.
    *   Step 3: Rename to `shard_X.parquet`.
3.  **Semantic Checker**: Tracks rolling statistics (peak counts, charge distribution) to detect dataset drift.

## 1.3 Execution and Testing

**Dry Run (Doctor Mode)**:
Checks configuration validity and environment health.
```bash
nexa-data doctor
```

**Build Shards**:
Executes the full pipeline.
```bash
python -m nexa_data.msms.cli --config configs/msms_v1.yaml build-shards
```

**Testing Guide**:
Refer to the interactive menu for all configuration options:
```bash
python -m nexa_data.msms.cli --interactive
```

## 1.4 Memory Allocation Strategy

To prevent worker pausing and optimize throughput, dynamic memory allocation is used:

```text
System Reserve = Total RAM × System Reserve % (20-25%)
Worker Pool = Total RAM - System Reserve
Memory per Worker = (Worker Pool / Num Workers) × Target Utilization %
```

This strategy ensures:
*   **No Pausing**: Workers stay below the 80% pause threshold.
*   **Max Throughput**: Optimal memory usage without waste.
*   **Stability**: Sufficient reserve for OS processes.

**Overrides**:
```bash
export NEXA_MSMS_MEMORY_LIMIT="4GB"
```

## 1.5 Quality Guarantees

Every run automatically generates a quality report (`quality_report.json`) verifying:
*   **Canonicalization**: Integrity errors and attrition rates.
*   **Shard Construction**: Checksums, duplicates, and schema validity.
*   **Training Readiness**: Spot checks (100 samples/shard) for NaN/Inf and tensor conversion.

## 1.6 Technical Specification

### Arrow Schema
The pipeline enforces a strict Arrow schema version 1.0.

```python
import pyarrow as pa

SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("mzs", pa.list_(pa.float32())),
    ("ints", pa.list_(pa.float32())),
    ("precursor_mz", pa.float32()),
    ("charge", pa.int8()),
    ("adduct", pa.string()),
    ("instrument_type", pa.string()),
    ("collision_energy", pa.float32()),
    ("smiles", pa.string()),
    ("inchikey", pa.string()),
    ("formula", pa.string()),
])
```

### Quality Tolerances (v1)

| Stage | Metric | Tolerance | Action |
| :--- | :--- | :--- | :--- |
| **1. Canonicalization** | Integrity Errors | ≤ 0.01% | Log & Skip |
| | Attrition Rate | 1-10% | Log |
| **2. Shard Construction** | Missing Samples | 0% | Hard Fail |
| | Duplicates | 0% | Hard Fail |
| | Checksum Mismatch | 0 | Hard Fail |
| **3. Training Readiness** | NaN/Inf in Tensors | 0% | Warn/Fail |
| | Batch Load Failures | 0 | Hard Fail |

---

# 2. Tool Use Protocol (Synthetic Data)

> **Location**: `nexa_data/tool_protocol/`

This pipeline generates high-quality, multi-turn conversation episodes to train agents in scientific tool usage and reasoning.

## 2.1 Episode Generators (`generate_episodes.py`)

The system constructs "Episodes"—self-contained interaction histories between a User, an Assistant, and a Tool Environment.

### Supported Scenarios
1.  **Physical Units**:
    *   **Tool**: `units.convert` (Pint-based).
    *   **Objective**: Train the model to delegate unit conversions to a calculator rather than hallucinating values.
2.  **Literature Search**:
    *   **Tools**: `papers.search`, `papers.fetch`.
    *   **Objective**: Train the model to formulate search queries, select relevant results, and retrieve full text (simulated Crossref).
3.  **Simulation**:
    *   **Tool**: `python.run` (Sandbox).
    *   **Objective**: Train the model to write, debug, and execute Python code to model physical systems (e.g., battery capacity vs. temperature).

## 2.2 Error Recovery (Repair Training)
To build robust agents, the pipeline deliberately injects errors into the training data:
*   **Malformed JSON**: The "User" (or Environment) mocks an invalid tool call format. The "Assistant" must detect the error and retry with valid JSON.
*   **Runtime Errors**: The Python sandbox returns a `TypeError` or `IndexError`. The "Assistant" must analyze the traceback and emit corrected code.
*   **Missing Citations**: The Assistant cites a paper it hasn't fetched. The "Environment" warns it, prompting a `papers.fetch` call.

## 2.3 Usage
```bash
# Generate 1000 episodes including 50 repair scenarios
python nexa_data/tool_protocol/generate_episodes.py \
  --project-slug scientific_assistant \
  --repair-count 50
```

---

# 3. Distillation Pipeline

> **Location**: `nexa_data/data_generation_job.py`

This pipeline manages the extraction of reasoning traces from teacher models (GPT-4o) to train smaller student models.

## 3.1 Workflow

1.  **Prompt Selection**: Loads a set of scientific questions/prompts (`teacher_inputs`).
2.  **Teacher Querying**:
    *   Uses `system_prompt_template.txt` to enforce the "Scientific Assistant" persona.
    *   Runs mostly in parallel (batch processing) to maximize throughput.
3.  **Quality Gating (SampleGate)**:
    *   A "Judge" model (or heuristic filter) scores the teacher's response.
    *   Responses below a quality threshold (e.g., score < 0.8) are rejected.
4.  **Materialization**:
    *   Converts valid traces into the standard SFT (Supervised Fine-Tuning) format: `{"messages": [...]}`.
    *   Applies augmentations via `nexa_data/augment.py`.

## 3.2 Execution
```bash
# Start the teacher collection job
python nexa_data/data_generation_job.py
```

---

# 4. Shared Infrastructure

> **Location**: `nexa_data/` root

### Data Preparation (`prepare.py`)
The standard entry point for finalizing a dataset version.
*   **Metadata**: Generates `dataset_metadata.json` with lineage info.
*   **Hashing**: Computes a merkle-tree-like hash of all shards to ensure immutable dataset identity.
*   **Artifacts**: Packages the data into a versioned artifact for the training cluster.

### Filters (`nexa_data/filters/`)
Reusable filter logic shared across pipelines:
*   **Label Filter**: Removes samples with blocklisted labels.
*   **Length Filter**: Drops samples exceeding context window limits.

---

# 5. Data Analysis

This directory contains Jupyter notebooks and utilities for analyzing and exploring NexaCompute datasets.

## Notebooks

- **`distill_data_overview.ipynb`** - Comprehensive analysis of distillation datasets:
  - Raw data file inspection and statistics
  - Enhanced prompt exemplar curation
  - Teacher input generation for distillation
  - Dataset capacity and distribution analysis
  - **Outputs**: Organized processed data in `data/processed/distillation/`

## Utilities

- **`query_data.py`** - Query interface for processed datasets:
  ```python
  from nexa_data.data_analysis.query_data import DataQuery
  query = DataQuery()
  df = query.get_teacher_inputs(version="v1")
  ```

## Usage

Notebooks should be run sequentially after ensuring the virtual environment has required dependencies:
- `pandas`
- `pyarrow`
- `matplotlib`
- `seaborn`
- `datasets` (HuggingFace)
- `zstandard` (for compressed JSONL files)

## Outputs

All outputs are organized in `data/processed/`:
- **Distillation**: `data/processed/distillation/teacher_inputs/` - Teacher request data
- **Manifests**: `data/processed/distillation/manifests/` - Dataset manifests
- **Templates**: `data/system_prompt_template.txt` - System prompt template

## Data Organization

- **Raw Data**: `data/raw/` (see `data/raw/README.md` and `DATA_INDEX.md`)
- **Processed Data**: `data/processed/` (organized by purpose, see `data/processed/README.md`)
- **Query Interface**: Use `query_data.py` for reliable data access

## Notes

- All file paths are relative to the project root
- Large datasets are loaded incrementally to avoid memory issues
- Output artifacts follow the NexaCompute storage policy
- Use versioned file names (e.g., `teacher_inputs_v1.parquet`)

---

# Appendix: Optimization & Performance

## Performance Benchmarks
*   **Baseline (No Dask)**: ~0.1-1 samples/sec
*   **Current (Dask)**: ~100-1000+ samples/sec (100x improvement)
*   **Rust Validator**: Additional 10-20% speedup, lower CPU usage.

## Completed Optimizations
*   **Real-Time Progress**: Unbuffered `tqdm` updates.
*   **Dask Integration**: Parallel chunked reading for HDF5.
*   **Rust Validation**: Fast finite checks and shape consistency.
*   **Memory Management**: 60% target utilization, periodic GC, and fragmentation accounting (0.85 factor).

## Testing Commands
*   **TUI Mode**: `python -m nexa_data.msms.cli --interactive`
*   **CLI Simple**: `python -m nexa_data.msms.cli build-shards ... --observability simple`
*   **CLI Granular**: `python -m nexa_data.msms.cli build-shards ... --observability granular --dashboard-port 8080`
