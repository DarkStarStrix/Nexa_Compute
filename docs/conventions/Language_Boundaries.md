# Language Boundaries

*Defines strict boundaries between Python, Rust, Go, and Bash per Scaling Policy Section 4.*

## Python — Orchestration Layer

### Responsibilities
- Control flow and workflow orchestration
- CLI/TUI interfaces
- Configuration loading and validation
- Pipeline wiring and coordination
- API server logic (FastAPI)
- High-level glue code
- Manifest management
- Registry patterns

### Forbidden
- ❌ Heavy data transforms
- ❌ CPU-bound loops
- ❌ High-volume data processing
- ❌ Memory-intensive operations

### Delegation Pattern
When Python needs to perform compute-heavy operations, it must delegate to Rust:

```python
# ✅ Correct: Python orchestrates, Rust computes
from nexa_compute.data.rust_core import rust_core
result = rust_core.shuffle_and_split(num_items=1000000, weights=[0.8, 0.2], seed=42)

# ❌ Wrong: Python doing heavy compute
indices = list(range(1000000))
random.shuffle(indices)  # Too slow for large datasets
```

## Rust — Kernel Layer

### Responsibilities
- Deterministic data transforms
- High-volume CPU-bound operations
- Packing, gating, shuffling sequences
- Statistical computations
- Scientific preprocessing
- Memory-sensitive or parallel work
- Content-addressable storage operations

### Location
Rust modules live in `rust/`:
- `rust/nexa_data_core/` - Data preprocessing
- `rust/nexa_data_quality/` - Quality filtering
- `rust/nexa_stats/` - Statistical operations
- `rust/nexa_train_pack/` - Sequence packing

### FFI Boundaries
Rust modules expose clean Python bindings via:
- `src/nexa_compute/data/rust_core.py` - Wraps `nexa_data_core`
- `src/nexa_compute/data/rust_quality.py` - Wraps `nexa_data_quality`
- `src/nexa_compute/data/stats_core.py` - Wraps `nexa_stats`
- `src/nexa_compute/data/pack_core.py` - Wraps `nexa_train_pack`

### Forbidden
- ❌ Orchestration logic
- ❌ High-level business logic
- ❌ Dependencies on Python modules (except for FFI)

## Go — Tooling & UX Layer

### Current Status
No Go code exists in the repository. If added in the future:

### Responsibilities (if used)
- TUIs (terminal user interfaces)
- Small binaries
- Admin tools
- Long-running lightweight services
- Operator utilities

### Forbidden
- ❌ ML logic
- ❌ Critical data transforms
- ❌ Core platform logic

## Bash — Environment & Process Layer

### Responsibilities
- Environment setup
- tmux wrappers
- Trivial glue commands
- Process coordination
- Deployment scripts

### Location
Bash scripts in `nexa_infra/scripts/`:
- `provision/deploy.sh` - Node deployment
- `orchestration/start_forge.sh` - Service startup
- `bootstrap_*.sh` - Environment bootstrap

### Forbidden
- ❌ Core logic
- ❌ Branching/computation
- ❌ Complex control flow
- ❌ Data processing

### Example
```bash
# ✅ Correct: Minimal wrapper
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/nexa_compute
python nexa_train/train.py --config-mode v1

# ❌ Wrong: Complex logic in bash
if [ $(wc -l < data.txt) -gt 1000 ]; then
    # Complex processing logic belongs in Python/Rust
fi
```

## Boundary Violations to Watch For

### Python Doing Heavy Compute
**Sign**: Python code with nested loops processing large datasets
**Fix**: Move to Rust module

### Rust Doing Orchestration
**Sign**: Rust code making high-level decisions or managing workflows
**Fix**: Keep Rust focused on compute, orchestrate from Python

### Bash Doing Logic
**Sign**: Bash scripts with complex conditionals or data processing
**Fix**: Move to Python script or Rust binary

## Verification

Run language boundary checks:
```bash
# Check for Python CPU-bound loops
grep -r "for.*in.*range.*:" src/nexa_compute --include="*.py" | grep -v "test"

# Verify Rust modules are compute-focused
grep -r "orchestrat\|workflow\|pipeline" rust/ --include="*.rs"
```

## Examples

### ✅ Correct: Python Orchestrates, Rust Computes

**Python** (`src/nexa_compute/data/pipeline.py`):
```python
def shuffle_dataset(self, seed: int) -> List[int]:
    """Orchestrate shuffling via Rust."""
    return rust_core.shuffle_and_split(
        num_items=len(self.dataset),
        weights=[1.0],
        seed=seed
    )[0]
```

**Rust** (`rust/nexa_data_core/src/lib.rs`):
```rust
#[pyfunction]
pub fn shuffle_and_split(
    num_items: usize,
    weights: Vec<f64>,
    seed: u64
) -> Vec<Vec<usize>> {
    // Heavy compute happens here
    // Deterministic, parallel, memory-efficient
}
```

### ❌ Wrong: Python Doing Heavy Compute

```python
def shuffle_dataset(self, seed: int) -> List[int]:
    """Wrong: Python doing heavy compute."""
    indices = list(range(len(self.dataset)))
    random.shuffle(indices)  # Too slow for large datasets
    return indices
```

