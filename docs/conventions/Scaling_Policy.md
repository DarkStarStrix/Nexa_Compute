```markdown
# NexaCompute Scaling Policy
*A governance document to prevent entropy, sprawl, and technical debt as NexaCompute evolves beyond V4.*

---

## 1. Purpose
This policy defines how NexaCompute scales **safely and coherently**, ensuring:

- strict module specialization  
- predictable multi-language interaction  
- reproducible workflows through manifests  
- controlled growth (anti-sprawl)  
- high-quality logging and observability  
- test-driven completeness  
- architectural clarity  

This governs **how the system evolves**, not just how code is written.

---

## 2. Monorepo Philosophy
NexaCompute is a **modular monorepo**:

- One repository  
- Multiple specialized modules  
- Each module has **one clear responsibility**  
- Modules interact through **explicit, versioned contracts**  
- Reproducibility and automation are first-class values  

The repo is treated as a **directed acyclic graph (DAG)**, not a code dump.

---

## 3. Module Specialization Standard
Every module must:

1. **Do one conceptual job extremely well**  
2. Have **clear inputs** (manifests, configs, datasets)  
3. Have **clear outputs** (manifests, artifacts, logs)  
4. Declare **dependencies explicitly**  
5. Avoid leaking responsibilities into other modules

### Forbidden
- Generic `utils/` or “common” catch-all folders  
- Multi-purpose modules  
- Mixed responsibilities within a single module  

### Required
A **one-sentence responsibility statement** per module.

---

## 4. Language Boundaries

### Python — Orchestration Layer
Python is responsible for:
- control flow  
- CLI/TUI  
- configs & manifests  
- pipeline wiring  
- API server logic  
- high-level glue  

Python **must not** implement heavy data transforms or CPU-bound loops.

---

### Rust — Kernel Layer
Rust is responsible for:
- deterministic data transforms  
- high-volume CPU-bound operations  
- packing, gating, shuffling  
- statistical computations  
- scientific preprocessing  
- memory-sensitive or parallel work  

Rust must expose **clean FFI boundaries** and never depend on Python logic.

---

### Go — Tooling & UX Layer
Go is used for:
- TUIs  
- small binaries  
- admin tools  
- long-running lightweight services  

Go does not own ML logic or heavy compute.

---

### Bash — Environment & Process Layer
Bash is used only for:
- environment setup  
- tmux wrappers  
- trivial glue commands  

Bash must remain small and free of core logic.

---

## 5. Logging, Tracking, and Observability
NexaCompute rejects silent failures.

Every major workflow must produce:

- **structured logs**  
- **run manifests**  
- **dataset manifests**  
- **clear error traces**  

### Required
- Every action has a `run_id`.  
- Every dataset has a versioned manifest.  
- Every error is explicit and contextual.

### Forbidden
- silent failure  
- swallowed exceptions  
- ambiguous or partial errors  

A large, informative error is preferred over a vague one.

---

## 6. Anti-Sprawl Policy

### Module Admission Checklist
Before adding a new feature/module:

1. Does it clearly belong to an existing module?  
2. If not, can it live under `experiments/` temporarily?  
3. Can its purpose be stated in one sentence?  
4. Does it fit cleanly into the system DAG?  
5. Does it have defined manifests for inputs/outputs?  

If any answer is “no,” the feature waits.

---

### Monthly Pruning
Once per month:

- remove stale experiments  
- delete abandoned scripts  
- clean up outdated configs  
- consolidate one-off utilities  
- remove unused pathways  

The system must be actively kept lean.

---

## 7. Testing as Governance

### Required
Every module must:
- have unit tests  
- have integration tests  
- be represented in the test CLI  
- satisfy golden-path smoke tests  

### Golden Pipeline Test
A minimal pipeline:
```

data → distill → pack → train → eval

```
must execute deterministically and quickly.

If this fails, the repo is “red.”

---

## 8. Architectural Clarity Standard

You must be able to **mentally model the entire system**.

### Required
- stable module boundaries  
- versioned schemas  
- manifests everywhere  
- deterministic behavior  
- no global implicit state  

### Forbidden
- folder-based conventions without manifests  
- circular dependencies  
- implicit side effects  
- deep cross-language call stacks  

If you can’t *draw* the architecture, it’s too complex.

---

## 9. Idempotency Guarantee

Every important operation must be idempotent:

- Running twice should yield the same results  
- Rerunning after a failure should be safe  
- Manifests ensure reproducibility  

Idempotency is the foundation of large-scale reliability.

---

## 10. Stability Comes From Contracts, Not Code
The following must be **versioned and stable**:

- dataset schemas  
- run manifests  
- packer outputs  
- Rust ABI boundaries  
- training config schemas  
- evaluation rubrics  

Internal implementations can change rapidly.  
Interfaces must evolve slowly.

---

## 11. DAG Discipline

NexaCompute must remain a **directed acyclic graph**.

Core flow:
```

NexaData → NexaDistill → NexaPack → NexaTrain → NexaEval → NexaInference

```

Cross-cutting systems (monitoring, registry, manifests, infra) may depend downward but must never create cycles.

If a cycle appears → extract a shared library.

---

## 12. The Operator Principle

Your job:
- high-level decisions  
- architecture  
- scientific correctness  
- resource acquisition  
- running large experiments  

The monorepo’s job:
- automation  
- determinism  
- reproducibility  
- correctness  
- observability  

This separation must remain sharp.

---

## 13. The Three Laws

### **LAW 1 — Boundaries Before Features**
If a new feature weakens module boundaries, reject or refactor.

### **LAW 2 — Contracts Over Implementation**
Version contracts. Refactor implementations freely.

### **LAW 3 — Automation Before Repetition**
If you repeat it twice, automate it.  
If you repeat it three times, formalize it as a CLI command.

---

## Appendix: Health Indicators

NexaCompute is healthy when:

- new collaborators (or future you) can reason about the system quickly  
- automation handles all repeatable workflows  
- manifests reconstruct entire histories  
- Rust boundaries never drift  
- tests reflect the structure of the monorepo  
- the golden pipeline always passes  
- complexity grows predictably, not exponentially  

If this policy is followed, NexaCompute can scale to:

- 7B+ pretraining  
- molecular pipelines  
- multi-team development  
- HPC clusters  
- distributed inference  

…without architectural decay.

```
