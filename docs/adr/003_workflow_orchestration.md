# ADR-003: Workflow Orchestration with Declarative DAGs

## Status
Accepted

## Context
Users need to define complex ML pipelines (Train -> Eval -> Deploy) with dependencies. Hardcoding these sequences in Python scripts is brittle and hard to visualize/monitor.

## Decision
We will build a lightweight **Declarative DAG Engine** using Pydantic and a custom Scheduler.

### Rationale
1. **Flexibility**: Workflows can be defined in YAML/JSON or Python code.
2. **Portability**: The DAG definition is decoupled from the execution environment (Local vs. Slurm vs. K8s).
3. **Control**: Building our own engine allows tight integration with our specific resource management and retry logic.

## Implementation
- **Definition**: `WorkflowDefinition` dataclass holding a list of `PipelineStep`.
- **Execution**: `WorkflowScheduler` manages state transitions and step execution.
- **Storage**: In-memory state for now; persistent DB backend planned.

## Consequences
- **Positive**: Clear visualization of dependencies, resume capability, separation of concerns.
- **Negative**: Maintenance burden of a custom scheduler compared to Airflow/Prefect (but significantly lower overhead).

