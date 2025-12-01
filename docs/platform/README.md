# NexaCompute Platform Guide

## Introduction
NexaCompute is an integrated platform for machine learning at scale. It unifies data engineering, distributed training, and model serving into a single coherent system.

## Key Capabilities

### 1. Observability
- **Distributed Tracing**: Track requests from API to GPU worker using OpenTelemetry.
- **Metrics**: Real-time Prometheus metrics for training loss, GPU utilization, and API latency.
- **Alerting**: Slack/PagerDuty integration for critical failures.

### 2. Model Operations (MLOps)
- **Model Registry**: Version control for models, including architecture, hyperparameters, and lineage.
- **Model Monitoring**: Automated drift detection (data & concept drift) for deployed models.
- **A/B Testing**: Experimentation framework for safe rollouts.

### 3. Data Operations (DataOps)
- **Versioning**: Git-like content-addressable storage for datasets.
- **Feature Store**: Centralized management of ML features with point-in-time correctness.
- **Quality**: Automated schema validation and anomaly detection.

### 4. Workflow Orchestration
- **DAG Engine**: Declarative definition of complex dependencies.
- **Scheduler**: Robust job scheduling with resume capability.
- **Templates**: Pre-built workflows for common tasks (e.g., "Train -> Eval -> Deploy").

### 5. Cost Optimization
- **Spot Instances**: Automatic handling of preemptible instances with checkpoint recovery.
- **Auto-Scaling**: Dynamic cluster sizing based on queue depth.
- **Budgets**: Project-level cost tracking and alerts.

## Getting Started

### Installation
```bash
git clone https://github.com/nexa/nexa-compute.git
cd nexa-compute
uv sync
```

### Configuration
Copy `.env.example` to `.env` and set your API keys (WandB, HuggingFace, AWS/GCP).

### Running a Workflow
```python
from nexa_compute.orchestration.templates import training_workflow
from nexa_compute.orchestration.scheduler import get_scheduler

workflow = training_workflow(
    model_name="my-model",
    dataset_name="my-dataset",
    epochs=5
)
get_scheduler().trigger_workflow(workflow.name)
```

