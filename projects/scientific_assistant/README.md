# Scientific Assistant Project

This project encapsulates all assets related to the tool-using scientific assistant lifecycle.

## Contents

- `configs/` – Training and evaluation configuration files scoped to the project
- `pipelines/` – Pipeline definitions and job orchestration manifests
- `docs/` – Project-specific specifications, post-mortems, and notes
- `manifests/` – Machine-readable metadata describing datasets, artifacts, and dependencies
- `data/` – Located under `data/{raw,processed}/scientific_assistant/`
- `artifacts/` – Located under `artifacts/scientific_assistant/`

## Lifecycle Coverage

The scientific assistant currently supports:

- Supervised fine-tuning (SFT) with tool protocol datasets
- Data curation and distillation workflows
- Evaluation and serving harnesses for tool-augmented outputs

Upcoming work will expand coverage to pre-training, RL/RLHF, and mid-training telemetry.

