# Compute Plans

This directory stores **Compute Plans**: specific run configurations, hardware allocations, and execution strategies for different projects.

Unlike general infrastructure templates (in `nexa_infra/configs`), these plans are project-specific and document the *intent* and *design* of a specific experimental campaign.

## Structure

- **Plan Name**: `project_variant_phase.yaml` (e.g., `molecular_v1_stability.yaml`)
- **Contents**:
  - `run`: Project metadata and wandb tagging
  - `cluster`: Node count, GPU type, and interconnect
  - `training`: Hyperparameters and strategy (FSDP/DDP)
  - `data`: Dataset URI and specific version hashes

## Usage

To execute a plan:
```bash
nexa train --plan docs/compute_plans/my_plan.yaml
```

