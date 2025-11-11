# Naming Conventions

Consistent naming keeps large multi-project deployments manageable.

## Project Slugs

- Format: lowercase, alphanumeric, underscores (`^[a-z0-9_]+$`)
- Must be unique across the repository
- Used in directory names, config paths, and artifact IDs

## Run Identifiers

```
{project_slug}_{run_type}_{timestamp}
```

- `run_type` should describe the workflow (`pretrain`, `sft`, `rlhf`, `eval`, `serve`)
- `timestamp` format: `YYYYMMDD_HHMMSS`
- Example: `scientific_assistant_toolproto_20251111_153000`

## Datasets

```
data/processed/{project_slug}/{domain}/<name>_v<version>.<ext>
```

- Example: `data/processed/scientific_assistant/tool_protocol/sft_toolproto_v1_train.jsonl`
- Use semantic versions (`v1`, `v2`) not timestamps

## Artifacts

- Checkpoints: `artifacts/{project_slug}/checkpoints/{run_id}/`
- Evaluations: `artifacts/{project_slug}/eval/{run_id}/`
- Runs: `artifacts/{project_slug}/runs/{run_id}/`

## Logging

- TensorBoard, WandB, and custom logs: `logs/{project_slug}/<component>/`

## Configuration Files

- Stored under `projects/{project_slug}/configs/`
- Naming pattern: `{purpose}.yaml` (e.g., `toolproto_qlora.yaml`, `eval_baseline.yaml`)

## Documentation

- Project specs: `projects/{project_slug}/docs/SPEC.md` (or similar)
- Post-mortems: `projects/{project_slug}/docs/postmortems/<incident>.md`

Adhering to these conventions allows guardrails to detect misplacements automatically.

