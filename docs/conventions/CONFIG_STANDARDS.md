# Configuration Standards

Project configs must be self-describing and enforce correct path scoping.

## Required Sections

```yaml
project:
  slug: scientific_assistant
  name: Scientific Assistant

experiment:
  name: toolproto_qlora_v1
  output_dir: artifacts/scientific_assistant/runs
  tags:
    owner: tooling
    project: scientific_assistant
```

- `project.slug` must match a registered project
- `experiment.output_dir` must point inside `artifacts/{project_slug}/`
- `experiment.tags.project` mirrors `project.slug`

## Data Configuration

```yaml
data:
  train_dataset: data/processed/scientific_assistant/tool_protocol/sft_toolproto_v1_train.jsonl
  val_dataset: data/processed/scientific_assistant/tool_protocol/sft_toolproto_v1_validation.jsonl
  format: chat_messages
```

- All datasets live under `data/processed/{project_slug}/`
- Use descriptive subdirectories (`distillation`, `tool_protocol`, `training`, `evaluation`)

## Logging and Artifacts

```yaml
monitoring:
  tensorboard:
    log_dir: logs/scientific_assistant/toolproto
```

- Logs point to `logs/{project_slug}/`
- Checkpoint directories must resolve inside `artifacts/{project_slug}/checkpoints`

## Validation Rules

The config loader will enforce:

1. `project.slug` exists in the project registry
2. All paths begin with the expected project namespace
3. Output directories are writable and created if absent
4. Tags include `project`

Configs failing validation should immediately raise an error.

