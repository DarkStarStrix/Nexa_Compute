---
title: Pipeline Spec (v2)
slug: overview/pipeline-v2
description: Declarative pipeline format, execution semantics, and caching rules.
---

# NexaCompute Pipeline Specification (v2)

The v2 runtime executes pipelines declared as YAML documents under
`pipelines/`. Each pipeline is transformed into a DAG of steps that exchange
artifacts via the shared protocol defined in `src/nexa_compute/core/artifacts.py`.

## File Structure

```yaml
pipeline:
  name: general_e2e
  steps:
    - id: train_hf
      uses: runners.train
      backend: hf
      scheduler: local
      params:
        model: distilbert-base-uncased
        dataset: glue
      out: artifacts/checkpoints/hf_baseline
    - id: eval_hf
      uses: runners.eval
      in:
        config: nexa_train/configs/baseline.yaml
        checkpoint: artifacts/checkpoints/hf_baseline
      out: artifacts/eval/hf_baseline
```

### Step Fields

| Field      | Required | Description |
|------------|----------|-------------|
| `id`       | ✅       | Unique step identifier used for caching and dependency tracking |
| `uses`     | ✅       | Logical entry point. Built-ins include `runners.train`, `runners.eval`, `runners.serve`, `core.registry.register`, `core.registry.promote` |
| `backend`  | optional | Backend selection for `runners.train`/`runners.serve` (e.g. `hf`, `axolotl`, `vllm`) |
| `scheduler`| optional | Scheduler selection (`local`, `slurm`, `k8s`) |
| `in`       | optional | Mapping of input URIs (referencing previous outputs or config files) |
| `out`      | optional | Mapping of output locations. For artifacts this should be a local path |
| `params`   | optional | Arbitrary configuration passed verbatim to the handler |
| `after`    | optional | List of explicit dependencies (defaults to previous order) |

Variables and templating are intentionally minimal in v2 to keep pipelines
transparent. Path interpolation can be achieved with standard YAML anchors or
by constructing values in Python before invoking the CLI.

## Execution Semantics

1. **Parsing:** The CLI loads the YAML and creates `PipelineStep` objects.
2. **Graph Build:** Dependencies are resolved (implicit order + `after` clauses).
3. **Caching:** The cache key is the SHA-256 hash of `uses`, `inputs`, `outputs`,
   `backend`, `scheduler`, `params`, and optional `cache_hint`. If an existing
   output artifact contains a `COMPLETE` marker and the cache key matches, the
   step is marked `SKIPPED`.
4. **Execution:** Handlers are dispatched according to `uses`.
5. **Persistence:** Step status and cache keys are saved to
   `.nexa_state/<pipeline>/pipeline_state.json`.
6. **Resume:** Re-running `pipeline run` automatically resumes from incomplete
   steps; `pipeline resume` is an alias for convenience.
7. **Failure:** When a handler raises an exception the step is marked `FAILED`
   and downstream steps are blocked. The CLI surfaces `rerun --from <step_id>`
   guidance in the console output.

## Artifact Lifecycle

- **Creation:** Handlers use `create_artifact(destination, producer)` to write
  outputs atomically (`<destination>.tmp` + `meta.json` + `COMPLETE`).
- **Metadata:** `meta.json` captures `kind`, `uri`, `hash`, `bytes`, `created_at`,
  `inputs`, and `labels`.
- **Promotion:** Registry steps call `core.registry.promote` which validates the
  `COMPLETE` marker before updating tag pointers.

## Built-in Handlers

### `runners.train`
- Backends: `hf`, `axolotl`
- Scheduler: currently `local` (Slurm/K8s stubs reject with `NotImplementedError`)
- Produces checkpoint artifact at `out`

### `runners.eval`
- Inputs: `config` (YAML) and optional `checkpoint`
- Produces `eval_report` artifact at `out`

### `runners.serve`
- Backends: `vllm`, `hf_runtime`
- Parameters may include `host`, `port`, `tensor_parallel`, `dry_run`
- Tracks handles inside the CLI session; use `serve stop` to terminate

### Registry Helpers
- `core.registry.register` expects `params` to contain `name`, `uri`, and
  `meta` (path to JSON metadata)
- `core.registry.promote` expects `name`, `version`, `tag`

## CLI Commands

```bash
# Execute pipeline, building state under .nexa_state/<name>/
python -m nexa_compute.cli.orchestrate pipeline run pipelines/general_e2e.yaml

# Resume after an interruption
python -m nexa_compute.cli.orchestrate pipeline resume pipelines/general_e2e.yaml

# Visualise the graph structure
python -m nexa_compute.cli.orchestrate pipeline viz pipelines/general_e2e.yaml
```

## Writing New Steps

1. Implement a handler (e.g., `def _run_my_step(step: PipelineStep) -> None`) in
   `cli/orchestrate.py`.
2. Register the handler inside `_execute_step` by matching `step.uses`.
3. Ensure outputs follow the artifact protocol (`create_artifact`).
4. Document the expected `params` / `in` / `out` schema for your team.

## Best Practices

- Prefer local filesystem paths for intermediate artifacts; wrap remote sync in
  dedicated steps if needed.
- Keep `params` small and serialisable—avoid passing large blobs inline.
- Use `dry_run: true` for steps that set up infrastructure (e.g., serving) when
  building CI smoke tests.
- Store pipeline-level variables in a YAML anchor section or drive pipeline
  execution from Python when dynamic substitution is required.
