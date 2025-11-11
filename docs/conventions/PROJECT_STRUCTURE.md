# Project Structure Conventions

Every NexaCompute project **must** adhere to the following layout. The scaffold command mirrors this structure for new projects.

```
projects/{project_slug}/
├── README.md
├── configs/
│   └── *.yaml
├── docs/
│   ├── README.md
│   └── postmortems/
├── manifests/
│   └── project_manifest.json
└── pipelines/
    └── *.yaml

data/
├── raw/{project_slug}/
└── processed/{project_slug}/
    ├── distillation/
    ├── tool_protocol/
    ├── training/
    └── evaluation/

artifacts/{project_slug}/
├── checkpoints/
├── eval/
└── runs/

logs/{project_slug}/
└── ...
```

## Core Principles

1. **Isolation:** Assets from different projects must never collide. All long-lived outputs live under the project slug.
2. **Discoverability:** Documentation, configs, and manifests reside alongside the project, not in shared top-level folders.
3. **Reproducibility:** Manifests and configs capture the exact state required to reproduce runs.
4. **Template Parity:** New projects begin from `projects/_template/` and inherit the same structure.

## Required Files

- `manifests/project_manifest.json` – Machine-readable metadata (slug, owner, dependencies, paths).
- `configs/*.yaml` – At least one training or evaluation config referencing project paths.
- `docs/README.md` – Outline of project documentation.
- `pipelines/*.yaml` – Optional in v1, required once automated pipelines are introduced.

Failure to comply with this structure causes the scaffold validation and CI guardrails to fail.

Run `python -m nexa_compute.cli.project validate` (also wired into pre-commit) to ensure a project meets these requirements.

