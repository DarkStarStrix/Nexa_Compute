# Data Organization

All datasets must be stored under the project slug to keep multi-project repos isolated.

## Raw Data

```
data/raw/{project_slug}/
```

- Contains source corpora, reference tables, and external assets
- Individual files retain descriptive names (`biology_enhanced.json`, etc.)

## Processed Data

```
data/processed/{project_slug}/
├── distillation/
├── tool_protocol/
├── training/
└── evaluation/
```

- Each subdirectory may contain additional hierarchy (`cleaned/`, `sft_datasets/`, `reports/`)
- Versioned filenames follow `{name}_v{version}.ext`

## Symlinks (Temporary)

During migration, compatibility symlinks may exist at `data/processed/<legacy>` and `data/raw/<legacy>` pointing to the project namespace. These must be removed once all references are updated.

## Metadata

- Store manifests in `projects/{project_slug}/manifests/`
- Include dataset lineage, schema, and provenance

## Storage Policy Alignment

- Ephemeral artifacts: `/workspace/tmp/{project_slug}/`
- Durable artifacts: `/mnt/nexa_durable/{project_slug}/`
- Local mirror: `~/nexa_compute/durable/{project_slug}/`

Follow the storage policy in `docs/Overview_of_Project/POLICY.md` for full details.

