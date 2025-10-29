# Run Manifest Schema

Each training run records a manifest at `runs/manifests/run_manifest.json` with the following structure:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NexaRunManifest",
  "type": "object",
  "properties": {
    "config": {"type": "string"},
    "run_dir": {"type": "string"},
    "metrics": {
      "type": "object",
      "additionalProperties": {"type": "number"}
    },
    "checkpoint": {"type": ["string", "null"]},
    "created_at": {"type": "string", "format": "date-time", "optional": true}
  },
  "required": ["config", "run_dir", "metrics"],
  "additionalProperties": false
}
```

This schema enables downstream services (leaderboards, monitoring, billing) to parse artifacts deterministically.
