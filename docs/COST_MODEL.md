# Cost Model

The cost tracker aggregates run-level compute spend captured per resource type.

## Inputs
- `compute_hours`: GPU/CPU billing hours multiplied by instance hourly rate.
- `storage_gb_month`: Persistent volume usage normalised to monthly cost.
- `egress_gb`: Network data egress charges.

## Outputs
Generated manifests (`runs/manifests/cost_<run_id>.json`) record:

```json
{
  "run_id": "baseline-2025-01-01",
  "breakdown": {
    "compute_hours": 42.5,
    "storage_gb_month": 12.4,
    "egress_gb": 1.2
  },
  "total": 56.1
}
```

`summarize_costs` aggregates totals for rapid reporting. For production usage integrate with billing APIs (AWS Cost Explorer, GCP Billing) and feed into this structure.
