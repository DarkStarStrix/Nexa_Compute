# Nexa Forge API Reference

## Overview
The Nexa Forge API orchestrates the entire machine learning lifecycle, from data preparation to model deployment. It is built on FastAPI and follows RESTful principles.

## Authentication
All endpoints (except health checks) require an API key.
- **Header**: `X-Nexa-Api-Key: <your_api_key>`
- **Rate Limits**: 
  - Default: 60 requests/minute
  - Burst: 10 requests

## Endpoints

### Jobs
Manage compute jobs (training, evaluation, distillation).

- `POST /api/jobs/submit`: Submit a new job.
- `GET /api/jobs/{job_id}`: Get job status.
- `POST /api/jobs/{job_id}/cancel`: Cancel a running job.

### Workflows
Manage complex, multi-step pipelines.

- `POST /api/workflows/submit`: Trigger a workflow execution.
- `GET /api/workflows/{run_id}`: Check workflow progress.

### Artifacts
Access generated artifacts (checkpoints, datasets).

- `GET /api/artifacts`: List available artifacts.
- `GET /api/artifacts/{artifact_id}/download`: Get a signed download URL.

### Metrics & Monitoring
- `GET /metrics`: Prometheus metrics (standard format).
- `GET /health`: System liveness.
- `GET /health/ready`: System readiness (database, storage connectivity).

## Error Handling
Errors are returned as JSON with standard HTTP status codes.

```json
{
  "detail": "Detailed error message",
  "code": "error_code"
}
```

- `400`: Bad Request (validation failure)
- `401`: Unauthorized (invalid API key)
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found
- `429`: Rate Limit Exceeded
- `500`: Internal Server Error

