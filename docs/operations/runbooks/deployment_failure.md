# Model Deployment Runbook

## Symptoms
- Alert: `Deployment Failed`
- API Latency spikes or timeouts (`504 Gateway Timeout`)
- Metrics: `nexa_api_request_latency_seconds` increasing

## Deployment Process
1. **Build Container**: `docker build -f docker/infer.Dockerfile ...`
2. **Push to Registry**: `docker push ghcr.io/nexa/nexa_infer:latest`
3. **Update Service**: `kubectl set image deployment/nexa-infer ...` (or equivalent)

## Rollback Procedure
If a new deployment fails health checks or degrades performance:

1. **Identify Previous Version**:
   - Check `deployment_history` in the database or CI/CD logs.
   
2. **Revert Image**:
   - `kubectl rollout undo deployment/nexa-infer`
   
3. **Verify Recovery**:
   - Check `/health` endpoint.
   - Monitor error rate (`rate(nexa_api_requests_total{status_code=~"5.."}[5m])`).

## Troubleshooting

### Container Crash Loop
- **Check Logs**: `docker logs <container_id>`
- **Common Causes**:
  - Missing model weights (volume mount failure).
  - Incompatible CUDA drivers.
  - OOM during model load (increase memory limit).

### Health Check Failures
- Ensure the model is fully loaded before the server accepts traffic.
- Increase `readinessProbe` initial delay if loading takes longer (e.g., large LLMs).

