# Training Failure Recovery Runbook

## Symptoms
- Alert: `Training Failed` (Critical)
- Log message: `Training terminated unexpectedly`
- Dashboard: `Training Loss` drops to 0 or `GPU Utilization` flatlines

## Triage
1. **Check Training Logs**:
   - Access logs via Grafana or `logs/train.log`
   - Look for Python exceptions (e.g., `RuntimeError`, `CUDA error`)

2. **Check Infrastructure**:
   - Is the node reachable?
   - `ssh` into the node and run `nvidia-smi` to check GPU health.

## Common Issues & Fixes

### CUDA Out of Memory (OOM)
- **Symptom**: `RuntimeError: CUDA out of memory`
- **Fix**:
  1. The system should automatically reduce batch size and retry.
  2. If it fails repeatedly, manually decrease `batch_size` in `config.yaml`.
  3. Enable gradient accumulation steps.

### Loss Divergence (NaN/Inf)
- **Symptom**: Loss becomes `nan` or `inf`.
- **Fix**:
  1. Lower the learning rate (`lr`).
  2. Check input data for anomalies (nulls, infinities).
  3. Enable gradient clipping (`max_grad_norm`).

### Distributed Training Hang
- **Symptom**: Processes waiting indefinitely (NCCL timeout).
- **Fix**:
  1. Check network connectivity between nodes.
  2. Verify `MASTER_ADDR` and `MASTER_PORT` are reachable.
  3. Restart the job using the checkpoint: `resume_from_checkpoint=True`.

## Escalation
If the issue persists after 3 retries or involves hardware failure (e.g., GPU ECC errors), escalate to the Infrastructure Team.

