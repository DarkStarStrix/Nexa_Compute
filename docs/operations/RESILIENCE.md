# NexaCompute Resilience Guide

This document outlines the fault-tolerance and safety mechanisms baked into the platform as of the latest update.

## Retry & Circuit Breakers
- Centralised retry helpers (`src/nexa_compute/utils/retry.py`) provide exponential backoff, jitter, and optional circuit breaker hooks.
- External API integrations such as `nexa_eval/clients/openrouter.py` now wrap requests with both retry logic and a circuit breaker to prevent cascading failures.
- Distributed training initialisation (`src/nexa_compute/training/distributed.py`) retries NCCL/TCP setup and falls back to single-node execution if the cluster handshake cannot be established.

## Security & Input Validation
- API requests are sanitized via Pydantic validators (`src/nexa_compute/api/models.py`) limiting payload sizes and preventing path traversal strings.
- Sandbox execution (`nexa_tools/sandbox.py`) now validates Python ASTs, blocks dangerous imports (os, subprocess, etc.), and runs user code inside an isolated Docker container with strict resource limits and a network-disabled environment.
- API authentication enforces production-only mode, per-key rate limiting, and key rotation helpers (`src/nexa_compute/api/auth.py`, `api/middleware.py`).

## Secrets Management
- `src/nexa_compute/utils/secrets.py` introduces a backend-agnostic loader (env, AWS Secrets Manager, Vault) with required-secret validation at startup (`api/config.py`).

## Training Resilience
- Training timeouts configurable via `training.timeout_seconds` guard against runaway runs (`Trainer.fit` wrapped with `execution_timeout`).
- Checkpoint resumption supports automatic discovery of the latest checkpoint and restoring trainer state (epoch, global step, metrics).
- CUDA OOM events trigger automatic gradient accumulation adjustments with GPU memory telemetry emitted every batch.

## Health, Monitoring, and Testing
- `/health`, `/health/ready`, and `/health/metrics` endpoints provide liveness, readiness, and lightweight operational stats.
- Resource sync helpers add checksum verification and retry logic for shard/cp copies (`storage.py`, `nexa_data/msms/shard_writer.py`).
- `tests/test_resilience.py` covers retry logic, circuit breaker state transitions, secret loading, timeout handling, and sandbox validation.

## Operational Checklist
1. Ensure required secrets are declared via `NEXA_REQUIRED_SECRETS`.
2. Monitor rate-limit headers (`X-RateLimit-*`) returned by API responses.
3. For distributed jobs, watch logs for `distributed_launch_failed`; the pipeline now auto-falls back to single-node.
4. When resuming training, pass `resume_from_checkpoint=True` (or a specific path) to `TrainingPipeline.run`.

