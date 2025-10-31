# Postmortem – Stress Test (2025-10-31)

## Objective
Run a high-capacity fine-tune (DeBERTa-v3-large on IMDB) across the instrumented NexaCompute stack to exercise training, telemetry, artifact sync, and manifests.

## Run Summary
- **Command**:
  ```bash
  python3 scripts/test_hf_train.py \
    --model microsoft/deberta-v3-large \
    --dataset imdb \
    --batch-size 1 \
    --grad-accumulation 4 \
    --learning-rate 3e-6 \
    --fp16 \
    --allow-tf32 \
    --telemetry-interval 10 \
    --num-workers 8 \
    --tags stress-test \
    --wandb-run-name stress-5090-single \
    --s3-uri s3://nexacompute/ML_Checkpoints
  ```
- **Run ID**: `train_20251031_014415`
- **W&B**: https://wandb.ai/allanw-mk-none/nexa-compute/runs/cgtawijk
- **Status**: Manually stopped at ~step 3146/8000 once loss flattened at ~0 (no additional signal).

## Outcomes
- Training loop, telemetry callbacks, checkpoint pruning, and S3 sync all executed without error.
- NVML telemetry (`scripts/gpu_monitor.py`) and `runs/logs/stress_5090_single.log` capture live GPU utilisation and loss smoothing.
- Manifest stored at `runs/manifests/train_20251031_014415.json` with config fingerprint for reproducibility.

## Issues Observed
1. **NCCL errors on multi-GPU torchrun** – dual RTX 5090 run hit `cuda invalid argument` during DDP init. We constrained stress run to single GPU. Future fix: upgrade NCCL / PyTorch or adjust P2P/GDR settings.
2. **Gradient checkpoint + large accumulation** – Enabling checkpointing with high `grad_accumulation` triggered `Trying to backward through the graph a second time`. For this run we disabled checkpointing instead of retaining the graph.
3. **Tokeniser warnings** – Hugging Face notes missing byte fallback for the fast tokenizer; harmless but logged.

## Follow-up Actions
- Investigate NCCL 2.25 behaviour on 5090s; test with `NCCL_P2P_DISABLE=1` & `NCCL_GRAPH_MIX_DISABLE=1` or upgrade to PyTorch 2.2 stable.
- Consider adding automated plateau detection to stop long runs earlier.
- Replace `pynvml` with `nvidia-ml-py` (done) to avoid deprecation warnings.
- Document stress-test configuration under `nexa_train/configs/stress_5090_single.yaml` for future reruns.

## Artifacts
- Checkpoints: `/workspace/tmp/checkpoints_temp/train_20251031_014415` (not promoted due to early stop)
- Logs: `runs/logs/stress_5090_single.log`
- Manifest: `runs/manifests/train_20251031_014415.json`
- Infra summary: `runs/manifests/infra_report.json`
- Analysis: `runs/manifests/analysis.json`
