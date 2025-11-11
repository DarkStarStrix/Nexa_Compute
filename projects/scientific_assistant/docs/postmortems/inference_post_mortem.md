# Nexa_Sci_distilled_Falcon-10B Inference Post-Mortem

## Overview

We benchmarked the `Allanatrix/Nexa_Sci_distilled_Falcon-10B` checkpoint on the dual
RTX 4090 (24 GB each) evaluation host using vLLM. The primary goal was to explore
throughput/latency trade-offs, capture long-form generations, and surface results
through a Streamlit dashboard for easy comparison.

## Configuration Summary

| Config Name               | Tensor Parallel | Max Seq Len | Max New Tokens | Max Num Seq | GPU Util | Tokens / s | Avg Latency (s) |
|---------------------------|-----------------|-------------|----------------|-------------|----------|------------|-----------------|
| `tp2_len1536_batch4`      | 2               | 1536        | 192            | 4           | 0.88     | 237.9      | 0.807           |
| `tp1_len1536_batch6`      | 1               | 1536        | 256            | 6           | 0.95     | 244.6      | 1.047           |
| `tp1_len1024_batch24`     | 1               | 1024        | 256            | 24          | 0.93     | 878.0      | 0.292           |
| `tp1_len768_batch32`      | 1               | 768         | 192            | 12          | 0.88     | 478.5      | 0.401           |
| `tp1_len2048_batch4_long` | 1               | 2048        | 1024           | 4           | 0.95     | 148.0      | 6.149           |

All runs used bf16 weights (no quantization). Long-form generations require
aggressive `max_new_tokens` and higher KV cache utilisation; constraining the batch
size to ≤ 4 kept memory stable.

## Key Findings

- **Throughput vs. Context:**
  - The best throughput (≈ 878 tokens/s) came from the TP=1, 24-way micro-batch with
    1 024-token context (`tp1_len1024_batch24`). This configuration is ideal for
    shorter completions.
  - Switching to 2 048-token contexts with 1 024 new tokens slashed throughput to
    ≈ 148 tokens/s. Long-form answers are feasible, but expect a ~6 s latency per
    request even with a small batch.

- **Memory Stability:**
  - TP=2 configs repeatedly hit CUDA OOM due to GPU0 retaining 21 GB of allocated
    memory between runs. Address by rebooting the host between high-KV workloads or
    setting up automated GPU resets.
  - Garbage-collecting between vLLM runs (`gc.collect() + torch.cuda.empty_cache()`)
    was necessary to keep TP=1 sweeps stable.

- **Dashboard Enhancements:**
  - Streamlit dashboard now reads all `*_metrics.json` and `*_responses.parquet`
    files, presenting full Q&A pairs, configuration metadata, and comparative charts.
  - vLLM responses are as generated; any truncation reflects the configured
    `max_new_tokens`.

- **Operational Lessons:**
  - Keep all benchmark commands inside tmux so operators stay in the loop.
  - Ngrok tunnelling is a quick way to share live dashboards; remember to pin the
    Streamlit server to localhost (`--server.address 127.0.0.1`) when exposing via
    ngrok.
  - Sync raw artifacts (`results/vllm_benchmarks/`) back to the repo before tearing
    down the GPU host to preserve provenance.

## Next Steps

- Explore INT4/AWQ quantized checkpoints to chase the 1 k tokens/s goal without
  sacrificing output length.
- Automate GPU memory resets or job isolation so TP=2 experiments can run without
  manual intervention.
- Extend the dashboard with evaluation judgments once the rubric pipeline is ready,
  enabling quality/performance trade-off analysis in one view.
