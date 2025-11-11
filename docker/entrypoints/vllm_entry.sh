#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${MODEL:-}" ]]; then
  echo "MODEL env var not set. Starting shell."
  exec /bin/bash
fi

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"

echo "Starting vLLM: model=${MODEL} port=${PORT} tp=${TP}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --host "${HOST}" \
  --tensor-parallel-size "${TP}" \
  ${VLLM_EXTRA_ARGS:-}

