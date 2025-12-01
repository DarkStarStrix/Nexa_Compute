#!/usr/bin/env bash
#
# Sanitized template for launching a Nexa training run locally or via torchrun.

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-nexa_train/configs/baseline_distill.yaml}"
RUN_NAME="${RUN_NAME:-sanitized-run}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"

export WANDB_API_KEY="${WANDB_API_KEY:-CHANGEME}"
export HF_TOKEN="${HF_TOKEN:-CHANGEME}"

if [[ "${NODES}" -eq 1 && "${GPUS_PER_NODE}" -eq 1 ]]; then
  echo "[train] running single GPU training for ${RUN_NAME}"
  python -m nexa_train.train --config "$CONFIG_PATH" --run-name "$RUN_NAME" "$@"
else
  echo "[train] running distributed training (${NODES} nodes x ${GPUS_PER_NODE} gpus)"
  torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --nnodes="${NODES}" \
    --node_rank="${MACHINE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m nexa_train.train \
    --config "$CONFIG_PATH" \
    --run-name "$RUN_NAME" \
    "$@"
fi

