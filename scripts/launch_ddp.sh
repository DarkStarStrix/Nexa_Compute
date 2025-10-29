#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/distributed.yaml"
NNODES=1
NPROC_PER_NODE=2
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --nodes)
      NNODES="$2"
      shift 2
      ;;
    --gpus)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --node-rank)
      NODE_RANK="$2"
      shift 2
      ;;
    --master-addr)
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

export MASTER_ADDR MASTER_PORT NODE_RANK NNODES

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/cli.py train --config "${CONFIG}" "${EXTRA_ARGS[@]}"
