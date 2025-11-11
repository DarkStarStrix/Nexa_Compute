#!/usr/bin/env bash

set -euo pipefail

IMAGE="${1:-ghcr.io/nexa/nexa_light:latest}"
if [ "$#" -gt 0 ]; then
  shift
fi
EXTRA_ARGS=("$@")
WORK="${WORK:-/workspace}"
NVME="${NVME:-/mnt/nvme}"
SHM_SIZE="${SHM_SIZE:-16g}"

mkdir -p "${NVME}" "${HOME}/.cache/huggingface"

docker pull "${IMAGE}"

DOCKER_CMD=(
  docker run -it --gpus all --rm
  --shm-size="${SHM_SIZE}"
  -v "${HOME}/.cache/huggingface":/home/runner/.cache/huggingface
  -v "${NVME}":"${NVME}"
  -v "${PWD}":"${WORK}"/nexa_compute
)

if [ -f ".env" ]; then
  DOCKER_CMD+=(--env-file .env)
fi

DOCKER_CMD+=("${EXTRA_ARGS[@]}")
DOCKER_CMD+=("${IMAGE}")

"${DOCKER_CMD[@]}"

