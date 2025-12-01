#!/usr/bin/env bash
#
# Sanitized template for starting the Nexa Forge API + dashboards on a single machine.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/forge}"
PORT="${PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"

mkdir -p "${LOG_DIR}"

echo "[forge] starting API server on port ${PORT}"
nohup bash -c "
  cd '${ROOT_DIR}'
  uvicorn src.nexa_compute.api.main:app \\
    --host 0.0.0.0 \\
    --port ${PORT}
" >\"${LOG_DIR}/api.log\" 2>&1 &
echo $! > "${LOG_DIR}/api.pid"

echo "[forge] starting dashboard on port ${UI_PORT}"
nohup bash -c "
  cd '${ROOT_DIR}/frontend'
  npm install
  npm run dev -- --host 0.0.0.0 --port ${UI_PORT}
" >\"${LOG_DIR}/ui.log\" 2>&1 &
echo $! > "${LOG_DIR}/ui.pid"

echo "[forge] services are launching. Tail logs via:"
echo "  tail -f ${LOG_DIR}/api.log"
echo "  tail -f ${LOG_DIR}/ui.log"

