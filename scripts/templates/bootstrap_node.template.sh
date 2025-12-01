#!/usr/bin/env bash
#
# Sanitized template for provisioning a remote training node.

set -euo pipefail

: "${NEXA_REMOTE_USER:?Set NEXA_REMOTE_USER to your SSH username}"
: "${NEXA_REMOTE_HOST:?Set NEXA_REMOTE_HOST to the remote host/IP}"
: "${NEXA_REMOTE_ROOT:=/workspace/nexa_compute}"

SSH_TARGET="${NEXA_REMOTE_USER}@${NEXA_REMOTE_HOST}"

echo "[bootstrap] creating remote workspace at ${NEXA_REMOTE_ROOT}"
ssh -o StrictHostKeyChecking=accept-new "${SSH_TARGET}" "mkdir -p ${NEXA_REMOTE_ROOT}"

echo "[bootstrap] syncing repository (sanitized)"
rsync -az --delete \
  --exclude '.git' \
  --exclude 'artifacts/' \
  --exclude 'data/' \
  . "${SSH_TARGET}:${NEXA_REMOTE_ROOT}"

echo "[bootstrap] installing dependencies on remote host"
ssh "${SSH_TARGET}" <<'REMOTE_CMDS'
set -euo pipefail
cd "${NEXA_REMOTE_ROOT}"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements/requirements-dev.lock
REMOTE_CMDS

echo "[bootstrap] remote node ready. Attach via:"
echo "ssh ${SSH_TARGET} 'cd ${NEXA_REMOTE_ROOT} && tmux a -t nexa || tmux new -s nexa'"

