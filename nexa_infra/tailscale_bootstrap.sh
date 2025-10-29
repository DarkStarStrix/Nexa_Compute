#!/usr/bin/env bash
set -euo pipefail

if ! command -v tailscale >/dev/null 2>&1; then
  echo "tailscale binary not found; please install it first" >&2
  exit 1
fi

echo "Authenticating to Tailscale..."
tailscale up --accept-routes --accept-dns=false --ssh

echo "Tailscale bootstrap complete"
