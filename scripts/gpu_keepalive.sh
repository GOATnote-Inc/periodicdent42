#!/usr/bin/env bash
set -euo pipefail
echo "[gpu_keepalive] starting keepalive loop (nvidia-smi every 5m)"
while true; do
  nvidia-smi >/dev/null 2>&1 || true
  sleep 300
done

