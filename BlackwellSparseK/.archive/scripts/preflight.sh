#!/usr/bin/env bash
set -euo pipefail

echo "[Preflight] Verifying H100 + CUDA 13.0.2 + CUTLASS 4.3.0 + sm_90a"

nvcc --version | grep -q "release 13.0" || { echo "Wrong CUDA version"; exit 1; }
nvidia-smi -q | grep -q "H100" || { echo "Not an H100 GPU"; exit 1; }

test -d /opt/cutlass || { echo "CUTLASS missing"; exit 1; }
grep -q "sm_90a" <(strings ./sparse_h100) || { echo "Binary not built for sm_90a"; exit 1; }

echo "[Preflight] OK"

