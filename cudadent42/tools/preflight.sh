#!/usr/bin/env bash
set -euo pipefail

echo "== Preflight =="

# CUDA toolchain in PATH/LD_LIBRARY_PATH (common gotcha on fresh VMs)
if ! command -v nvcc >/dev/null 2>&1; then
  if [[ -d /usr/local/cuda/bin ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
  fi
fi
if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

# GPU + driver present
nvidia-smi >/dev/null

# Python + Torch CUDA present & usable
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("Torch sees no CUDA device")
print(f"torch={torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)}")
PY

echo "Preflight OK"

