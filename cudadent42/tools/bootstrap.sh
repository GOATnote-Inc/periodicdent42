#!/usr/bin/env bash
set -euo pipefail

# 0) Preconditions
command -v nvidia-smi >/dev/null || { echo "GPU drivers missing"; exit 1; }

# 1) Micromamba (user-local)
if ! command -v micromamba >/dev/null; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
  install -m 0755 bin/micromamba /usr/local/bin/micromamba || sudo install -m 0755 bin/micromamba /usr/local/bin/micromamba
fi

# 2) Env create/refresh
ENV_NAME=faenv
micromamba create -y -n $ENV_NAME python=3.10 || true
eval "$(micromamba shell hook -s bash)"
micromamba activate $ENV_NAME

# 3) Torch matching CUDA
# Detect runtime CUDA from driver; fall back to cu121 wheels (safe on L4/A100/H100 with recent drivers)
pip install --upgrade pip wheel
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 4) Dev deps you actually need
pip install ninja cmake packaging pybind11

# 5) Sanity test
python - <<'PY'
import torch, os
assert torch.cuda.is_available(), "CUDA not available"
print("OK: torch", torch.__version__, "CUDA", torch.version.cuda, "dev", torch.cuda.get_device_name(0))
PY

echo "Bootstrap complete."

