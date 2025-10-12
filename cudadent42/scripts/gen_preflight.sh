#!/usr/bin/env bash
set -euo pipefail
mkdir -p tools
if [[ ! -f tools/preflight.sh ]]; then
  cat > tools/preflight.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "== Preflight =="
if ! command -v nvcc >/dev/null 2>&1; then
  if [[ -d /usr/local/cuda/bin ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
  fi
fi
if [[ -d /usr/local/cuda/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
nvidia-smi >/dev/null
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("Torch sees no CUDA device")
print(f"torch={torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)}")
PY
echo "Preflight OK"
EOF
  chmod +x tools/preflight.sh
fi
echo "Preflight present at tools/preflight.sh"

