#!/usr/bin/env bash
# =============================================================================
# GPU Preflight Checks for FlashCore
# =============================================================================
# Validates GPU, CUDA, PyTorch before compile/bench/profile
# Usage: bash scripts/preflight.sh
# Exit codes: 0 = all pass, 1 = warning, 2 = critical failure

set -euo pipefail

# Source CUDA environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_cuda_l4.sh"

echo ""
echo "=============================================================================="
echo "FlashCore GPU Preflight Check"
echo "=============================================================================="

# Track failures
WARNINGS=0
ERRORS=0

# =============================================================================
# 1. GPU Hardware Check
# =============================================================================
echo ""
echo "== GPU Hardware =="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
        echo "[ERROR] nvidia-smi failed to query GPU"
        ERRORS=$((ERRORS + 1))
    }
else
    echo "[ERROR] nvidia-smi not found - no NVIDIA GPU detected"
    ERRORS=$((ERRORS + 1))
fi

# =============================================================================
# 2. CUDA Toolkit Check
# =============================================================================
echo ""
echo "== CUDA Toolkit =="
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release" || true
    echo "[OK] nvcc found: $(which nvcc)"
else
    echo "[ERROR] nvcc not found in PATH"
    ERRORS=$((ERRORS + 1))
fi

# =============================================================================
# 3. Nsight Compute Check
# =============================================================================
echo ""
echo "== Nsight Compute =="
if command -v ncu &> /dev/null; then
    echo "[OK] ncu found: $(which ncu)"
    ncu --version | head -1 || true
else
    echo "[WARNING] ncu not found - profiling will be unavailable"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# 4. PyTorch CUDA Check
# =============================================================================
echo ""
echo "== PyTorch CUDA =="
python3 - <<'PY' || {
    echo "[ERROR] PyTorch CUDA check failed"
    ERRORS=$((ERRORS + 1))
}
import torch
import sys

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: {cap[0]}.{cap[1]}")
    
    # Check for L4 (sm_89)
    if cap == (8, 9):
        print("[OK] L4 GPU detected (sm_89)")
    else:
        print(f"[WARNING] Expected L4 (sm_89), got {cap[0]}.{cap[1]}")
    
    sys.exit(0)
else:
    print("[ERROR] CUDA not available in PyTorch")
    sys.exit(2)
PY

# =============================================================================
# 5. Environment Variables Check
# =============================================================================
echo ""
echo "== Environment Variables =="
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "CUDA_ARCH: ${CUDA_ARCH:-not set}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-not set}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-not set}"

if [ -z "${CUDA_ARCH:-}" ]; then
    echo "[WARNING] CUDA_ARCH not set (should be 8.9 for L4)"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================================================="
echo "Preflight Summary"
echo "=============================================================================="
echo "Warnings: $WARNINGS"
echo "Errors: $ERRORS"

if [ $ERRORS -gt 0 ]; then
    echo "[FAILED] Critical errors detected - cannot proceed"
    exit 2
elif [ $WARNINGS -gt 0 ]; then
    echo "[WARNING] Some checks failed - proceed with caution"
    exit 1
else
    echo "[PASS] All preflight checks passed ✅"
    exit 0
fi

