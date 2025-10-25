#!/usr/bin/env bash
# =============================================================================
# CUDA Environment Setup for L4 GPU
# =============================================================================
# Sets CUDA_HOME, PATH, and LD_LIBRARY_PATH for CUDA 12.2 on L4 instance
# Usage: source scripts/env_cuda_l4.sh (idempotent)

set -euo pipefail

# CUDA installation path (adjust if needed)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"

# Add CUDA binaries to PATH
export PATH="$CUDA_HOME/bin:${PATH}"

# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# PyTorch CUDA allocator settings (prevent fragmentation)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Compilation defaults for L4 (Ada, sm_89)
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_ARCH="8.9"

# Verify nvcc is accessible
if ! command -v nvcc &> /dev/null; then
    echo "[ERROR] nvcc not found in PATH after setting CUDA_HOME=$CUDA_HOME"
    echo "[ERROR] Check that CUDA 12.2 is installed in $CUDA_HOME"
    exit 1
fi

echo "[env] CUDA environment configured:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  nvcc: $(which nvcc)"
echo "  nvcc version: $(nvcc --version | grep release)"
echo "  CUDA_ARCH: $CUDA_ARCH (sm_89 for L4)"

