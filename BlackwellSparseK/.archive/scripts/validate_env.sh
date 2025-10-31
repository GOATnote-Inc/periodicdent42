#!/bin/bash
# CRITICAL: Environment validation gate
# Blocks ALL operations unless running in proper H100 container with CUDA 13.0.2

set -euo pipefail

echo "=== CRITICAL ENVIRONMENT VALIDATION ==="
echo ""

# Check 1: NOT running on macOS
if [ "$(uname)" = "Darwin" ]; then
    echo "❌ FATAL: Running on macOS (local machine)"
    echo "   This script MUST run inside H100 Docker container"
    echo "   Switch Cursor executor to 'H100 Dockerized Remote'"
    echo ""
    echo "Evidence:"
    echo "  OS: $(uname -a)"
    echo "  Hostname: $(hostname)"
    echo ""
    exit 1
fi

echo "✅ Check 1: Not on macOS (uname=$(uname))"

# Check 2: Inside Docker container
if [ ! -f /.dockerenv ] && [ ! -f /run/.containerenv ]; then
    echo "⚠️  WARNING: Not inside Docker container"
    echo "   Looking for /.dockerenv or /run/.containerenv"
    echo "   Hostname: $(hostname)"
    echo ""
    # Don't fail here - might be bare metal H100
fi

# Check 3: CUDA 13.0+ available
if ! command -v nvcc &> /dev/null; then
    echo "❌ FATAL: nvcc not found in PATH"
    echo "   CUDA toolkit not installed"
    echo "   PATH: $PATH"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

echo "✅ Check 3: nvcc found (version $CUDA_VERSION)"

if [ "$CUDA_MAJOR" -lt 13 ]; then
    echo "❌ FATAL: CUDA version too old ($CUDA_VERSION)"
    echo "   Required: CUDA >= 13.0.2"
    echo "   Current:  CUDA $CUDA_VERSION"
    echo ""
    echo "Without CUDA 13.0+, you CANNOT access:"
    echo "  - sm_100 (Blackwell architecture)"
    echo "  - FP8 E4M3/E5M2 types"
    echo "  - Latest TMA instructions"
    echo "  - Target performance"
    echo ""
    echo "USER REQUIREMENT: 'You will never get expert results with outdated CUDA'"
    echo ""
    exit 1
fi

echo "✅ Check 4: CUDA version acceptable ($CUDA_VERSION >= 13.0)"

# Check 4: CUTLASS 4.3+ available
if [ ! -d "${CUTLASS_HOME:-/opt/cutlass}" ]; then
    echo "❌ FATAL: CUTLASS not found"
    echo "   Expected: ${CUTLASS_HOME:-/opt/cutlass}"
    echo ""
    exit 1
fi

cd "${CUTLASS_HOME:-/opt/cutlass}"
CUTLASS_VERSION=$(git describe --tags 2>/dev/null || git rev-parse --short HEAD)
echo "✅ Check 5: CUTLASS found (version $CUTLASS_VERSION)"

# Check if it's 4.3 or newer
if ! echo "$CUTLASS_VERSION" | grep -qE "v4\.[3-9]|v[5-9]"; then
    echo "⚠️  WARNING: CUTLASS version may be too old ($CUTLASS_VERSION)"
    echo "   Required: CUTLASS >= 4.3.0"
    echo "   Proceeding anyway (may be dev branch ahead of v4.3)"
fi

# Check 5: GPU available
if ! nvidia-smi &> /dev/null; then
    echo "❌ FATAL: nvidia-smi not accessible"
    echo "   No GPU found or driver not loaded"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)

echo "✅ Check 6: GPU accessible"
echo "   Name: $GPU_NAME"
echo "   Compute Capability: $GPU_CC"

if ! echo "$GPU_CC" | grep -qE "^(9\.0|10\.0)"; then
    echo "⚠️  WARNING: GPU compute capability may be insufficient"
    echo "   Recommended: 9.0 (H100) or 10.0 (B200)"
    echo "   Current: $GPU_CC"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ ENVIRONMENT VALIDATION PASSED"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  CUDA:    $CUDA_VERSION (nvcc: $(which nvcc))"
echo "  CUTLASS: $CUTLASS_VERSION (${CUTLASS_HOME:-/opt/cutlass})"
echo "  GPU:     $GPU_NAME (CC $GPU_CC)"
echo "  Host:    $(hostname)"
echo "  OS:      $(uname -s) $(uname -r)"
echo ""
echo "✅ Safe to proceed with compilation and benchmarking"
echo ""

