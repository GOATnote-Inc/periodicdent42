#!/bin/bash
# Bootstrap EvoEngineer + robust-kbench tools
# Phase 1: Tool Integration

set -e

echo "============================================================"
echo "Bootstrapping Optimization Tools"
echo "============================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python: $PYTHON_VERSION"

# Check required packages
echo ""
echo "Checking Python dependencies..."

REQUIRED_PACKAGES=(
    "torch"
    "numpy"
    "pyyaml"
)

MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        echo "  ✓ $pkg ($VERSION)"
    else
        echo "  ✗ $pkg (missing)"
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Missing packages: ${MISSING_PACKAGES[*]}"
    echo "    Install with: pip3 install --user ${MISSING_PACKAGES[*]}"
    echo "    Or on GPU instance: pip3 install ${MISSING_PACKAGES[*]}"
    
    # Only attempt install if not in externally-managed environment
    if pip3 install --help | grep -q "break-system-packages"; then
        echo "    Skipping auto-install (externally-managed environment)"
    else
        pip3 install "${MISSING_PACKAGES[@]}"
    fi
fi

# Verify tool installations
echo ""
echo "Verifying tool installations..."

if python3 -c "import sys; sys.path.insert(0, '.'); from third_party.evoengineer import KernelOptimizer" 2>/dev/null; then
    echo "  ✓ EvoEngineer"
else
    echo "  ✗ EvoEngineer failed to import"
    exit 1
fi

if python3 -c "import sys; sys.path.insert(0, '.'); from third_party.robust_kbench import BenchmarkRunner" 2>/dev/null; then
    echo "  ✓ robust-kbench"
else
    echo "  ✗ robust-kbench failed to import"
    exit 1
fi

# Print tool versions
echo ""
echo "Tool versions:"
echo "  • EvoEngineer: $(python3 -c 'import sys; sys.path.insert(0, "."); from third_party.evoengineer import __version__; print(__version__)')"
echo "  • robust-kbench: $(python3 -c 'import sys; sys.path.insert(0, "."); from third_party.robust_kbench import __version__; print(__version__)')"

# Verify CUDA availability
echo ""
echo "Verifying CUDA environment..."
if python3 -c "import torch; assert torch.cuda.is_available(); print('✓ CUDA available')" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    COMPUTE_CAP=$(python3 -c "import torch; cap = torch.cuda.get_device_capability(0); print(f'{cap[0]}.{cap[1]}')")
    echo "  • GPU: $GPU_NAME"
    echo "  • Compute Capability: $COMPUTE_CAP"
else
    echo "  ⚠️  CUDA not available (GPU benchmarks will fail)"
fi

echo ""
echo "============================================================"
echo "✅ Bootstrap Complete"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review third_party/LOCKFILE.md for tool versions"
echo "  2. Run Phase 2: Baselines (correctness + SDPA benchmarks)"
echo "  3. Check logs in benchmarks/l4/$(date +%Y-%m-%d)/"

