#!/bin/bash
# Bootstrap script for EvoEngineer + robust-kbench optimization workflow
# Sets up Python environment, verifies dependencies, and validates tool imports

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================================"
echo "Bootstrapping CUDA Kernel Optimization Tools"
echo "================================================================================"
echo "Repository: ${REPO_ROOT}"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================================
# 1) Python Version Check
# ============================================================================
echo "1) Python Version Check"
echo "--------------------------------------------------------------------------------"

PYTHON_CMD="python3"
if ! command -v ${PYTHON_CMD} &> /dev/null; then
    echo "❌ ERROR: python3 not found in PATH"
    exit 1
fi

PYTHON_VERSION=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PYTHON_MAJOR=$(${PYTHON_CMD} -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(${PYTHON_CMD} -c "import sys; print(sys.version_info.minor)")

echo "✓ Python: ${PYTHON_VERSION}"

if [ "${PYTHON_MAJOR}" -lt 3 ] || ([ "${PYTHON_MAJOR}" -eq 3 ] && [ "${PYTHON_MINOR}" -lt 8 ]); then
    echo "❌ ERROR: Python 3.8+ required, found ${PYTHON_VERSION}"
    exit 1
fi

# ============================================================================
# 2) Required Python Packages
# ============================================================================
echo ""
echo "2) Required Python Packages"
echo "--------------------------------------------------------------------------------"

REQUIRED_PACKAGES=("torch" "numpy" "pyyaml")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ${PYTHON_CMD} -c "import ${pkg}" &> /dev/null; then
        VERSION=$(${PYTHON_CMD} -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null || echo "N/A")
        echo "  ✓ ${pkg} (${VERSION})"
    else
        echo "  ✗ ${pkg} (missing)"
        MISSING_PACKAGES+=("${pkg}")
    fi
done

# Check for ninja (required for PyTorch C++ extensions)
if ${PYTHON_CMD} -c "import ninja" &> /dev/null; then
    echo "  ✓ ninja ($(${PYTHON_CMD} -c 'import ninja; print(ninja.__version__)'))"
else
    echo "  ✗ ninja (missing)"
    MISSING_PACKAGES+=("ninja")
fi

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Missing packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "Install with:"
    echo "  pip3 install --user ${MISSING_PACKAGES[*]}"
    echo "  OR (on GPU instance): pip3 install ${MISSING_PACKAGES[*]}"
    echo ""
    
    # Only attempt auto-install if NOT in externally-managed environment
    if pip3 install --help 2>&1 | grep -q "break-system-packages"; then
        echo "ℹ️  Externally-managed environment detected. Skipping auto-install."
        echo "   Please install manually using --user flag or virtual environment."
        exit 1
    else
        echo "Attempting auto-install..."
        if pip3 install "${MISSING_PACKAGES[@]}"; then
            echo "✓ Packages installed successfully"
        else
            echo "❌ Failed to install packages. Please install manually."
            exit 1
        fi
    fi
fi

# ============================================================================
# 3) CUDA/GPU Availability (if running locally with GPU)
# ============================================================================
echo ""
echo "3) CUDA/GPU Availability"
echo "--------------------------------------------------------------------------------"

if ${PYTHON_CMD} -c "import torch; torch.cuda.is_available()" &> /dev/null; then
    GPU_NAME=$(${PYTHON_CMD} -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    CUDA_VERSION=$(${PYTHON_CMD} -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Unknown")
    COMPUTE_CAP=$(${PYTHON_CMD} -c "import torch; major, minor = torch.cuda.get_device_capability(0); print(f'{major}.{minor}')" 2>/dev/null || echo "Unknown")
    
    echo "  ✓ CUDA Available: Yes"
    echo "  ✓ GPU: ${GPU_NAME}"
    echo "  ✓ CUDA Version: ${CUDA_VERSION}"
    echo "  ✓ Compute Capability: ${COMPUTE_CAP}"
    
    # Validate L4 (sm_89) for this workflow
    if [ "${COMPUTE_CAP}" == "8.9" ]; then
        echo "  ✅ Target GPU validated: L4 (sm_89)"
    else
        echo "  ⚠️  Warning: Expected L4 (sm_89), found compute capability ${COMPUTE_CAP}"
    fi
else
    echo "  ⚠️  CUDA not available (running on CPU-only machine)"
    echo "     GPU tests will need to be run on GPU instance"
fi

# ============================================================================
# 4) Tool Import Verification
# ============================================================================
echo ""
echo "4) Tool Import Verification"
echo "--------------------------------------------------------------------------------"

cd "${REPO_ROOT}"

# Add repo root to PYTHONPATH for imports
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Verify EvoEngineer
if ${PYTHON_CMD} -c "from third_party.evoengineer.optimizer import EvoEngineerOptimizer" &> /dev/null; then
    echo "  ✓ EvoEngineer: optimizer.py (importable)"
else
    echo "  ✗ EvoEngineer: optimizer.py (import failed)"
    exit 1
fi

if ${PYTHON_CMD} -c "from third_party.evoengineer.evaluator import EvoEngineerEvaluator" &> /dev/null; then
    echo "  ✓ EvoEngineer: evaluator.py (importable)"
else
    echo "  ✗ EvoEngineer: evaluator.py (import failed)"
    exit 1
fi

if ${PYTHON_CMD} -c "from third_party.evoengineer.mutator import EvoEngineerMutator" &> /dev/null; then
    echo "  ✓ EvoEngineer: mutator.py (importable)"
else
    echo "  ✗ EvoEngineer: mutator.py (import failed)"
    exit 1
fi

# Verify robust-kbench
if ${PYTHON_CMD} -c "from third_party.robust_kbench.config import RBKConfig" &> /dev/null; then
    echo "  ✓ robust-kbench: config.py (importable)"
else
    echo "  ✗ robust-kbench: config.py (import failed)"
    exit 1
fi

if ${PYTHON_CMD} -c "from third_party.robust_kbench.runner import RBKRunner" &> /dev/null; then
    echo "  ✓ robust-kbench: runner.py (importable)"
else
    echo "  ✗ robust-kbench: runner.py (import failed)"
    exit 1
fi

if ${PYTHON_CMD} -c "from third_party.robust_kbench.reporter import RBKReporter" &> /dev/null; then
    echo "  ✓ robust-kbench: reporter.py (importable)"
else
    echo "  ✗ robust-kbench: reporter.py (import failed)"
    exit 1
fi

# ============================================================================
# 5) Pinned Versions Summary
# ============================================================================
echo ""
echo "5) Pinned Versions Summary"
echo "--------------------------------------------------------------------------------"
echo "See third_party/LOCKFILE.md for full details:"
echo ""
echo "  • EvoEngineer: Internal v1.0 (2025-10-15)"
echo "  • robust-kbench: Internal v1.0 (2025-10-15)"
echo ""
echo "  PyTorch: $(${PYTHON_CMD} -c 'import torch; print(torch.__version__)')"
echo "  NumPy: $(${PYTHON_CMD} -c 'import numpy; print(numpy.__version__)')"
echo "  PyYAML: $(${PYTHON_CMD} -c 'import yaml; print(yaml.__version__)')"

# ============================================================================
# 6) Bootstrap Complete
# ============================================================================
echo ""
echo "================================================================================"
echo "✅ Bootstrap Complete"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Run correctness tests: python3 tests/test_sdpa_parity.py"
echo "  2. Run baseline benchmarks: python3 scripts/bench_sdpa_baseline.py"
echo "  3. Run robust-kbench: python3 scripts/run_rbk_benchmark.py --config rbk_config.yaml"
echo ""
echo "For GPU execution:"
echo "  gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a"
echo "================================================================================"

exit 0
