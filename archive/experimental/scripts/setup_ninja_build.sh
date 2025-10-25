#!/bin/bash
# Setup script to enable Ninja for PyTorch JIT compilation
# Ensures fast parallel builds for CUDA kernels

set -euo pipefail

echo "================================================================================"
echo "Ninja Build System Setup for PyTorch CUDA Extensions"
echo "================================================================================"
echo ""

# ============================================================================
# 1) Check if Ninja is installed
# ============================================================================
echo "1) Checking Ninja Installation"
echo "--------------------------------------------------------------------------------"

if command -v ninja &> /dev/null; then
    NINJA_VERSION=$(ninja --version 2>/dev/null || echo "unknown")
    echo "✓ Ninja found: $(which ninja)"
    echo "  Version: ${NINJA_VERSION}"
elif python3 -c "import ninja" &> /dev/null; then
    NINJA_VERSION=$(python3 -c "import ninja; print(ninja.__version__)")
    echo "✓ Ninja Python package found"
    echo "  Version: ${NINJA_VERSION}"
else
    echo "✗ Ninja not found"
    echo ""
    echo "Install Ninja:"
    echo "  • Mac:    brew install ninja"
    echo "  • Ubuntu: sudo apt-get install ninja-build"
    echo "  • Pip:    pip3 install ninja"
    echo ""
    exit 1
fi

# ============================================================================
# 2) Export environment variables for PyTorch
# ============================================================================
echo ""
echo "2) Setting PyTorch Build Environment Variables"
echo "--------------------------------------------------------------------------------"

# Enable Ninja
export USE_NINJA=1
echo "✓ USE_NINJA=1"

# Set parallel jobs (use all CPU cores)
MAX_JOBS=$(python3 -c "import os; print(os.cpu_count() or 4)")
export MAX_JOBS
echo "✓ MAX_JOBS=${MAX_JOBS}"

# ============================================================================
# 3) Verify PyTorch can use Ninja
# ============================================================================
echo ""
echo "3) Verifying PyTorch Ninja Integration"
echo "--------------------------------------------------------------------------------"

TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
echo "PyTorch extensions cache: ${TORCH_EXTENSIONS_DIR}"

# Quick probe to see if Ninja is available to PyTorch
python3 - <<'PYEOF'
import torch
from torch.utils.cpp_extension import CUDA_HOME, _get_build_directory

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA Home: {CUDA_HOME}")
    
# Check for existing build.ninja files
import pathlib
root = pathlib.Path.home() / ".cache" / "torch_extensions"
if root.exists():
    ninja_files = list(root.glob("**/build.ninja"))
    if ninja_files:
        print(f"✓ Found {len(ninja_files)} existing build.ninja file(s)")
        for f in ninja_files[:3]:
            print(f"  - {f}")
    else:
        print("ℹ️  No existing build.ninja files (will be created on next JIT compile)")
else:
    print("ℹ️  No torch_extensions cache yet (will be created on first JIT compile)")
PYEOF

# ============================================================================
# 4) Generate environment script
# ============================================================================
echo ""
echo "4) Generating Environment Script"
echo "--------------------------------------------------------------------------------"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_SCRIPT="${SCRIPT_DIR}/ninja_env.sh"

cat > "${ENV_SCRIPT}" <<'ENVEOF'
#!/bin/bash
# Source this script before building/running PyTorch CUDA kernels
export USE_NINJA=1
export MAX_JOBS=$(python3 -c "import os; print(os.cpu_count() or 4)")
echo "✓ Ninja build environment configured (USE_NINJA=1, MAX_JOBS=${MAX_JOBS})"
ENVEOF

chmod +x "${ENV_SCRIPT}"
echo "✓ Environment script created: ${ENV_SCRIPT}"
echo ""
echo "Usage: source ${ENV_SCRIPT}"

# ============================================================================
# 5) Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "✅ Ninja Setup Complete"
echo "================================================================================"
echo ""
echo "To use Ninja for PyTorch CUDA kernel builds:"
echo "  export USE_NINJA=1"
echo "  export MAX_JOBS=${MAX_JOBS}"
echo ""
echo "Or simply:"
echo "  source ${ENV_SCRIPT}"
echo ""
echo "To verify Ninja is being used:"
echo "  1. Look for 'ninja' in JIT compilation logs"
echo "  2. Check for build.ninja files in ~/.cache/torch_extensions/"
echo "  3. Compilation should be significantly faster"
echo "================================================================================"

exit 0

