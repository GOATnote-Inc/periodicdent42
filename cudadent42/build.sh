#!/usr/bin/env bash
set -euo pipefail

# CUDAdent42 Build Script (CMake-based)
# Usage: ./build.sh [clean]

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  CUDAdent42: Automated Build (CMake)                                  ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if clean requested
if [[ "${1:-}" == "clean" ]]; then
    echo "🧹 Cleaning previous build..."
    rm -rf build flashmoe_science/_C*.so
    echo "✅ Clean complete"
    echo ""
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Configure with CMake"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Build"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Build with parallel jobs
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build . --parallel ${NPROC}

cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Verify"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if .so exists
SO_FILE=$(ls flashmoe_science/_C*.so 2>/dev/null | head -n1 || echo "")
if [[ -n "${SO_FILE}" ]]; then
    echo "✅ Build successful!"
    echo ""
    echo "Output: ${SO_FILE}"
    echo "Size: $(du -h ${SO_FILE} | cut -f1)"
    echo ""
    
    # Check symbols
    echo "Symbols:"
    nm ${SO_FILE} | grep -E '(flash_attention_forward|PyInit)' || echo "  (nm not available)"
    echo ""
else
    echo "❌ Build failed - no .so file found"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Build Complete!                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Test with:"
echo "  python3 -c 'import sys; sys.path.insert(0, \".\"); import flashmoe_science._C as m; print(m)'"
echo ""
echo "Or run:"
echo "  python3 tests/test_basic.py"

