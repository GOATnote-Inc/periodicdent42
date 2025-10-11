#!/bin/bash
# FlashMoE-Science: Build and Test Script
# Run this on machine with CUDA GPU

set -e

echo "🔨 Building CUDA extensions..."
python setup.py build_ext --inplace

echo ""
echo "✅ Build complete!"
echo ""
echo "🧪 Running tests..."
echo ""

# Run specific test for small sequence first (most likely to pass)
pytest tests/test_attention_correctness.py::TestFlashAttentionCorrectness::test_forward_vs_pytorch -v -k "128-64"

echo ""
echo "📊 Test Results Summary:"
echo "If test passed: ✅ Basic tiling implementation works!"
echo "If test failed: Review error messages and debug"
echo ""
echo "Next steps:"
echo "  1. If passing: Run full test suite"
echo "  2. Profile with: ncu --set full python benchmarks/attention_benchmarks.py"
echo "  3. Continue to Day 4-6: Online softmax"

