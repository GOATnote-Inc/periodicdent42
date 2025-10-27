#!/bin/bash
# debug_cublaslt_nan.sh
# Systematic debugging of cuBLASLt NaN/Inf issue
# Based on expert analysis and Status Report Oct 27

set -e

REMOTE_IP="154.57.34.90"
REMOTE_PORT="14727"
REMOTE_USER="root"
REMOTE_DIR="/workspace/flashcore_hopper"

echo "🔍 FlashCore cuBLASLt NaN/Inf Debugging"
echo "========================================"
echo ""

# Step 1: Run with compute-sanitizer
echo "📌 Step 1: Memory Safety Check (compute-sanitizer)"
echo "---------------------------------------------------"
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_IP} << 'ENDSSH'
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export FLASHCORE_CUBLASLT_WS_MB=256
cd /workspace/flashcore_hopper

echo "Running compute-sanitizer memcheck..."
compute-sanitizer --tool memcheck --print-limit 10 ./build/bin/test_hopper 2>&1 | tee sanitizer_memcheck.log | head -100

echo ""
echo "Checking for memory errors..."
grep -i "error\|invalid\|out-of-bounds" sanitizer_memcheck.log || echo "✅ No memory errors detected"
ENDSSH

echo ""
echo "---------------------------------------------------"
echo ""

# Step 2: Run with detailed cuBLASLt logging
echo "📌 Step 2: cuBLASLt Error Code Validation"
echo "---------------------------------------------------"
echo "TODO: Add cuBLASLt error checking to kernel (see STATUS_OCT27_EVENING.md)"
echo "  - Check cublasLtMatmul return codes"
echo "  - Add NaN checks after each GEMM"
echo "  - Print intermediate results"
echo ""

# Step 3: Profile with Nsight Compute
echo "📌 Step 3: Nsight Compute Profile (SKIPPED - takes 10+ min)"
echo "---------------------------------------------------"
echo "To run manually:"
echo "  ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_IP}"
echo "  cd ${REMOTE_DIR}"
echo "  ncu --set full --target-processes all ./build/bin/test_hopper > ncu_profile.txt"
echo ""

# Step 4: Compare with PyTorch SDPA
echo "📌 Step 4: Reference Comparison (PyTorch SDPA)"
echo "---------------------------------------------------"
echo "TODO: Run PyTorch SDPA on same random inputs, compare outputs"
echo "  - Generate deterministic test inputs"
echo "  - Run both kernels"
echo "  - Check element-wise diff"
echo ""

# Step 5: Matrix size experiments
echo "📌 Step 5: Matrix Size Experiments (Q@K^T Workspace)"
echo "---------------------------------------------------"
echo "Hypothesis: Q@K^T matrix too small for Tensor Core algos"
echo ""
echo "Current: S=2048, D=64 → Q@K^T is (2048×64) @ (64×128)"
echo ""
echo "Experiments to try:"
echo "  A. S=4096 (double sequence length)"
echo "  B. S=8192 (4× sequence length)"
echo "  C. D=128 (double head dimension)"
echo ""
echo "TODO: Modify test_hopper_kernel.cu to test these configs"
echo ""

# Summary
echo "========================================"
echo "📋 SUMMARY"
echo "========================================"
echo ""
echo "✅ Completed: compute-sanitizer memcheck"
echo "⏳ Pending:   cuBLASLt error code checks (needs kernel instrumentation)"
echo "⏳ Pending:   Nsight Compute profile (10+ min, run manually if needed)"
echo "⏳ Pending:   PyTorch SDPA comparison (needs test harness)"
echo "⏳ Pending:   Matrix size experiments (needs code changes)"
echo ""
echo "📄 Full status: docs/STATUS_OCT27_EVENING.md"
echo "📁 Logs: ${REMOTE_DIR}/sanitizer_memcheck.log"
echo ""
echo "🎯 RECOMMENDATION: Review sanitizer output first!"
echo "   If no memory errors, add cuBLASLt error checking to kernel."
echo ""

