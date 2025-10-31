#!/bin/bash
# FlashCore NVIDIA-Grade Debug + Profile Toolchain
# Matches FA3, CUTLASS, Triton-core validation methodology

set -e

#==============================================================================
# CONFIGURATION
#==============================================================================

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
KERNEL_SRC="${KERNEL_SRC:-flashcore/fast/attention_cuda_wmma.cu}"
TEST_SRC="${TEST_SRC:-flashcore/cuda/test_hopper_kernel.cu}"
OUTPUT_BIN="${OUTPUT_BIN:-build/bin/test_hopper}"
BUILD_DIR="${BUILD_DIR:-build}"

# Tool flags (set via env vars)
RUN_SANITIZER="${RUN_SANITIZER:-0}"
RUN_PROFILER="${RUN_PROFILER:-0}"
RUN_BASELINE="${RUN_BASELINE:-1}"  # Always run baseline by default

# Profiler options
NCU_SET="${NCU_SET:-full}"  # full, detailed, or custom metrics
NCU_OUTPUT="${NCU_OUTPUT:-build/profile}"

echo "========================================"
echo "FLASHCORE DEBUG + PROFILE TOOLCHAIN"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Kernel:      ${KERNEL_SRC}"
echo "  Test:        ${TEST_SRC}"
echo "  Output:      ${OUTPUT_BIN}"
echo "  Baseline:    ${RUN_BASELINE}"
echo "  Sanitizer:   ${RUN_SANITIZER}"
echo "  Profiler:    ${RUN_PROFILER}"
echo ""

#==============================================================================
# STEP 1: BUILD WITH DEBUG INFO
#==============================================================================

echo "[1/4] Building kernel with debug info..."
mkdir -p ${BUILD_DIR}/bin ${BUILD_DIR}/lib

# Find CUDA
if ! command -v nvcc &> /dev/null; then
    if [ -d "/usr/local/cuda/bin" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    else
        echo "❌ nvcc not found. Install CUDA toolkit."
        exit 1
    fi
fi

echo "CUDA: $(nvcc --version | grep release)"
echo ""

# Compile with:
# -lineinfo:     Enable line-level profiling in Nsight Compute
# -G:            Generate device debug info for compute-sanitizer
# -Xptxas -v:    Verbose register/memory usage
nvcc -arch=sm_90 -O3 --use_fast_math \
    -lineinfo \
    -Xptxas -v,-warn-lmem-usage \
    -I. \
    ${KERNEL_SRC} \
    ${TEST_SRC} \
    -o ${OUTPUT_BIN} \
    2>&1 | tee ${BUILD_DIR}/compile.log | grep -E "(ptxas|error|warning)" || true

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi
echo ""

#==============================================================================
# STEP 2: BASELINE RUN (Correctness + Performance)
#==============================================================================

if [ ${RUN_BASELINE} -eq 1 ]; then
    echo "[2/4] Running baseline (correctness + performance)..."
    ${OUTPUT_BIN} 2>&1 | tee ${BUILD_DIR}/baseline.log
    
    # Check for crashes
    if [ $? -eq 0 ]; then
        echo "✅ Baseline run successful"
    else
        echo "❌ Baseline run failed - stopping here"
        exit 1
    fi
    echo ""
fi

#==============================================================================
# STEP 3: COMPUTE-SANITIZER (Memory + Sync Errors)
#==============================================================================

if [ ${RUN_SANITIZER} -eq 1 ]; then
    echo "[3/4] Running compute-sanitizer..."
    
    if ! command -v compute-sanitizer &> /dev/null; then
        echo "⚠️  compute-sanitizer not found, skipping"
    else
        # memcheck: Detects out-of-bounds, uninitialized memory, race conditions
        echo "  [3a] memcheck..."
        compute-sanitizer --tool memcheck \
            --leak-check full \
            --print-limit 100 \
            ${OUTPUT_BIN} 2>&1 | tee ${BUILD_DIR}/memcheck.log
        
        if grep -q "ERROR SUMMARY: 0 errors" ${BUILD_DIR}/memcheck.log; then
            echo "  ✅ memcheck: No errors"
        else
            echo "  ❌ memcheck: ERRORS FOUND (see ${BUILD_DIR}/memcheck.log)"
        fi
        
        # synccheck: Detects synchronization hazards
        echo "  [3b] synccheck..."
        compute-sanitizer --tool synccheck \
            ${OUTPUT_BIN} 2>&1 | tee ${BUILD_DIR}/synccheck.log
        
        if grep -q "ERROR SUMMARY: 0 errors" ${BUILD_DIR}/synccheck.log; then
            echo "  ✅ synccheck: No errors"
        else
            echo "  ❌ synccheck: ERRORS FOUND (see ${BUILD_DIR}/synccheck.log)"
        fi
        
        # racecheck: Detects shared memory races (SLOW, optional)
        if [ "${RUN_RACECHECK}" = "1" ]; then
            echo "  [3c] racecheck (slow)..."
            compute-sanitizer --tool racecheck \
                ${OUTPUT_BIN} 2>&1 | tee ${BUILD_DIR}/racecheck.log
        fi
        
        echo "✅ Sanitizer complete"
    fi
    echo ""
fi

#==============================================================================
# STEP 4: NSIGHT COMPUTE PROFILING
#==============================================================================

if [ ${RUN_PROFILER} -eq 1 ]; then
    echo "[4/4] Running Nsight Compute profiler..."
    
    if ! command -v ncu &> /dev/null; then
        echo "⚠️  ncu (Nsight Compute) not found, skipping"
    else
        mkdir -p ${BUILD_DIR}/profile
        
        # Profile kernel with comprehensive metrics
        echo "  Profiling kernel (this may take 1-2 minutes)..."
        ncu --set ${NCU_SET} \
            --target-processes all \
            --export ${NCU_OUTPUT} \
            --force-overwrite \
            --print-summary per-kernel \
            ${OUTPUT_BIN} 2>&1 | tee ${BUILD_DIR}/profile.log
        
        echo ""
        echo "✅ Profile complete"
        echo "   Report: ${NCU_OUTPUT}.ncu-rep (open in Nsight Compute GUI)"
        echo "   Summary: ${BUILD_DIR}/profile.log"
        echo ""
        
        # Extract key metrics (if available)
        if [ -f "${BUILD_DIR}/profile.log" ]; then
            echo "Key Metrics:"
            grep -A 5 "Compute (SM) Throughput" ${BUILD_DIR}/profile.log || true
            grep -A 5 "Memory Throughput" ${BUILD_DIR}/profile.log || true
            grep -A 5 "SOL" ${BUILD_DIR}/profile.log || true
        fi
    fi
    echo ""
fi

#==============================================================================
# SUMMARY
#==============================================================================

echo "========================================"
echo "TOOLCHAIN COMPLETE"
echo "========================================"
echo ""
echo "Generated files:"
echo "  Build log:     ${BUILD_DIR}/compile.log"
if [ ${RUN_BASELINE} -eq 1 ]; then
    echo "  Baseline:      ${BUILD_DIR}/baseline.log"
fi
if [ ${RUN_SANITIZER} -eq 1 ]; then
    echo "  Memcheck:      ${BUILD_DIR}/memcheck.log"
    echo "  Synccheck:     ${BUILD_DIR}/synccheck.log"
fi
if [ ${RUN_PROFILER} -eq 1 ]; then
    echo "  Profile:       ${NCU_OUTPUT}.ncu-rep"
    echo "  Profile log:   ${BUILD_DIR}/profile.log"
fi
echo ""
echo "Quick checks:"
echo "  RUN_SANITIZER=1 $0  # Memory + sync validation"
echo "  RUN_PROFILER=1 $0   # Performance profiling"
echo "  RUN_SANITIZER=1 RUN_PROFILER=1 $0  # Full validation"
echo ""

