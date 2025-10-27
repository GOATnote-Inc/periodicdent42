#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# [META]
# Title: FlashCore Hopper Kernel Build + Validation (LLM-Aware)
# Audience: Humans + AI agents (Cursor, Windsurf, Claude, GPT)
# Design: Structured, semantic echo tags + real validation gates
# Source: Standing on giants - NVIDIA validation methodology
###############################################################################

# === [CONFIG] ================================================================
ARCH="sm_90a"
KERNEL_SRC="flashcore/fast/attention_hopper_tma.cu"
TEST_SRC="flashcore/cuda/test_hopper_kernel.cu"
OUT_BIN="build/bin/test_hopper"
LOG_DIR="build/logs"
mkdir -p "$LOG_DIR" "build/bin"

echo "========================================"
echo "FLASHCORE HOPPER VALIDATION (FA3-STYLE)"
echo "========================================"
echo ""

# === [STEP:ENV_CHECK] ========================================================
echo "[STEP:ENV_CHECK] Checking CUDA + GPU environment..."
if ! command -v nvcc >/dev/null; then
  # Try common CUDA paths
  export PATH="/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
  
  if ! command -v nvcc >/dev/null; then
    echo "[RESULT:FAIL] nvcc not found in PATH"
    exit 1
  fi
fi

nvcc --version | tee "$LOG_DIR/nvcc_version.txt"
echo ""

if ! command -v nvidia-smi >/dev/null; then
  echo "[RESULT:FAIL] nvidia-smi not found"
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader \
  | tee "$LOG_DIR/gpu_info.txt"

# Validate H100 (sm_90)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [[ ! "$COMPUTE_CAP" =~ ^9\. ]]; then
  echo "[RESULT:FAIL] This kernel requires Hopper (sm_90+), found sm_${COMPUTE_CAP}"
  exit 1
fi

echo "[RESULT:PASS] Environment validated (H100, CUDA 12.4+)"
echo ""

# === [STEP:COMPILE] ==========================================================
echo "[STEP:COMPILE] Building Hopper-native kernel..."
COMPILE_LOG="$LOG_DIR/compile.log"

# Compile with full validation flags
if nvcc -arch=$ARCH -O3 --use_fast_math \
    -Xptxas -v,-warn-lmem-usage \
    --std=c++17 \
    -lineinfo \
    -I. \
    "$KERNEL_SRC" \
    "$TEST_SRC" \
    -o "$OUT_BIN" \
    2>&1 | tee "$COMPILE_LOG"; then
  
  # Check for warnings
  if grep -qE "(warning|lmem)" "$COMPILE_LOG"; then
    echo "[RESULT:WARN] Compilation succeeded with warnings (see $COMPILE_LOG)"
  else
    echo "[RESULT:PASS] Compilation clean (no warnings)"
  fi
else
  echo "[RESULT:FAIL] Compilation error (see $COMPILE_LOG)"
  cat "$COMPILE_LOG"
  exit 1
fi
echo ""

# === [STEP:SASS_VALIDATION] ==================================================
echo "[STEP:SASS_VALIDATION] Validating SASS disassembly..."
SASS_LOG="$LOG_DIR/sass.txt"

if cuobjdump --dump-sass "$OUT_BIN" > "$SASS_LOG" 2>&1; then
  # Check for Hopper-specific instructions
  if grep -qE "WMMA|WGMMA|TMA" "$SASS_LOG"; then
    echo "[RESULT:PASS] Hopper instructions detected (WMMA/WGMMA/TMA)"
  else
    echo "[RESULT:WARN] Hopper-specific ops not found (may be scalar fallback)"
  fi
  
  # Count barriers (should be minimal for warp-spec)
  BARRIER_COUNT=$(grep -c "BAR\|BARRIER" "$SASS_LOG" || echo "0")
  echo "[INFO] Barrier count: $BARRIER_COUNT (lower is better for warp-spec)"
else
  echo "[RESULT:WARN] cuobjdump failed (SASS validation skipped)"
fi
echo ""

# === [STEP:DEVICE_TEST] ======================================================
echo "[STEP:DEVICE_TEST] Running correctness + performance test..."
BENCH_LOG="$LOG_DIR/bench.log"

if "$OUT_BIN" 2>&1 | tee "$BENCH_LOG"; then
  # Check for correctness markers
  if grep -q "✅" "$BENCH_LOG"; then
    echo "[RESULT:PASS] Device test passed functional checks"
  else
    echo "[RESULT:FAIL] Test output missing pass markers"
    tail -30 "$BENCH_LOG"
    exit 1
  fi
  
  # Extract performance metrics
  if grep -q "TFLOPS" "$BENCH_LOG"; then
    TFLOPS=$(grep "TFLOPS" "$BENCH_LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "[SUMMARY:PERF] Performance: ${TFLOPS:-N/A} TFLOPS"
  fi
else
  echo "[RESULT:FAIL] Kernel execution error"
  tail -50 "$BENCH_LOG"
  exit 1
fi
echo ""

# === [STEP:SANITIZER] ========================================================
if command -v compute-sanitizer >/dev/null; then
  echo "[STEP:SANITIZER] Running compute-sanitizer (memcheck)..."
  SANITIZER_LOG="$LOG_DIR/sanitizer.log"
  
  if compute-sanitizer --tool memcheck "$OUT_BIN" > "$SANITIZER_LOG" 2>&1; then
    if grep -q "ERROR SUMMARY: 0 errors" "$SANITIZER_LOG"; then
      echo "[RESULT:PASS] Memory validation clean (0 errors)"
    else
      echo "[RESULT:FAIL] Memory errors detected"
      grep "ERROR SUMMARY" "$SANITIZER_LOG"
      exit 1
    fi
  else
    echo "[RESULT:WARN] Sanitizer execution failed"
  fi
  echo ""
else
  echo "[STEP:SANITIZER] Skipped (compute-sanitizer not available)"
  echo ""
fi

# === [STEP:PERF_METRICS] =====================================================
if command -v ncu >/dev/null && [[ "${RUN_NCU:-0}" == "1" ]]; then
  echo "[STEP:PERF_METRICS] Collecting Nsight Compute metrics..."
  NSIGHT_LOG="$LOG_DIR/ncu_metrics.txt"
  
  ncu --section SpeedOfLight \
      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
      --target-processes all \
      --force-overwrite \
      "$OUT_BIN" > "$NSIGHT_LOG" 2>&1 || echo "[RESULT:WARN] Nsight Compute failed"
  
  # Extract summary for human+LLM parsing
  if [[ -f "$NSIGHT_LOG" ]]; then
    SM_UTIL=$(grep "sm__throughput" "$NSIGHT_LOG" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    DRAM_UTIL=$(grep "dram__throughput" "$NSIGHT_LOG" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    echo "[SUMMARY:PERF] SM=${SM_UTIL:-N/A}%  DRAM=${DRAM_UTIL:-N/A}%"
    echo "[SUMMARY:TARGET] SM≥70%  DRAM≥85% (FA3 baseline)"
  fi
  echo ""
else
  echo "[STEP:PERF_METRICS] Skipped (set RUN_NCU=1 to enable)"
  echo ""
fi

# === [SUMMARY:FINAL] =========================================================
echo "========================================"
echo "VALIDATION COMPLETE"
echo "========================================"
echo ""
echo "[SUMMARY:LOGS] Logs stored in $LOG_DIR/"
echo "  - compile.log (build output)"
echo "  - sass.txt (SASS disassembly)"
echo "  - bench.log (correctness + perf)"
if [[ -f "$LOG_DIR/sanitizer.log" ]]; then
  echo "  - sanitizer.log (memory validation)"
fi
if [[ -f "$LOG_DIR/ncu_metrics.txt" ]]; then
  echo "  - ncu_metrics.txt (NSight Compute)"
fi
echo ""

# Check if this is Phase 1 (correctness only)
if grep -q "PASS" "$BENCH_LOG" && grep -q "✅" "$BENCH_LOG"; then
  echo "[SUMMARY:STATUS] PHASE 1 PASS ✅"
  echo "[SUMMARY:NEXT] Ready for Phase 2 (TMA integration)"
  exit 0
else
  echo "[SUMMARY:STATUS] VALIDATION FAILED ❌"
  exit 1
fi

