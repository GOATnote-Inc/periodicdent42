#!/bin/bash
# CUTLASS 4.3 + CUDA 13.0 STRICT BUILD (H100 / sm_90a)
# No heuristics, no auto-fix by tools. Literal execution only.

set -euo pipefail

banner() { echo -e "\n========================================================================\n$1\n========================================================================\n"; }

banner "STRICT-EXECUTION MODE: H100 (sm_90a) + cuBLAS + Release"

# -------- ENV LOCK --------
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/compat:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# IMPORTANT: H100 needs 90a
export TORCH_CUDA_ARCH_LIST="9.0a"
export CUDAARCHS="90a"

echo "CUDA_HOME:            $CUDA_HOME"
echo "nvcc version:         $(nvcc --version | awk -F'V' '/release/ {print $2}')"
echo "Target Architecture:  sm_90a (H100)"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "CUDAARCHS:            $CUDAARCHS"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)
if [[ -z "${GPU_NAME}" || "$GPU_NAME" != *"H100"* ]]; then
  echo "❌ FATAL: Expected H100, found: ${GPU_NAME:-UNKNOWN}"; exit 1
fi
echo "✅ GPU Verification: $GPU_NAME"

# -------- JOBS (avoid thrash) --------
MEM_GB=$(awk '/MemTotal/ {printf "%.0f",$2/1024/1024}' /proc/meminfo)
CPU_CORES=$(nproc)
# Rule of thumb: one NVCC TU can peak ~2–4GB; cap parallelism by memory
if (( MEM_GB <= 32 ));    then JOBS=4
elif (( MEM_GB <= 64 ));  then JOBS=8
elif (( MEM_GB <= 128 )); then JOBS=16
else JOBS=$CPU_CORES; fi
JOBS=${JOBS:-8}
echo "Build parallelism (JOBS): $JOBS (Mem=${MEM_GB}GB, Cores=${CPU_CORES})"

# -------- CLEAN --------
banner "SECTION 2: Clean Build Dir"
cd /opt/cutlass
rm -rf build_release
mkdir -p build_release
cd build_release
echo "✅ Clean build directory created"

# -------- CMAKE (force 90a) --------
banner "SECTION 3: CMake Configure (Release + cuBLAS + Profiler)"
echo "Config:"
echo "  CMAKE_BUILD_TYPE=Release"
echo "  CUTLASS_ENABLE_CUBLAS=ON"
echo "  CUTLASS_ENABLE_PROFILER=ON"
echo "  CUTLASS_TEST_LEVEL=2"
echo "  CUTLASS_NVCC_ARCHS=90a"
echo "  CMAKE_CUDA_ARCHITECTURES=90a"
echo ""

# Fallback flags add -arch and two gencodes to guarantee 90a real + 90 PTX
CMAKE_FALLBACK_FLAGS="-lineinfo -Xptxas=-v -arch=sm_90a \
  --generate-code=arch=compute_90,code=sm_90a \
  --generate-code=arch=compute_90,code=compute_90"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_ENABLE_CUBLAS=ON \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_TEST_LEVEL=2 \
  -DCUTLASS_NVCC_ARCHS="90a" \
  -DCMAKE_CUDA_ARCHITECTURES="90a" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_FLAGS="$CMAKE_FALLBACK_FLAGS" \
  | tee /tmp/cutlass_cmake.log

echo "✅ CMake configuration complete"

# -------- BUILD: example 88 FIRST (fast), profiler AFTER (optional) --------
banner "SECTION 4A: Build Example 88 (fast)"
# stream output so "stall" is visible
set -o pipefail
make -j${JOBS} 88_hopper_fmha 2>&1 | tee /tmp/cutlass_build_88.log

# Verify artifact exists
if [[ ! -f examples/88_hopper_fmha/88_hopper_fmha ]]; then
  echo "❌ FATAL: 88_hopper_fmha missing"; tail -50 /tmp/cutlass_build_88.log; exit 1
fi

# Robust arch check
banner "SECTION 5: Binary Arch Check (expect sm_90a)"
if cuobjdump --dump-sass examples/88_hopper_fmha/88_hopper_fmha | head -50 | grep -q "arch = sm_90a"; then
  echo "✅ Binary contains SASS for sm_90a"
else
  echo "❌ FATAL: No sm_90a SASS found"; cuobjdump --dump-sass examples/88_hopper_fmha/88_hopper_fmha | head -80; exit 1
fi

# -------- QUICK VERIFY (small case) --------
banner "SECTION 6: Correctness (small case)"
pushd examples/88_hopper_fmha >/dev/null
./88_hopper_fmha --b=1 --h=8 --q=512 --k=512 --d=64 --iterations=5 --verify | tee /tmp/verify.log
if grep -q "ERROR" /tmp/verify.log; then
  echo "❌ Verification failed"; exit 1
fi
echo "✅ Small-case verification passed"
popd >/dev/null

# -------- BUILD PROFILER (heavy, can be toggled) --------
BUILD_PROFILER=${BUILD_PROFILER:-1}
if [[ "$BUILD_PROFILER" == "1" ]]; then
  banner "SECTION 4B: Build CUTLASS Profiler (heavy)"
  make -j${JOBS} cutlass_profiler 2>&1 | tee /tmp/cutlass_build_profiler.log
  [[ -f tools/profiler/cutlass_profiler ]] || { echo "❌ Profiler missing"; exit 1; }
  echo "✅ Profiler present: tools/profiler/cutlass_profiler"
else
  echo "⏭️  Skipping profiler build (BUILD_PROFILER=0)"
fi

# -------- TARGET BASELINE --------
banner "SECTION 7: Target Workload Baseline (B=16 H=96 Q=4096 D=128)"
./examples/88_hopper_fmha/88_hopper_fmha \
  --b=16 --h=96 --q=4096 --k=4096 --d=128 --iterations=100 \
  | tee /tmp/baseline.log

banner "BUILD + BASELINE COMPLETE"
echo "Artifacts:"
echo "  - 88 binary : /opt/cutlass/build_release/examples/88_hopper_fmha/88_hopper_fmha"
[[ "$BUILD_PROFILER" == "1" ]] && echo "  - profiler  : /opt/cutlass/build_release/tools/profiler/cutlass_profiler"
echo "  - cmake log : /tmp/cutlass_cmake.log"
echo "  - build 88  : /tmp/cutlass_build_88.log"
[[ "$BUILD_PROFILER" == "1" ]] && echo "  - build prof: /tmp/cutlass_build_profiler.log"
echo "  - verify    : /tmp/verify.log"
echo "  - baseline  : /tmp/baseline.log"

