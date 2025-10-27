#!/usr/bin/env bash
# H100 cuBLASLt link unblocker: Step 1 -> Step 2 with checks
# Copy/paste as-is from your project root.

set -Eeuo pipefail

echo "==[0] Environment discovery =="
ARCH="${ARCH:-sm_90a}"
CXXSTD="${CXXSTD:-c++17}"

# Prefer explicit CUDA 12.4 if present, else fall back to /usr/local/cuda
if [ -d /usr/local/cuda-12.4 ]; then
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
else
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

# Choose lib dir (lib64 is canonical; targets/.../lib is fallback)
if [ -d "$CUDA_HOME/lib64" ]; then
  LIBDIR="$CUDA_HOME/lib64"
else
  LIBDIR="$CUDA_HOME/targets/x86_64-linux/lib"
fi

echo "CUDA_HOME=${CUDA_HOME}"
echo "LIBDIR=${LIBDIR}"
echo "ARCH=${ARCH}"
echo "CXXSTD=${CXXSTD}"
echo

echo "== nvcc version =="
nvcc --version || true
echo

echo "== library presence =="
ls -l "$LIBDIR"/libcublasLt.so* "$LIBDIR"/libcublas.so* || {
  echo "ERROR: cuBLAS/cuBLASLt .so not found in ${LIBDIR}" >&2
  exit 1
}
echo

# Ensure runtime finds the real libs (and not stubs)
echo "== sanitize LD_LIBRARY_PATH (remove any *stubs* entries) =="
export LD_LIBRARY_PATH="${LIBDIR}:${LD_LIBRARY_PATH-}"
if echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -q '/stubs'; then
  export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/stubs' | paste -sd: -)"
  echo "Removed stubs from LD_LIBRARY_PATH"
fi
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo

###############################################################################
# STEP 1 — Single-step build & link (canonical -L/-l form)
###############################################################################
echo "==[1] Step 1: single-step compile+link with -L/-l (expected to pass) =="
rm -f test_hopper link1.log ldd1.txt || true

# Tip: --use_fast_math + c++17 to match your settings; add -Wl,-rpath for runtime
set +e
time nvcc -arch="${ARCH}" -O3 --use_fast_math --std="${CXXSTD}" \
  flashcore/fast/attention_hopper_minimal.cu flashcore/fast/attention_cublaslt.cu flashcore/cuda/test_hopper_kernel.cu \
  -o build/bin/test_hopper \
  -L"${LIBDIR}" \
  -Wl,-rpath,"${LIBDIR}" \
  -Wl,--no-as-needed -lcublasLt -lcublas -Wl,--as-needed \
  -Xlinker -v |& tee link1.log
rc_step1=$?
set -e

if [ ${rc_step1} -eq 0 ]; then
  echo
  echo "== ldd sanity for Step 1 =="
  ldd ./build/bin/test_hopper | tee ldd1.txt
  if grep -E 'stubs' ldd1.txt >/dev/null; then
    echo "ERROR: linked against STUB libraries — wrong path in link or runtime." >&2
    exit 2
  fi
  if ! grep -E "libcublasLt\.so.*=>.*${LIBDIR}" ldd1.txt >/dev/null; then
    echo "WARNING: libcublasLt is not resolving to ${LIBDIR}. Check rpath/LD_LIBRARY_PATH." >&2
  fi
  echo
  echo "== Smoke run (ignore if your binary expects args) =="
  ./build/bin/test_hopper || true
  echo
  echo "✅ Step 1 SUCCEEDED. You are unblocked."
  exit 0
fi

echo
echo "⚠️  Step 1 failed with rc=${rc_step1}. Proceeding to Step 2 (separate compilation + device link)…"
echo

###############################################################################
# STEP 2 — Separate compilation + device link
###############################################################################
echo "==[2] Step 2: separate compilation & device-link =="
rm -f build/*.o build/dlink.o build/bin/test_hopper link2.log ldd2.txt || true

echo "-- Compile objects with relocatable device code (-dc) --"
time nvcc -arch="${ARCH}" -O3 --use_fast_math --std="${CXXSTD}" -dc \
  flashcore/fast/attention_hopper_minimal.cu -o build/attention_hopper_minimal.o
time nvcc -arch="${ARCH}" -O3 --use_fast_math --std="${CXXSTD}" -dc \
  flashcore/fast/attention_cublaslt.cu -o build/attention_cublaslt.o
time nvcc -arch="${ARCH}" -O3 --use_fast_math --std="${CXXSTD}" -dc \
  flashcore/cuda/test_hopper_kernel.cu -o build/test_hopper_kernel.o

echo "-- Device link stage (-dlink) --"
time nvcc -arch="${ARCH}" -dlink \
  build/attention_hopper_minimal.o build/attention_cublaslt.o build/test_hopper_kernel.o \
  -o build/dlink.o

echo "-- Final host link with cuBLASLt/cuBLAS --"
set +e
time nvcc -arch="${ARCH}" \
  build/attention_hopper_minimal.o build/attention_cublaslt.o build/test_hopper_kernel.o build/dlink.o \
  -o build/bin/test_hopper \
  -L"${LIBDIR}" \
  -Wl,-rpath,"${LIBDIR}" \
  -Wl,--no-as-needed -lcublasLt -lcublas -Wl,--as-needed \
  -Xlinker -v |& tee link2.log
rc_step2=$?
set -e

if [ ${rc_step2} -eq 0 ]; then
  echo
  echo "== ldd sanity for Step 2 =="
  ldd ./build/bin/test_hopper | tee ldd2.txt
  if grep -E 'stubs' ldd2.txt >/dev/null; then
    echo "ERROR: linked against STUB libraries — wrong path in link or runtime." >&2
    exit 3
  fi
  if ! grep -E "libcublasLt\.so.*=>.*${LIBDIR}" ldd2.txt >/dev/null; then
    echo "WARNING: libcublasLt is not resolving to ${LIBDIR}. Check rpath/LD_LIBRARY_PATH." >&2
  fi
  echo
  echo "== Smoke run (ignore if your binary expects args) =="
  ./build/bin/test_hopper || true
  echo
  echo "✅ Step 2 SUCCEEDED. You are unblocked."
  exit 0
fi

echo
echo "❌ Both Step 1 and Step 2 failed (rc2=${rc_step2}). Running a minimal cuBLASLt sanity check…"
echo

###############################################################################
# OPTIONAL — Minimal cuBLASLt sanity check (environment-level)
###############################################################################
cat > test_minimal.cu <<'CU'
#include <cstdio>
#include <cublasLt.h>
int main() {
  cublasLtHandle_t h{};
  cublasStatus_t s = cublasLtCreate(&h);
  if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublasLtCreate failed: %d\n", (int)s); return 1; }
  cublasLtDestroy(h);
  std::puts("cublasLtCreate OK");
  return 0;
}
CU

set +e
time nvcc -arch="${ARCH}" -O3 --std="${CXXSTD}" test_minimal.cu -o test_minimal \
  -L"${LIBDIR}" -Wl,-rpath,"${LIBDIR}" -lcublasLt -Xlinker -v |& tee link_min.log
rc_min=$?
set -e

if [ ${rc_min} -ne 0 ]; then
  echo "ENVIRONMENT ERROR: minimal cuBLASLt link failed. Inspect link_min.log for missing/incorrect libs." >&2
  echo "Check for competing CUDA installs, stubs in path, or ABI/toolkit mismatch." >&2
  exit 4
else
  echo "Minimal cuBLASLt link succeeded — issue is likely in build/link flags for the full target." >&2
  ./test_minimal
  exit 5
fi

