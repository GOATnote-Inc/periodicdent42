#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-artifacts}"
TS=$(date -u +"%Y%m%dT%H%M%SZ")
RUN_DIR="${OUT_ROOT}/run_${TS}"
mkdir -p "${RUN_DIR}"

# 1) Environment & toolchain snapshot
{
  echo "=== ENV ==="
  uname -a || true
  nvidia-smi || true
  nvcc --version || true
  python3 - <<'PY'
import sys, platform, torch
print("Python :", sys.version.replace("\n"," "))
print("Platform:", platform.platform())
try:
  import triton, triton.language as tl
  print("Triton  :", triton.__version__)
except Exception as e:
  print("Triton  : import failed:", e)
print("PyTorch :", torch.__version__)
PY
} | tee "${RUN_DIR}/env.txt"

# 2) Force Triton to dump IR/PTX for this run only
export TRITON_KERNEL_DUMP=1
export TRITON_KERNEL_DUMP_DIR="${RUN_DIR}/dump"

# 3) Run the Python test (provided below)
python3 scripts/tma_user_repro.py | tee "${RUN_DIR}/stdout.txt"

# 4) Canonical greps over **Triton IR** (not just PTX)
#    (We search both .ttir and .ttgir; names may vary by build.)
shopt -s globstar nullglob
TTIR_FILES=("${RUN_DIR}"/dump/**/*.ttir)
TTGIR_FILES=("${RUN_DIR}"/dump/**/*.ttgir)
PTX_FILES=("${RUN_DIR}"/dump/**/*.ptx)

> "${RUN_DIR}/grep_summary.txt"
for f in "${TTIR_FILES[@]}" "${TTGIR_FILES[@]}" "${PTX_FILES[@]}"; do
  [ -f "$f" ] && {
    CNT_ASYNC=$(grep -c -E "async_tma|ttng\.async_tma" "$f" || true)
    CNT_CP=$(grep -c -E "cp\.async\.bulk\.tensor" "$f" || true)
    echo "$(realpath "$f")  async_tma:${CNT_ASYNC}  cp.async.bulk.tensor:${CNT_CP}" >> "${RUN_DIR}/grep_summary.txt"
  }
done

# 5) Checksums: reviewers can verify you didn't edit artifacts
( cd "${RUN_DIR}" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
echo "Artifacts in ${RUN_DIR}"

