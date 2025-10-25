#!/usr/bin/env bash
set -euo pipefail

# --- 0) Setup unique dump dir (prevents reuse of old IR) ---------------------
ROOT="${PWD}/tma-verify-$(date -u +%Y%m%dT%H%M%SZ)"
DUMP="${ROOT}/dump"
ART="${ROOT}/artifacts"
mkdir -p "$DUMP" "$ART"/{env,ir,logs}

export TRITON_KERNEL_DUMP=1
export TRITON_KERNEL_DUMP_DIR="$DUMP"

# --- 1) Environment snapshot (auditable context) -----------------------------
{
  echo "UTC: $(date -u +%F' '%T)"; echo
  nvidia-smi -L || true
  echo
  python3 - <<'PY'
import sys, torch
try:
  import triton
  print("Python:", sys.version.replace("\n"," "))
  print("Torch:", torch.__version__)
  print("Triton:", triton.__version__)
except Exception as e:
  print("EnvError:", e)
PY
  echo
  nvcc --version || true
} | tee "$ART/env/snapshot.txt" >/dev/null

# --- 2) Kernels: positive / negative / stress (3.0.x block pointers) ---------
cat > "$ROOT/10_emit_positive.py" <<'PY'
import torch, triton, triton.language as tl
@triton.jit
def copy_bp(X, Y, M: tl.constexpr, N: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    off_m = pid * BM
    x_blk = tl.make_block_ptr(X, (M, N), (N, 1), (off_m, 0), (BM, BN), (1, 0))
    y_blk = tl.make_block_ptr(Y, (M, N), (N, 1), (off_m, 0), (BM, BN), (1, 0))
    x = tl.load(x_blk, boundary_check=(0,1))
    tl.store(y_blk, x, boundary_check=(0,1))
def run():
    M,N=4096,4096; BM,BN=128,128
    x = torch.randn((M,N), device="cuda", dtype=torch.float16)
    y = torch.zeros_like(x)
    grid = (triton.cdiv(M,BM),)
    copy_bp[grid](x,y,M,N,BM,BM,num_stages=4,num_warps=8)
    torch.cuda.synchronize()
if __name__=="__main__": run()
PY

cat > "$ROOT/11_emit_negative.py" <<'PY'
# Flip an invariant (order) so TMA should NOT appear.
import torch, triton, triton.language as tl
@triton.jit
def copy_bp_bad(X, Y, M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    off_m = pid * BM
    # order=(0,1) intentionally wrong for TMA emission
    x_blk = tl.make_block_ptr(X, (M, N), (N, 1), (off_m, 0), (BM, BN), (0, 1))
    y_blk = tl.make_block_ptr(Y, (M, N), (N, 1), (off_m, 0), (BM, BN), (0, 1))
    x = tl.load(x_blk, boundary_check=(0,1))
    tl.store(y_blk, x, boundary_check=(0,1))
def run():
    M,N=4096,4096; BM,BN=128,128
    x = torch.randn((M,N), device="cuda", dtype=torch.float16)
    y = torch.zeros_like(x)
    grid = (triton.cdiv(M,BM),)
    copy_bp_bad[grid](x,y,M,N,BM,BM,num_stages=4,num_warps=8)
    torch.cuda.synchronize()
if __name__=="__main__": run()
PY

cat > "$ROOT/20_emit_stress.py" <<'PY'
# Stress patterns: re-create block_ptr each iter, predicate stores, add tails.
import torch, triton, triton.language as tl
@triton.jit
def stress_bp(X, Y, M: tl.constexpr, N: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    base_m = pid * BM
    # simulate descriptor-in-loop via new block_ptr per iteration
    for it in range(3):
        off_m = base_m + it*7  # tail assured
        x_blk = tl.make_block_ptr(X, (M, N), (N, 1), (off_m, 0), (BM, BN), (1, 0))
        y_blk = tl.make_block_ptr(Y, (M, N), (N, 1), (off_m, 0), (BM, BN), (1, 0))
        x = tl.load(x_blk, boundary_check=(0,1))
        # predicated store: write only on even it
        if (it % 2) == 0:
            tl.store(y_blk, x, boundary_check=(0,1))
def run():
    M,N=4224,4352; BM,BN=128,128  # tails by construction
    x = torch.randn((M,N), device="cuda", dtype=torch.float16)
    y = torch.zeros_like(x)
    grid = (triton.cdiv(M,BM),)
    stress_bp[grid](x,y,M,N,BM,BM,num_stages=4,num_warps=8)
    torch.cuda.synchronize()
if __name__=="__main__": run()
PY

# --- 3) Run all three and capture IR/MLIR/PTX dumps --------------------------
run_py() { python3 "$1" 2>&1 | tee "$ART/logs/$(basename "$1" .py).log" >/dev/null; }

echo "Running P0 (positive control)..."
run_py "$ROOT/10_emit_positive.py"
echo "Running P1 (negative control)..."
run_py "$ROOT/11_emit_negative.py"
echo "Running P2 (stress test)..."
run_py "$ROOT/20_emit_stress.py"

# --- 4) IR oracle (truth source) ---------------------------------------------
scan_ir() {
  local tag="$1"
  local kernel_pattern="$2"
  
  # Find IR files for this specific kernel
  local ttir_files=$(find "$DUMP" -name "${kernel_pattern}.ttir" 2>/dev/null || true)
  local ttgir_files=$(find "$DUMP" -name "${kernel_pattern}.ttgir" 2>/dev/null || true)
  
  local async_count=0
  
  # Check TTIR
  if [ -n "$ttir_files" ]; then
    for f in $ttir_files; do
      local cnt=$(grep -c "async_tma\|ttng\.async_tma" "$f" 2>/dev/null || echo 0)
      async_count=$((async_count + cnt))
    done
  fi
  
  # Check TTGIR
  if [ -n "$ttgir_files" ]; then
    for f in $ttgir_files; do
      local cnt=$(grep -c "async_tma\|ttng\.async_tma" "$f" 2>/dev/null || echo 0)
      async_count=$((async_count + cnt))
    done
  fi
  
  echo "$tag async_tma_hits=$async_count"
  echo "$tag $async_count" >> "$ART/ir/summary.tsv"
}

echo ""
echo "=== IR ORACLE RESULTS ==="
scan_ir "POSITIVE" "copy_bp"
scan_ir "NEGATIVE" "copy_bp_bad"
scan_ir "STRESS" "stress_bp"

# Save raw dumps & hashes
find "$DUMP" -type f \( -name "*.ttir" -o -name "*.ttgir" -o -name "*.llir" -o -name "*.ptx" \) -print0 2>/dev/null \
 | xargs -0 -I{} sh -c 'cp "{}" "'"$ART"'/ir/$(basename "{}")" 2>/dev/null' || true
( cd "$ART/ir" && sha256sum * > SHA256SUMS 2>/dev/null || true )

# --- 5) Determinism + basic speed sanity (only if stress shows async_tma) ----
if [ -f "$ART/ir/summary.tsv" ]; then
  ASY=$(awk '/STRESS/{print $2}' "$ART/ir/summary.tsv" 2>/dev/null || echo 0)
  if [ "${ASY:-0}" -gt 0 ]; then
    echo "STRESS has TMA - running determinism test..."
    python3 - <<'PY'
import os, torch, time, sys
sys.path.insert(0, os.environ["ROOT"])
from importlib import util

# dynamic import stress kernel
spec = util.spec_from_file_location("stress", os.path.join(os.environ["ROOT"], "20_emit_stress.py"))
m = util.module_from_spec(spec)
spec.loader.exec_module(m)

def bench():
  M,N=4224,4352
  x = torch.randn((M,N), device="cuda", dtype=torch.float16)
  y = torch.zeros_like(x)
  m.stress_bp[(32,)](x,y,M,N,128,128,num_stages=4,num_warps=8)
  torch.cuda.synchronize()
  return y

# warmup
for _ in range(10): bench()
torch.cuda.synchronize()

# determinism check
ref = bench()
mismatches = 0
for i in range(100):
    y = bench()
    if not torch.equal(ref, y):
        mismatches += 1

print(f"Determinism: {100-mismatches}/100 runs matched")

# speed
t0=time.time()
for _ in range(100): bench()
torch.cuda.synchronize()
print(f"Speed: mean={(time.time()-t0)*10:.3f}ms")
PY
  else
    echo "STRESS has no async_tma - skipping determinism" | tee "$ART/logs/determinism_speed.log"
  fi
fi

# --- 6) Minimal sanitizer pass (if available) --------------------------------
if command -v compute-sanitizer >/dev/null 2>&1; then
  echo "Running sanitizers..."
  compute-sanitizer --tool memcheck  python3 "$ROOT/20_emit_stress.py" > "$ART/logs/memcheck.log" 2>&1 || true
  compute-sanitizer --tool racecheck python3 "$ROOT/20_emit_stress.py" > "$ART/logs/racecheck.log" 2>&1 || true
fi

# --- 7) Bundle with manifest (anti‑fabrication) ------------------------------
( cd "$ROOT" && find artifacts -type f -exec sha256sum {} + 2>/dev/null | sort -k2,2 > MANIFEST.SHA256 )
tar -C "$ROOT" -czf "${ROOT}.tgz" artifacts MANIFEST.SHA256 2>/dev/null
sha256sum "${ROOT}.tgz" | tee "${ROOT}.tgz.SHA256"
echo ""
echo "✅ Bundle: ${ROOT}.tgz"
echo ""
echo "=== SUMMARY ==="
cat "$ART/ir/summary.tsv" 2>/dev/null || echo "No summary"

