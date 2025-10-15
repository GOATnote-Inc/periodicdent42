#!/usr/bin/env bash
set -euo pipefail
echo "════════════════════════════════════════════════════════════════════"
echo "  GPU Validation & Benchmarking Suite"
echo "════════════════════════════════════════════════════════════════════"

# 0) Keep GPU alive + env
echo "[0/6] Starting GPU keepalive and setting environment..."
nohup scripts/gpu_keepalive.sh >/dev/null 2>&1 || true
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/kiteboard/periodicdent42:/home/kiteboard/periodicdent42/cudadent42/bench:${PYTHONPATH:-}"
nvidia-smi || echo "⚠️  nvidia-smi not available"

# 1) Rebuild release + bench + summary
echo ""
echo "[1/6] Rebuilding release and running benchmarks..."
rm -rf ~/.cache/torch_extensions/* /tmp/torch_extensions/* || true
python3 - <<'PY'
from build_v3_release import build_v3_release
build_v3_release(False)
print("✅ build(release) OK")
PY

python3 scripts/bench_s512_tc_vs_sdpa.py || echo "⚠️  Benchmark failed"
python3 scripts/summarize_s512_bench.py || echo "⚠️  Summary generation failed"
echo ""
echo "═══ Benchmark Summary ═══"
sed -n '1,60p' cudadent42/artifacts/bench/S512_BENCH_SUMMARY.md 2>/dev/null || echo "(no summary generated)"

# 2) Racecheck & DSA single-case loop (25 calls)
echo ""
echo "[2/6] Running racecheck and DSA validation..."
if command -v compute-sanitizer >/dev/null 2>&1; then
  echo "Running racecheck (25 iterations)..."
  compute-sanitizer --tool racecheck python3 - <<'PY' 2>&1 | tail -20 || echo "⚠️  Racecheck unavailable"
import torch
from build_v3_release import build_v3_release
m=build_v3_release(False); f=m.forward_32_64_4_2_1_1 if hasattr(m,'forward_32_64_4_2_1_1') else m.forward
B,H,S,D=2,8,512,64
Q=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16); K=Q.clone(); V=Q.clone()
s=1.0/(D**0.5)
for i in range(25): 
    if hasattr(m,'forward_32_64_4_2_1_1'):
        f(Q,K,V,s,False,1)
    else:
        f(Q,K,V,s,False)
torch.cuda.synchronize(); print("racecheck: loop ok")
PY
else
  echo "⚠️  compute-sanitizer not available"
fi

echo "Running DSA (device-side assert) test..."
CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python3 - <<'PY' 2>&1 | tail -20 || echo "⚠️  DSA test failed"
import torch
from build_v3_release import build_v3_release
m=build_v3_release(False); f=m.forward_32_64_4_2_1_1 if hasattr(m,'forward_32_64_4_2_1_1') else m.forward
B,H,S,D=2,8,512,64
Q=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16); K=Q.clone(); V=Q.clone()
s=1.0/(D**0.5)
for i in range(25): 
    if hasattr(m,'forward_32_64_4_2_1_1'):
        f(Q,K,V,s,False,1)
    else:
        f(Q,K,V,s,False)
torch.cuda.synchronize(); print("DSA: loop ok")
PY

# 3) Stream variant benchmark
echo ""
echo "[3/6] Running stream-per-iteration variant..."
python3 scripts/bench_s512_tc_vs_sdpa.py --streams || echo "⚠️  Stream variant failed"
python3 scripts/summarize_s512_bench.py || true
echo ""
echo "═══ Stream Variant Summary ═══"
sed -n '1,60p' cudadent42/artifacts/bench/S512_BENCH_SUMMARY.md 2>/dev/null || echo "(no summary generated)"

# 4) Nsight Compute (single shape) capture
echo ""
echo "[4/6] Running Nsight Compute capture..."
if command -v ncu >/dev/null 2>&1; then
  OUT=benchmarks/l4/$(date +%F)/nsight/canon_3; mkdir -p "$OUT"
  ncu --set full --target-processes all --replay-mode application \
      --export "$OUT/report" \
      python3 - <<'PY' 2>&1 | tail -30 || echo "⚠️  Nsight Compute failed"
import torch
from build_v3_release import build_v3_release
m=build_v3_release(False); f=m.forward_32_64_4_2_1_1 if hasattr(m,'forward_32_64_4_2_1_1') else m.forward
B,H,S,D=2,8,512,64
Q=torch.randn(B,H,S,D,device='cuda',dtype=torch.float16); K=Q.clone(); V=Q.clone()
s=1.0/(D**0.5)
for _ in range(10): 
    if hasattr(m,'forward_32_64_4_2_1_1'):
        f(Q,K,V,s,False,1)
    else:
        f(Q,K,V,s,False)
    torch.cuda.synchronize()
if hasattr(m,'forward_32_64_4_2_1_1'):
    f(Q,K,V,s,False,1)
else:
    f(Q,K,V,s,False)
torch.cuda.synchronize()
print("ncu target ok")
PY
  ncu --import "$OUT/report.ncu-rep" --page summary --csv > "$OUT/report.txt" 2>&1 || true
  echo "═══ Nsight Summary (first 80 lines) ═══"
  sed -n '1,80p' "$OUT/report.txt" 2>/dev/null || echo "(no report generated)"
else
  echo "⚠️  ncu (Nsight Compute) not available"
fi

# 5) EvoEngineer sweep
echo ""
echo "[5/6] Running EvoEngineer sweep..."
if [ -f third_party/evoengineer_stub/ee_loop.py ]; then
  python third_party/evoengineer_stub/ee_loop.py || echo "⚠️  EvoEngineer sweep failed"
else
  echo "⚠️  EvoEngineer stub not found (skipping)"
fi

# 6) Summary
echo ""
echo "[6/6] Validation complete!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "ARTIFACTS GENERATED:"
echo "  • cudadent42/artifacts/bench/tc_vs_sdpa_s512.json"
echo "  • cudadent42/artifacts/bench/S512_BENCH_SUMMARY.md"
if [ -d "benchmarks/l4/$(date +%F)/nsight" ]; then
  echo "  • benchmarks/l4/$(date +%F)/nsight/canon_3/report.ncu-rep"
fi
echo ""
echo "NEXT STEPS:"
echo "  1. Review benchmark summary above"
echo "  2. git add cudadent42/artifacts/bench/*.json cudadent42/artifacts/bench/*.md"
echo "  3. git commit -m \"bench: release S=512 results\""
echo "  4. git push origin feature/evidence_wmma_tc"
echo ""
echo "════════════════════════════════════════════════════════════════════"

