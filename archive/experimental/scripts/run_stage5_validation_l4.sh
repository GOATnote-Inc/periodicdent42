#!/usr/bin/env bash
#
# Stage-5 WS Validation Script (L4 GPU Required)
# ==============================================
# Executes Steps 2-7 from the Stage-5 prompt:
# - Build control (Stage-2) and WS variants with PTXAS capture
# - Run robust benchmarks (100-run medians, PyTorch comparison)
# - NCU profiling (compute-bound diagnosis)
# - EvoEngineer-Full autotune (elite K=3)
# - Package reproducible artifacts
#
# Prerequisites:
# - NVIDIA L4 GPU (sm_89)
# - CUDA 12.2+ with ncu available
# - Python venv with torch, ninja
#
# Usage:
#   bash scripts/run_stage5_validation_l4.sh
#

set -euo pipefail

# ====================
# 0. Environment Setup
# ====================
echo "======================================================================"
echo "Stage-5 WS Validation on L4"
echo "======================================================================"

# Check GPU
if ! nvidia-smi -L | grep -q "L4"; then
    echo "⚠️  Warning: Expected NVIDIA L4, but got:"
    nvidia-smi -L
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Setup environment
export PATH=/usr/local/cuda-12.2/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9

# Create logs directory
mkdir -p kbench/logs

# Activate venv (adjust path if needed)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ No venv found. Please create one first:"
    echo "   python3 -m venv venv && source venv/bin/activate"
    echo "   pip install torch numpy ninja"
    exit 1
fi

# ====================
# 1. Sanity Check (Optional)
# ====================
echo ""
echo "======================================================================"
echo "Step 1: Environment Sanity Check"
echo "======================================================================"

python - <<'PY'
import torch
import platform

print(f"Python:      {platform.python_version()}")
print(f"PyTorch:     {torch.__version__}")
print(f"CUDA:        {torch.version.cuda}")
print(f"GPU:         {torch.cuda.get_device_name(0)}")
print(f"SM:          {torch.cuda.get_device_capability(0)}")
print(f"CUDA avail:  {torch.cuda.is_available()}")
PY

# ====================
# 2. Build Control (Stage-2) + WS Variants
# ====================
echo ""
echo "======================================================================"
echo "Step 2: Build Variants (Control + WS)"
echo "======================================================================"

# Stage-2 Baseline (Control)
echo ""
echo "[2.1] Building Stage-2 control (cp.async + WMMA P·V)..."
USE_CP_ASYNC=1 USE_WMMA_PV=1 \
python -m tasks.fp8_sdpa_stage_c_wmma.build 2>&1 | tee kbench/logs/build_stage2_control.txt

echo ""
echo "PTXAS Stats (Stage-2 Control):"
grep -E "Used [0-9]+ registers|spill|smem" kbench/logs/build_stage2_control.txt || echo "(No PTXAS output found)"

# WS with 1 producer
echo ""
echo "[2.2] Building Stage-5 WS (NUM_PRODUCER_WARPS=1)..."
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_WARP_SPECIALIZATION=1 NUM_PRODUCER_WARPS=1 \
python -m tasks.fp8_sdpa_stage_c_wmma.build 2>&1 | tee kbench/logs/build_ws_p1.txt

echo ""
echo "PTXAS Stats (WS P=1):"
grep -E "Used [0-9]+ registers|spill|smem" kbench/logs/build_ws_p1.txt || echo "(No PTXAS output found)"

# WS with 2 producers (optional, only if P=1 passes PTXAS)
if ! grep -q "spill" kbench/logs/build_ws_p1.txt; then
    echo ""
    echo "[2.3] Building Stage-5 WS (NUM_PRODUCER_WARPS=2)..."
    USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_WARP_SPECIALIZATION=1 NUM_PRODUCER_WARPS=2 \
    python -m tasks.fp8_sdpa_stage_c_wmma.build 2>&1 | tee kbench/logs/build_ws_p2.txt
    
    echo ""
    echo "PTXAS Stats (WS P=2):"
    grep -E "Used [0-9]+ registers|spill|smem" kbench/logs/build_ws_p2.txt || echo "(No PTXAS output found)"
else
    echo ""
    echo "⚠️  Skipping WS P=2 (P=1 has spills or high regs)"
fi

# ====================
# 3. Robust Benchmarks
# ====================
echo ""
echo "======================================================================"
echo "Step 3: Robust Benchmarks (100-run medians + PyTorch comparison)"
echo "======================================================================"

# Stage-2 baseline
echo ""
echo "[3.1] Benchmarking Stage-2 control..."
python scripts/bench_sdpa.py --iters 100 --warmup 20 --shapes small,mission,long \
  --out kbench/baseline_stage2.json 2>&1 | tee kbench/logs/bench_stage2.txt

# WS P=1
echo ""
echo "[3.2] Benchmarking WS (P=1)..."
USE_WARP_SPECIALIZATION=1 NUM_PRODUCER_WARPS=1 \
python scripts/bench_sdpa.py --iters 100 --warmup 20 --shapes small,mission,long \
  --out kbench/ws_p1.json 2>&1 | tee kbench/logs/bench_ws_p1.txt

# WS P=2 (if built)
if [ -f kbench/logs/build_ws_p2.txt ]; then
    echo ""
    echo "[3.3] Benchmarking WS (P=2)..."
    USE_WARP_SPECIALIZATION=1 NUM_PRODUCER_WARPS=2 \
    python scripts/bench_sdpa.py --iters 100 --warmup 20 --shapes small,mission,long \
      --out kbench/ws_p2.json 2>&1 | tee kbench/logs/bench_ws_p2.txt
fi

# ====================
# 4. Compare Results
# ====================
echo ""
echo "======================================================================"
echo "Step 4: Compare Results (Stage-2 vs WS)"
echo "======================================================================"

python - <<'PY'
import json

def load_results(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def print_comparison(name, results):
    if not results:
        print(f"❌ {name}: No results")
        return
    for r in results:
        status = "✅ PASS" if r.get("correctness_pass", False) else "❌ FAIL"
        print(f"{name:15s} | {r['shape']:8s} | p50={r['p50_us']:7.2f}μs | speedup={r['speedup_vs_torch']:5.1f}× | max_err={r['max_err']:.4f} | {status}")

print("Shape Comparison:")
print("-" * 100)
print("Variant         | Shape    | p50 Latency | vs PyTorch | max_err | Status")
print("-" * 100)

stage2 = load_results("kbench/baseline_stage2.json")
ws_p1 = load_results("kbench/ws_p1.json")
ws_p2 = load_results("kbench/ws_p2.json")

if stage2:
    print_comparison("Stage-2", stage2)
if ws_p1:
    print_comparison("WS (P=1)", ws_p1)
if ws_p2:
    print_comparison("WS (P=2)", ws_p2)

print("-" * 100)

# Gate checks
if ws_p1 and stage2:
    mission_s2 = next((r for r in stage2 if r['shape'] == 'mission'), None)
    mission_ws = next((r for r in ws_p1 if r['shape'] == 'mission'), None)
    
    if mission_s2 and mission_ws:
        speedup_pct = 100 * (mission_s2['p50_us'] - mission_ws['p50_us']) / mission_s2['p50_us']
        print(f"\nWS (P=1) vs Stage-2 (mission): {speedup_pct:+.1f}%")
        
        if speedup_pct >= 10:
            print(f"✅ Performance gate PASS (≥+10%)")
        else:
            print(f"⚠️  Performance gate MARGINAL (<+10%)")
        
        if mission_ws['speedup_vs_torch'] >= 15:
            print(f"✅ PyTorch speedup gate PASS (≥15×)")
        else:
            print(f"⚠️  PyTorch speedup gate FAIL (<15×)")
PY

# ====================
# 5. NCU Profiling (Optional, requires sudo)
# ====================
echo ""
echo "======================================================================"
echo "Step 5: NCU Profiling (Optional - requires sudo)"
echo "======================================================================"

if command -v ncu &> /dev/null; then
    read -p "Run NCU profiling? (requires sudo) (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running NCU profiling..."
        bash scripts/ncu_sdpa.sh 2>&1 | tee kbench/logs/ncu_stage5.txt
        echo "NCU results saved to kbench/ncu_stage5.ncu-rep"
    else
        echo "Skipping NCU profiling"
    fi
else
    echo "⚠️  ncu not found, skipping profiling"
fi

# ====================
# 6. EvoEngineer-Full Autotune (Optional)
# ====================
echo ""
echo "======================================================================"
echo "Step 6: EvoEngineer-Full Autotune (Optional - takes ~2-4 hours)"
echo "======================================================================"

read -p "Run EvoEngineer-Full autotune? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running autotune (this will take a while)..."
    python kbench/autotune_evo_full.py 2>&1 | tee kbench/logs/autotune_full.txt
    
    if [ -f kbench/elite.json ]; then
        echo ""
        echo "Autotune complete! Elite configurations:"
        python - <<'PY'
import json
with open("kbench/elite.json") as f:
    elites = json.load(f)
for i, e in enumerate(elites, 1):
    print(f"{i}. p50={e['score']:.2f} μs — {e['config']}")
PY
    fi
else
    echo "Skipping autotune"
fi

# ====================
# 7. Package Artifacts
# ====================
echo ""
echo "======================================================================"
echo "Step 7: Package Artifacts"
echo "======================================================================"

# Capture environment
git rev-parse HEAD > kbench/GIT_SHA.txt 2>/dev/null || echo "N/A" > kbench/GIT_SHA.txt
git branch --show-current > kbench/GIT_BRANCH.txt 2>/dev/null || echo "N/A" > kbench/GIT_BRANCH.txt
nvidia-smi -L > kbench/NVIDIA_SMI.txt

python - <<'PY'
import platform, json, torch

env = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(0),
    "sm": list(torch.cuda.get_device_capability(0)),
    "platform": platform.system(),
    "machine": platform.machine(),
}

with open("kbench/ENV.json", "w") as f:
    json.dump(env, f, indent=2)

print(json.dumps(env, indent=2))
PY

# ====================
# 8. Summary
# ====================
echo ""
echo "======================================================================"
echo "Stage-5 Validation COMPLETE"
echo "======================================================================"
echo ""
echo "Artifacts saved to kbench/:"
ls -lh kbench/*.json kbench/*.txt 2>/dev/null || echo "(No artifacts found)"
echo ""
echo "Logs saved to kbench/logs/:"
ls -lh kbench/logs/*.txt 2>/dev/null || echo "(No logs found)"
echo ""
echo "Next steps:"
echo "1. Review benchmark results above"
echo "2. If gates pass: git add kbench/ && git commit -m '...' && git push"
echo "3. If gates fail: Review logs, debug, iterate"
echo ""
echo "✅ Validation script complete!"

