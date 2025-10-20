#!/bin/bash
set -euo pipefail

cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9

echo "=================================================================="
echo "Stage-3B Fused Softmax: Full Validation"
echo "=================================================================="
echo ""

# =========================================
# 1. PTXAS Validation
# =========================================
echo "=== 1. PTXAS VALIDATION ==="
echo ""

echo "[1a] Stage-2 baseline (USE_FUSED_SOFTMAX=0)..."
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX=0 USE_SMEM_SWIZZLE_XOR=0 \
python -m tasks.fp8_sdpa_stage_c_wmma.build 2>&1 | tee .build_s2_baseline.log
echo ""

echo "[1b] Stage-3B fused (USE_FUSED_SOFTMAX=1)..."
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX=1 USE_SMEM_SWIZZLE_XOR=0 \
python -m tasks.fp8_sdpa_stage_c_wmma.build 2>&1 | tee .build_s3b_fused.log
echo ""

echo "--- PTXAS Summary ---"
echo "Stage-2 baseline:"
grep -E "Used [0-9]+ registers|spill|smem" .build_s2_baseline.log | grep "sdpa_fp8_stage_c_wmma_kernel" || echo "  (no stats)"
echo ""
echo "Stage-3B fused:"
grep -E "Used [0-9]+ registers|spill|smem" .build_s3b_fused.log | grep "sdpa_fp8_stage_c_wmma_kernel" || echo "  (no stats)"
echo ""

# =========================================
# 2. Correctness Validation
# =========================================
echo "=== 2. CORRECTNESS VALIDATION (6 tests) ==="
echo ""

echo "[2a] Stage-2 baseline..."
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX=0 \
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small,mission --seeds 0,1,2 \
2>&1 | tee .corr_s2_baseline.log
echo ""

echo "[2b] Stage-3B fused..."
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX=1 \
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small,mission --seeds 0,1,2 \
2>&1 | tee .corr_s3b_fused.log
echo ""

echo "--- Correctness Summary ---"
echo "Stage-2 baseline:"
grep -E "PASS|FAIL" .corr_s2_baseline.log | tail -10
echo ""
echo "Stage-3B fused:"
grep -E "PASS|FAIL" .corr_s3b_fused.log | tail -10
echo ""

# =========================================
# 3. Performance Benchmark
# =========================================
echo "=== 3. PERFORMANCE BENCHMARK (mission shape, 500 iters) ==="
echo ""

echo "[3a] Stage-2 baseline..."
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX=0 \
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 500 \
2>&1 | tee .perf_s2_baseline.log
echo ""

echo "[3b] Stage-3B fused..."
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX=1 \
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 500 \
2>&1 | tee .perf_s3b_fused.log
echo ""

# =========================================
# 4. Extract Results & Save Artifacts
# =========================================
echo "=== 4. RESULTS & ARTIFACTS ==="
echo ""

mkdir -p results/2025-Stage3B-Fused/
cp .build_*.log .corr_*.log .perf_*.log results/2025-Stage3B-Fused/

python - <<'PY'
import re, json, pathlib

p = pathlib.Path("results/2025-Stage3B-Fused")

def extract_p50(path):
    if not pathlib.Path(path).exists():
        return None
    txt = pathlib.Path(path).read_text()
    m = re.search(r"p50\s*=\s*([\d.]+)\s*μs", txt)
    return float(m.group(1)) if m else None

def extract_correctness(path):
    if not pathlib.Path(path).exists():
        return None
    txt = pathlib.Path(path).read_text()
    passed = len(re.findall(r"✅ PASS", txt))
    failed = len(re.findall(r"❌ FAIL", txt))
    return {"passed": passed, "failed": failed}

s2_p50 = extract_p50(".perf_s2_baseline.log")
s3b_p50 = extract_p50(".perf_s3b_fused.log")

s2_corr = extract_correctness(".corr_s2_baseline.log")
s3b_corr = extract_correctness(".corr_s3b_fused.log")

speedup = None
if s2_p50 and s3b_p50 and s2_p50 > 0:
    speedup = ((s2_p50 - s3b_p50) / s2_p50) * 100

summary = {
    "stage2_baseline": {
        "p50_us": s2_p50,
        "correctness": s2_corr
    },
    "stage3b_fused": {
        "p50_us": s3b_p50,
        "correctness": s3b_corr,
        "speedup_pct": round(speedup, 2) if speedup else None
    }
}

(p/"validation_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo ""
echo "=================================================================="
echo "✅ Validation complete! Artifacts saved to results/2025-Stage3B-Fused/"
echo "=================================================================="

