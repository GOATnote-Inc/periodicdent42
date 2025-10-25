#!/bin/bash
set -euo pipefail

cd ~/periodicdent42
source venv/bin/activate
export PATH=/usr/local/cuda-12.2/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9

echo "===================================================================="
echo "Step 2: Performance Benchmark (mission shape, 500 iters)"
echo "===================================================================="

# Run performance benchmark
USE_CP_ASYNC=1 USE_WMMA_PV=1 USE_FUSED_SOFTMAX=0 \
python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 500 \
2>&1 | tee .perf_step2_mission.log

echo ""
echo "===================================================================="
echo "Performance Summary:"
echo "===================================================================="
grep -E "p50|p90|p99|mean" .perf_step2_mission.log | tail -10 || echo "Performance output"

echo ""
echo "===================================================================="
echo "Performance validation complete. Log saved to .perf_step2_mission.log"
echo "===================================================================="

# Create results directory and save artifacts
mkdir -p results/2025-Stage3-Fusion-Full/step2-xor/
cp .build_step2.log .corr_s2_step2.log .perf_step2_mission.log \
   results/2025-Stage3-Fusion-Full/step2-xor/

# Extract p50 and record summary
python - <<'PY'
import re, json, pathlib
p = pathlib.Path("results/2025-Stage3-Fusion-Full/step2-xor")
txt = pathlib.Path(".perf_step2_mission.log").read_text()
m = re.search(r"p50\s*=\s*([\d.]+)\s*μs", txt)
out = {"mission_p50_us": float(m.group(1)) if m else None}
(p/"perf_summary.json").write_text(json.dumps(out, indent=2))
print(f"✅ Performance summary: {out}")
PY

echo ""
echo "✅ Artifacts saved to results/2025-Stage3-Fusion-Full/step2-xor/"

