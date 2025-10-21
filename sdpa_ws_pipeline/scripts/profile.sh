#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute profiling for baseline + top-3 candidates.
# Exports:
#   artifacts/ncu/baseline.ncu-rep
#   artifacts/ncu/candidate_{1,2,3}.ncu-rep
#   artifacts/ncu/summary.json (aggregated)

mkdir -p artifacts/ncu

METRICS="\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
lts__t_sectors_srcunit_tex_op_read_lookup_hit_rate.pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_eligible_per_cycle_avg,\
gpu__time_duration.sum\
"

# Baseline A
echo "[NCU] Profiling baseline A..."
ncu --target-processes all --set full --section SpeedOfLight --metrics $METRICS \
    --export artifacts/ncu/baseline --force-overwrite \
    python3 scripts/kbench.py --backend baseline_a --shape mission --iters 40 --warmup 10 >/dev/null

# Candidates (top-3 from artifacts/tune/topk.json if present; else defaults)
CANDS=$(python3 - <<'PY'
import json, os, sys
p="artifacts/tune/topk.json"
if os.path.exists(p):
    obj=json.load(open(p)); 
    cs=[c["backend"] for c in obj.get("topk",[])]
    if cs: 
        print(" ".join(cs[:3])); sys.exit(0)
print("candidate_triton_ws candidate_triton_flashlike candidate_cuda_stub")
PY
)
i=1
for c in $CANDS; do
  echo "[NCU] Profiling $c ..."
  ncu --target-processes all --set full --section SpeedOfLight --metrics $METRICS \
      --export "artifacts/ncu/candidate_$i" --force-overwrite \
      python3 scripts/kbench.py --backend "$c" --shape mission --iters 40 --warmup 10 >/dev/null
  i=$((i+1))
done

# Aggregate metrics
python3 scripts/parse_ncu.py --in artifacts/ncu/baseline.ncu-rep --name baseline --out artifacts/ncu/summary.json
[ -f artifacts/ncu/candidate_1.ncu-rep ] && python3 scripts/parse_ncu.py --in artifacts/ncu/candidate_1.ncu-rep --name candidate_1 --out artifacts/ncu/summary.json || true
[ -f artifacts/ncu/candidate_2.ncu-rep ] && python3 scripts/parse_ncu.py --in artifacts/ncu/candidate_2.ncu-rep --name candidate_2 --out artifacts/ncu/summary.json || true
[ -f artifacts/ncu/candidate_3.ncu-rep ] && python3 scripts/parse_ncu.py --in artifacts/ncu/candidate_3.ncu-rep --name candidate_3 --out artifacts/ncu/summary.json || true

echo "NCU artifacts ready under artifacts/ncu/"
