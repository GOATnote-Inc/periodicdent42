#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ART="$ROOT/artifacts/ncu"
BENCH="$ROOT/artifacts/bench"
TUNE="$ROOT/artifacts/tune"
mkdir -p "$ART"

# Metrics robust across Ada/Hopper; script tolerates missing fields.
# We'll export both section sets and specific metrics to ensure availability.
SECTIONS="--section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section SchedulerStats --section WarpStateStats --section Occupancy"
METRICS=(
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  smsp__sass_thread_inst_executed_op_hmma_pred_on.sum
  smsp__inst_executed_pipe_tensor.sum
  sm__warps_active.avg.pct_of_peak_sustained_active
  sm__warps_active.avg.per_cycle_active
  smsp__warps_eligible.avg_per_cycle_active
  smsp__average_warps_issue_stalled_per_active_cycle.pct
  smsp__average_warp_latency_per_issue_active
  lts__t_sectors.avg.pct_of_peak_sustained_active
  lts__t_sectors_aperture_sysmem_op_read.sum
  lts__t_sectors_aperture_sysmem_op_write.sum
  lts__t_sectors_srcunit_tex_op_read.sum
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  sm__maximum_warps_active.avg
  sm__sass_average_data_bytes_per_sector_mem_global.pct
  sm__inst_executed.sum
  smsp__thread_inst_executed_per_inst_executed.ratio
)
METARG=$(printf -- "--metrics %s " "${METRICS[@]}")

profile_one () {
  local tag="$1"; shift
  local rep="$ART/$tag.ncu-rep"
  echo "[NCU] Profiling $tag → $rep"
  # Run kbench once under NCU; skip warmups via kbench args.
  sudo /usr/local/cuda/bin/ncu -o "$rep" --target-processes all --set full $SECTIONS $METARG \
      python3 "$ROOT/scripts/kbench.py" --iters 20 --warmup 5 --variants "$@"
}

# Baseline (PyTorch SDPA fastest/native won't generate a single kernel name -> profile run as a whole)
profile_one baseline candidate_triton_flashlike   # we collect env / control kernel

# Read topk to profile best three (or fall back to defaults)
if [[ -f "$TUNE/topk.json" ]]; then
  python3 - <<'PY' "$TUNE/topk.json"
import json,sys; j=json.load(open(sys.argv[1]))
# We profile both Stage-2 and WS variants to be safe.
print("candidate_triton_flashlike")
print("candidate_cuda_stub")
print("candidate_triton_ws")
PY
else
  echo "candidate_triton_flashlike"
  echo "candidate_cuda_stub"
  echo "candidate_triton_ws"
fi | while read v; do
  tag=$(echo "$v" | sed 's/candidate_//g')
  profile_one "candidate_${tag}" "$v"
done

# Parse to summary.json
python3 "$ROOT/scripts/parse_ncu.py" --repdir "$ART" --out "$ART/summary.json"
