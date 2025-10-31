#!/bin/bash
# BlackwellSparseK Production Benchmark Script
# CUDA Events timing + Nsight Compute profiling + Clock locking
# Usage: ./bench.sh [--profile] [--trials N] [--warmup SECONDS]

set -euo pipefail

# ============================= Configuration ==============================

TRIALS=${TRIALS:-30}
WARMUP=${WARMUP:-60}
SEED=${SEED:-42}
PROFILE=${PROFILE:-0}
BINARY=${BINARY:-build/sparse_h100}
OUTPUT_DIR=${OUTPUT_DIR:-artifacts}
CONFIG=${CONFIG:-"M=8192 N=8192 K=8192 TOPK=16"}

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --profile) PROFILE=1; shift ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

# ============================= Functions ==================================

banner() {
  echo ""
  echo "========================================================================"
  echo "$1"
  echo "========================================================================"
  echo ""
}

check_gpu() {
  if ! nvidia-smi &>/dev/null; then
    echo "❌ ERROR: nvidia-smi failed. No GPU detected."
    exit 1
  fi
  
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  
  echo "✅ GPU: $GPU_NAME (CC $GPU_CC, Driver $DRIVER)"
}

lock_clocks() {
  banner "Locking GPU Clocks"
  
  # Enable persistence mode
  nvidia-smi -pm 1 2>&1 | grep -v "^$" || true
  
  # Get max clocks
  MAX_GFX=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
  MAX_MEM=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1)
  
  echo "Max clocks: Graphics=$MAX_GFX MHz, Memory=$MAX_MEM MHz"
  
  # Lock to max (for H100, typical: gfx=1980, mem=1593)
  if nvidia-smi -lgc "$MAX_GFX,$MAX_GFX" 2>/dev/null; then
    echo "✅ Graphics clock locked to $MAX_GFX MHz"
  else
    echo "⚠️  Cannot lock graphics clock (insufficient permissions)"
  fi
  
  if nvidia-smi -lmc "$MAX_MEM,$MAX_MEM" 2>/dev/null; then
    echo "✅ Memory clock locked to $MAX_MEM MHz"
  else
    echo "⚠️  Cannot lock memory clock (insufficient permissions)"
  fi
}

unlock_clocks() {
  banner "Unlocking GPU Clocks"
  nvidia-smi -rgc 2>/dev/null || echo "⚠️  Cannot reset graphics clock"
  nvidia-smi -rmc 2>/dev/null || echo "⚠️  Cannot reset memory clock"
  echo "✅ Clocks reset to default"
}

thermal_warmup() {
  banner "Thermal Warm-up ($WARMUP seconds)"
  
  local end_time=$((SECONDS + WARMUP))
  local count=0
  
  while [ $SECONDS -lt $end_time ]; do
    $BINARY > /dev/null 2>&1 || true
    count=$((count + 1))
    
    if [ $((count % 10)) -eq 0 ]; then
      local remaining=$((end_time - SECONDS))
      echo "  Warm-up progress: $count runs, ${remaining}s remaining..."
    fi
  done
  
  echo "✅ Warm-up complete ($count runs)"
  
  # Let GPU stabilize
  sleep 2
}

run_trials() {
  banner "Running $TRIALS Trials (CUDA Events timing)"
  
  local timings_file="$OUTPUT_DIR/timings_$(date +%Y%m%d_%H%M%S).txt"
  
  echo "Trial,Latency_ms,TFLOPS" > "$timings_file"
  
  for trial in $(seq 1 $TRIALS); do
    # Run with CUDA Events timing
    output=$($BINARY 2>&1)
    
    # Parse output (kernel should print: "Latency: X.XXX ms, TFLOPS: YYY.Y")
    latency=$(echo "$output" | grep -oP 'Latency: \K[0-9.]+' || echo "0")
    tflops=$(echo "$output" | grep -oP 'TFLOPS: \K[0-9.]+' || echo "0")
    
    echo "$trial,$latency,$tflops" >> "$timings_file"
    
    if [ $((trial % 5)) -eq 0 ]; then
      echo "  Progress: $trial/$TRIALS trials complete"
    fi
  done
  
  echo "✅ Trials complete: $timings_file"
  
  # Compute statistics
  python3 - <<EOF
import pandas as pd
import numpy as np

df = pd.read_csv("$timings_file")
latencies = df['Latency_ms'].values

mean = np.mean(latencies)
std = np.std(latencies)
cv = (std / mean) * 100 if mean > 0 else 0
p50 = np.percentile(latencies, 50)
p99 = np.percentile(latencies, 99)

print(f"\n📊 Timing Statistics (N={len(latencies)}):")
print(f"  Mean:   {mean:.3f} ms")
print(f"  Std:    {std:.3f} ms")
print(f"  CV:     {cv:.2f}%")
print(f"  p50:    {p50:.3f} ms")
print(f"  p99:    {p99:.3f} ms")

if cv < 3.0:
    print(f"\n✅ CV < 3% - Excellent repeatability")
elif cv < 5.0:
    print(f"\n⚠️  CV {cv:.2f}% - Acceptable but investigate jitter")
else:
    print(f"\n❌ CV {cv:.2f}% - High jitter, check thermal/clocks")

# Save summary
with open("$OUTPUT_DIR/summary.txt", "w") as f:
    f.write(f"mean: {mean:.3f}\n")
    f.write(f"std: {std:.3f}\n")
    f.write(f"cv: {cv:.2f}\n")
    f.write(f"p50: {p50:.3f}\n")
    f.write(f"p99: {p99:.3f}\n")
EOF
}

run_nsight() {
  banner "Nsight Compute Profiling"
  
  local ncu_out="$OUTPUT_DIR/nsight_$(date +%Y%m%d_%H%M%S)"
  
  echo "Running Nsight Compute (light preset)..."
  
  ncu --target-processes all \
      --set full \
      --section LaunchStats,Occupancy,MemoryWorkloadAnalysis \
      --metrics \
        sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
        sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed \
      --csv \
      --export "$ncu_out" \
      $BINARY 2>&1 | tee "$OUTPUT_DIR/ncu.log"
  
  echo "✅ Nsight report: $ncu_out.ncu-rep"
  
  # Extract metrics
  if [ -f "$ncu_out.csv" ]; then
    python3 /workspace/scripts/extract_baseline.py \
      --csv "$ncu_out.csv" \
      --output "$OUTPUT_DIR/nsight_metrics.txt"
  elif [ -f "$OUTPUT_DIR/ncu.log" ]; then
    python3 /workspace/scripts/extract_baseline.py \
      --log "$OUTPUT_DIR/ncu.log" \
      --output "$OUTPUT_DIR/nsight_metrics.txt"
  fi
}

capture_environment() {
  banner "Capturing Environment"
  
  local env_file="$OUTPUT_DIR/environment_$(date +%Y%m%d_%H%M%S).txt"
  
  {
    echo "=== Hardware ==="
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,vbios_version --format=csv
    
    echo ""
    echo "=== Software ==="
    echo "CUDA: $(nvcc --version | grep release)"
    echo "Nsight Compute: $(ncu --version 2>&1 | head -1)"
    echo "CUTLASS: $(cd /opt/cutlass && git describe --tags 2>/dev/null || echo "main")"
    
    echo ""
    echo "=== Build ==="
    strings $BINARY | grep -E "(sm_90|sm_100)" | head -5 || echo "Unknown architecture"
    
    echo ""
    echo "=== Container ==="
    cat /etc/os-release | grep -E "(PRETTY_NAME|VERSION_ID)"
    
    echo ""
    echo "=== Config ==="
    echo "Trials: $TRIALS"
    echo "Warmup: $WARMUP seconds"
    echo "Seed: $SEED"
    echo "Config: $CONFIG"
    
  } > "$env_file"
  
  echo "✅ Environment captured: $env_file"
  cat "$env_file"
}

# ============================= Main Workflow ==============================

trap unlock_clocks EXIT

banner "BlackwellSparseK Production Benchmark"

check_gpu
lock_clocks
capture_environment
thermal_warmup
run_trials

if [ "$PROFILE" -eq 1 ]; then
  run_nsight
fi

banner "Benchmark Complete"

echo "📁 Output directory: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR/"

exit 0

