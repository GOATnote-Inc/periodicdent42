#!/bin/bash
# ============================================================================
# Gate 7 Nsight Compute Profiling Script
# ============================================================================
# Captures comprehensive metrics for TMA + WGMMA kernel
# ============================================================================

set -e

echo "========================================"
echo "GATE 7 - NSIGHT COMPUTE PROFILING"
echo "========================================"
echo ""

# Configuration
KERNEL_BIN="build/bin/attention_gate7"
PROFILE_DIR="reports/gate7_bundle"
METRICS_FILE="build/results/gate7_metrics.txt"

# Check kernel exists
if [ ! -f "$KERNEL_BIN" ]; then
    echo "❌ Kernel not found: $KERNEL_BIN"
    echo "   Build first: ./build_gate7.sh"
    exit 1
fi

# Check ncu
if ! command -v ncu &> /dev/null; then
    echo "❌ ncu (Nsight Compute) not found"
    echo "   Install: https://developer.nvidia.com/nsight-compute"
    exit 1
fi

mkdir -p $PROFILE_DIR

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "NCU: $(ncu --version | head -1)"
echo ""

# ============================================================================
# Profile 1: Quick Metrics (30 seconds)
# ============================================================================

echo "[1/3] Quick profiling (key metrics only)..."
echo ""

ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    launch__registers_per_thread,\
    launch__shared_mem_per_block_driver,\
    gpu__time_duration.sum \
    --target-processes all \
    --csv \
    $KERNEL_BIN 2>&1 | tee $PROFILE_DIR/quick_metrics.csv

echo ""

# ============================================================================
# Profile 2: Compute Workload Analysis (2 minutes)
# ============================================================================

echo "[2/3] Compute workload analysis..."
echo ""

ncu --set full \
    --section ComputeWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --export $PROFILE_DIR/compute_profile \
    --force-overwrite \
    $KERNEL_BIN 2>&1 | tee $PROFILE_DIR/compute_profile.log

echo ""

# ============================================================================
# Profile 3: Memory Workload Analysis (2 minutes)
# ============================================================================

echo "[3/3] Memory workload analysis..."
echo ""

ncu --set full \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables \
    --export $PROFILE_DIR/memory_profile \
    --force-overwrite \
    $KERNEL_BIN 2>&1 | tee $PROFILE_DIR/memory_profile.log

echo ""

# ============================================================================
# Parse and Display Key Metrics
# ============================================================================

echo "========================================"
echo "KEY METRICS SUMMARY"
echo "========================================"
echo ""

if [ -f "$PROFILE_DIR/quick_metrics.csv" ]; then
    python3 << 'EOF'
import csv
import sys

try:
    with open('reports/gate7_bundle/quick_metrics.csv', 'r') as f:
        lines = [line for line in f if not line.strip().startswith('"ID"') and line.strip()]
        
    if not lines:
        print("No data in CSV")
        sys.exit(0)
    
    # Parse CSV manually (handle Nsight Compute format)
    for line in lines:
        if 'sm__throughput' in line and '%' in line:
            val = line.split(',')[-1].strip().replace('%', '').replace('"', '')
            try:
                print(f"SM Throughput:              {float(val):.1f}%")
            except: pass
        
        elif 'tensor_op_hmma' in line and '%' in line:
            val = line.split(',')[-1].strip().replace('%', '').replace('"', '')
            try:
                print(f"Tensor Core Utilization:    {float(val):.1f}%")
            except: pass
        
        elif 'dram__throughput' in line and '%' in line:
            val = line.split(',')[-1].strip().replace('%', '').replace('"', '')
            try:
                print(f"DRAM Bandwidth:             {float(val):.1f}%")
            except: pass
        
        elif 'warps_active' in line and '%' in line:
            val = line.split(',')[-1].strip().replace('%', '').replace('"', '')
            try:
                print(f"Occupancy (warps active):   {float(val):.1f}%")
            except: pass
        
        elif 'registers_per_thread' in line:
            val = line.split(',')[-1].strip().replace('"', '')
            try:
                print(f"Registers/thread:           {int(float(val))}")
            except: pass
        
        elif 'shared_mem_per_block' in line:
            val = line.split(',')[-1].strip().replace('"', '')
            try:
                smem_kb = int(float(val)) / 1024
                print(f"Shared memory/block:        {smem_kb:.1f} KB")
            except: pass
        
        elif 'gpu__time_duration' in line:
            val = line.split(',')[-1].strip().replace('"', '')
            try:
                time_ms = float(val) / 1e6  # ns to ms
                print(f"Kernel duration:            {time_ms:.3f} ms")
            except: pass

except Exception as e:
    print(f"Error parsing metrics: {e}")
EOF
fi

echo ""
echo "========================================"
echo "PROFILING COMPLETE"
echo "========================================"
echo ""
echo "Generated files:"
echo "  Quick metrics:     $PROFILE_DIR/quick_metrics.csv"
echo "  Compute profile:   $PROFILE_DIR/compute_profile.ncu-rep"
echo "  Memory profile:    $PROFILE_DIR/memory_profile.ncu-rep"
echo ""
echo "View in Nsight Compute GUI:"
echo "  ncu-ui $PROFILE_DIR/compute_profile.ncu-rep"
echo ""
echo "Gate 7 Targets:"
echo "  ✅ SM Throughput ≥ 85%"
echo "  ✅ Tensor Core Util ≥ 97%"
echo "  ✅ DRAM Bandwidth ≥ 95%"
echo "  ✅ Occupancy ≥ 85%"
echo ""
