#!/bin/bash
# Profile Phase 4.X Expert kernel on H100
# Get Tensor Core utilization, memory bandwidth, and bottleneck analysis

set -e

echo "========================================"
echo "PROFILING PHASE 4.X EXPERT KERNEL"  
echo "========================================"
echo ""

# Check for ncu
if ! command -v ncu &> /dev/null; then
    echo "❌ ERROR: ncu (Nsight Compute) not found"
    echo "Install from: https://developer.nvidia.com/nsight-compute"
    exit 1
fi

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "NCU Version: $(ncu --version | head -1)"
echo ""

# Build the kernel first
echo "[1/3] Building kernel..."
./build_cuda_simple.sh 2>&1 | grep -E "(Compilation|COMPLETE|error)" || true
echo ""

# Run quick profiling (key metrics only)
echo "[2/3] Profiling - Tensor Core Utilization..."
ncu --metrics \
    sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
    smsp__sass_thread_inst_executed_ops_tensor.sum,\
    sm__sass_inst_executed_op_tensor_op.sum \
    --target-processes all \
    ./build/bin/test_hopper 2>&1 | tail -50

echo ""
echo "[3/3] Profiling - Memory Bandwidth..."
ncu --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__bytes.sum.per_second,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    --target-processes all \
    ./build/bin/test_hopper 2>&1 | tail -50

echo ""
echo "========================================"
echo "PROFILING COMPLETE"
echo "========================================"
echo ""
echo "Key Metrics to Check:"
echo "  1. Tensor Core util: Target >60%"
echo "  2. Memory bandwidth: Target >70% of 3.35 TB/s"
echo "  3. DRAM throughput: Should be primary bottleneck"
echo ""
echo "For full analysis, run:"
echo "  ncu --set full -o phase4x_profile.ncu-rep ./build/bin/test_hopper"
echo "  ncu-ui phase4x_profile.ncu-rep"
echo ""

