#!/bin/bash
# Rapid iteration on H100 - run this ON the RunPod
# Builds, tests, profiles in one command

STEP=${1:-"1"}

echo "⚡ Testing Step $STEP"

# Build
nvcc -arch=sm_90a -O3 --use_fast_math -lineinfo --ptxas-options=-v \
    -I. test_wgmma_single_corrected.cu \
    -o test_step${STEP} 2>&1 | grep -E "registers|spill"

# Quick test
./test_step${STEP} 2>&1 | grep -E "Median:|TFLOPS|Status"

# Profile (if requested)
if [ "$2" == "profile" ]; then
    echo "📊 Profiling..."
    ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
        ./test_step${STEP} 2>&1 | tail -20
fi

echo "✅ Done"

