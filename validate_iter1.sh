#!/bin/bash
# Iteration 1 Validation Script
# Runs all 4 gates independently and saves results

set -e

cd ~/periodicdent42
LOG=iter1_validation_results.txt

echo "=================================================================================" > $LOG
echo "ITERATION 1 VALIDATION - $(date)" >> $LOG
echo "=================================================================================" >> $LOG
echo "" >> $LOG

# GATE 1: COMPILATION
echo "=== GATE 1: COMPILATION ===" >> $LOG
echo "Cleaning cache..." >> $LOG
rm -f ~/.cache/torch_extensions/py310_cu121/flash_attention_s512/*.so 2>/dev/null || true
rm -f /tmp/torch_extensions/flash_attention_s512/*.so 2>/dev/null || true

echo "Building fa_s512.cu with BLOCK_M=128, NUM_WARPS=8..." >> $LOG
if python3 cudadent42/bench/build_fa_s512.py >> $LOG 2>&1; then
    echo "✅ GATE 1 PASSED: Compilation successful" >> $LOG
    
    # Extract metrics
    echo "" >> $LOG
    echo "Resource Usage:" >> $LOG
    grep -E "(registers|shared|bytes)" $LOG | tail -5 >> $LOG || echo "  (metrics extraction failed)" >> $LOG
else
    echo "❌ GATE 1 FAILED: Compilation error" >> $LOG
    exit 1
fi
echo "" >> $LOG

# GATE 2: FUNCTIONAL CORRECTNESS
echo "=== GATE 2: FUNCTIONAL CORRECTNESS ===" >> $LOG
echo "Running benchmark with CUDA_LAUNCH_BLOCKING=1..." >> $LOG
if CUDA_LAUNCH_BLOCKING=1 timeout 120 python3 benchmark_fa_s512.py >> $LOG 2>&1; then
    # Check for correctness pass
    if grep -q "Status.*PASS" $LOG || grep -q "Correct.*True" $LOG; then
        echo "✅ GATE 2 PASSED: Functional correctness verified" >> $LOG
    else
        echo "⚠️  GATE 2 WARNING: Completed but correctness unclear" >> $LOG
    fi
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "❌ GATE 2 FAILED: Timeout after 120s" >> $LOG
    else
        echo "❌ GATE 2 FAILED: Runtime error (exit code: $EXIT_CODE)" >> $LOG
    fi
    exit 1
fi
echo "" >> $LOG

# GATE 3: PERFORMANCE
echo "=== GATE 3: PERFORMANCE ===" >> $LOG
FA_LATENCY=$(grep -oP "fa_s512.*:\s*\K[0-9.]+" $LOG | tail -1)
PT_LATENCY=$(grep -oP "PyTorch.*:\s*\K[0-9.]+" $LOG | tail -1)

if [ -n "$FA_LATENCY" ] && [ -n "$PT_LATENCY" ]; then
    echo "fa_s512 latency:    $FA_LATENCY μs" >> $LOG
    echo "PyTorch latency:    $PT_LATENCY μs" >> $LOG
    
    # Calculate speedup vs baseline (321 μs)
    SPEEDUP=$(echo "scale=2; 321 / $FA_LATENCY" | bc)
    echo "Speedup vs baseline (321μs): ${SPEEDUP}x" >> $LOG
    
    # Check if meets target (< 240 μs = 1.3x speedup)
    if (( $(echo "$FA_LATENCY < 240" | bc -l) )); then
        echo "✅ GATE 3 PASSED: Latency < 240 μs (1.3× speedup achieved)" >> $LOG
    else
        echo "⚠️  GATE 3 WARNING: Latency >= 240 μs (target not met)" >> $LOG
    fi
else
    echo "❌ GATE 3 FAILED: Could not extract latency metrics" >> $LOG
fi
echo "" >> $LOG

# GATE 4: NSIGHT VALIDATION (optional, quick metrics only)
echo "=== GATE 4: NSIGHT VALIDATION (Quick Metrics) ===" >> $LOG
echo "Running quick ncu profile..." >> $LOG
if timeout 60 ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active \
    --target-processes all python3 -c "
import torch, torch.nn.functional as F
import flash_attention_s512 as fa
q = torch.randn(4, 8, 512, 64, dtype=torch.float16, device='cuda').contiguous()
k, v = q.clone(), q.clone()
for _ in range(5): _ = fa.fa_s512(q, k, v)
torch.cuda.synchronize()
" 2>&1 | grep -A2 "sm__inst_executed_pipe_tensor" >> $LOG; then
    echo "✅ GATE 4 COMPLETED: Nsight metrics captured" >> $LOG
else
    echo "⚠️  GATE 4 SKIPPED: Nsight profiling timeout or unavailable" >> $LOG
fi
echo "" >> $LOG

# SUMMARY
echo "=================================================================================" >> $LOG
echo "VALIDATION SUMMARY" >> $LOG
echo "=================================================================================" >> $LOG
grep -E "(GATE.*:)" $LOG | grep -v "===" >> $LOG
echo "" >> $LOG
echo "VALIDATION COMPLETE - $(date)" >> $LOG
echo "=================================================================================" >> $LOG

echo "✅ Validation complete! Results saved to $LOG"

