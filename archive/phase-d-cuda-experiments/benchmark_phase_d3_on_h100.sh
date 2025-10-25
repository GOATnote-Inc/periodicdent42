#!/bin/bash
# Benchmark Phase D.3 (WMMA/Shared Memory) vs PyTorch SDPA
set -euo pipefail

RUNPOD_IP="${1:-154.57.34.90}"
RUNPOD_PORT="${2:-36088}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "=========================================="
echo "PHASE D.3: WMMA/SHARED MEMORY BENCHMARK"
echo "=========================================="
echo "Target: 10-20 μs (2× faster than SDPA 24.83 μs)"
echo ""

# Upload
echo "📦 Uploading Phase D.3 kernel..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" "mkdir -p /workspace/phase_d3"

scp -P "$RUNPOD_PORT" $SSH_OPTS \
    flashcore/kernels/attention_phase_d3_wmma.cu \
    root@"$RUNPOD_IP":/workspace/phase_d3/

echo "✅ Uploaded"
echo ""

# Compile and benchmark
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" 'bash -s' <<'REMOTE'
set -euxo pipefail

cd /workspace/phase_d3

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "STEP 1: COMPILE PHASE D.3"
echo "=========================================="

nvcc -std=c++17 -O3 -Xptxas -O3 \
     --use_fast_math \
     -gencode arch=compute_90,code=sm_90 \
     -cubin \
     attention_phase_d3_wmma.cu \
     -o attention_d3.cubin 2>&1 | tee compile_d3.log

echo "✅ Compiled"
ls -lh attention_d3.cubin
echo ""

echo "=========================================="
echo "STEP 2: SASS VALIDATION"
echo "=========================================="
cuobjdump -sass attention_d3.cubin > sass_d3.txt

BRANCH_COUNT=$(grep -cP '@P\d+\s+BRA' sass_d3.txt || echo "0")
echo "Predicated branches: $BRANCH_COUNT"

SPILL_COUNT=$(grep -cP '\b(LD|ST)\.LCL' sass_d3.txt || echo "0")
echo "Register spills: $SPILL_COUNT"

# Check for shared memory usage
SMEM_USAGE=$(grep -c '\.shared\.' sass_d3.txt || echo "0")
echo "Shared memory instructions: $SMEM_USAGE"
echo ""

echo "=========================================="
echo "STEP 3: CREATE BENCHMARK WITH D.3 KERNEL"
echo "=========================================="
cat > benchmark_d3.py <<'EOF'
import torch
import torch.nn.functional as F
import ctypes
import time
import numpy as np

# Configuration
B, H, S, D = 1, 8, 512, 64
device = 'cuda'
dtype = torch.float16

print(f"Configuration: B={B}, H={H}, S={S}, D={D}")
print(f"Device: {torch.cuda.get_device_name()}")
print("")

# Create inputs
torch.manual_seed(42)
Q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=False)
K = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=False)
V = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=False)
O = torch.zeros(B, H, S, D, device=device, dtype=dtype)

# Load compiled kernel
try:
    cuda = ctypes.CDLL('libcudart.so')
    
    # Load cubin
    module = ctypes.c_void_p()
    cubin_file = 'attention_d3.cubin'
    
    # Try to load with cuModuleLoad (requires CUDA driver API)
    print("Note: Full kernel launch requires CUDA driver API integration")
    print("For now, using PyTorch SDPA as reference")
    print("")
except Exception as e:
    print(f"Kernel loading note: {e}")
    print("Using PyTorch SDPA for comparison")
    print("")

# Benchmark PyTorch SDPA (baseline)
print("========================================")
print("PYTORCH SDPA BASELINE")
print("========================================")

# Warmup
for _ in range(100):
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
torch.cuda.synchronize()

times_ms = []
for i in range(1000):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    O_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    times_ms.append(elapsed_ms)

times_us = [t * 1000.0 for t in times_ms]
times_us.sort()

sdpa_median = times_us[len(times_us) // 2]
sdpa_p99 = times_us[int(len(times_us) * 0.99)]

print(f"PyTorch SDPA:")
print(f"  Median: {sdpa_median:7.2f} μs")
print(f"  p99:    {sdpa_p99:7.2f} μs")
print("")

# Save results
with open('benchmark_d3_results.txt', 'w') as f:
    f.write(f"SDPA_MEDIAN_US={sdpa_median:.2f}\n")
    f.write(f"SDPA_P99_US={sdpa_p99:.2f}\n")
    f.write(f"TARGET_5X_US={sdpa_median/5:.2f}\n")
    f.write(f"TARGET_2X_US={sdpa_median/2:.2f}\n")
    f.write(f"\n")
    f.write(f"# Phase D.3 kernel compilation successful\n")
    f.write(f"# Next: Integrate kernel launch for actual performance test\n")

print("========================================")
print("RESULTS SUMMARY")
print("========================================")
print(f"SDPA Baseline:  {sdpa_median:.2f} μs")
print(f"Target (2×):    {sdpa_median/2:.2f} μs")
print(f"Target (5×):    {sdpa_median/5:.2f} μs")
print("")
print("✅ Phase D.3 kernel compiled successfully")
print("Next step: Integrate kernel launch for actual performance test")
EOF

python3 benchmark_d3.py 2>&1 | tee benchmark_d3_output.txt

echo ""
echo "=========================================="
echo "PHASE D.3 STATUS"
echo "=========================================="
cat benchmark_d3_results.txt
echo ""

# Create summary
cat > phase_d3_summary.txt <<EOF
PHASE D.3 VALIDATION SUMMARY
============================

Compilation: SUCCESS
Cubin Size: $(ls -lh attention_d3.cubin | awk '{print $5}')
Branches: $BRANCH_COUNT
Spills: $SPILL_COUNT
Shared Memory Instructions: $SMEM_USAGE

Next Steps:
1. Integrate kernel launch (requires CUDA driver API wrapper)
2. Benchmark actual D.3 performance
3. Compare to SDPA baseline (24.83 μs)
4. If slower: Optimize (likely due to unoptimized WMMA)
5. If faster: Proceed to D.4 (true WMMA implementation)

Status: Kernel compiled, ready for integration
EOF

cat phase_d3_summary.txt

exit 0
REMOTE

# Download results
echo ""
echo "⬇️  Downloading results..."
scp -P "$RUNPOD_PORT" $SSH_OPTS \
    root@"$RUNPOD_IP":/workspace/phase_d3/benchmark_d3_results.txt \
    root@"$RUNPOD_IP":/workspace/phase_d3/phase_d3_summary.txt \
    root@"$RUNPOD_IP":/workspace/phase_d3/sass_d3.txt \
    root@"$RUNPOD_IP":/workspace/phase_d3/compile_d3.log \
    . 2>/dev/null || echo "Some files not found"

echo ""
echo "=========================================="
echo "PHASE D.3 COMPLETE"
echo "=========================================="
[ -f phase_d3_summary.txt ] && cat phase_d3_summary.txt

echo ""
echo "Next: Create kernel launcher for actual performance test"

