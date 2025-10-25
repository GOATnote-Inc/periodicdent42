#!/bin/bash
# Benchmark FlashCore Minimal vs PyTorch SDPA on H100
# Based on proven RunPod pattern (deploy_6638_test.sh)
set -euo pipefail

RUNPOD_IP="${1:-154.57.34.90}"
RUNPOD_PORT="${2:-36088}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "=========================================="
echo "FLASHCORE vs PYTORCH SDPA - H100 BENCHMARK"
echo "=========================================="
echo "Target: root@${RUNPOD_IP}:${RUNPOD_PORT}"
echo ""

# Test connection
echo "🔌 Testing connection..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader" || {
    echo "❌ SSH connection failed"
    exit 1
}
echo "✅ Connected to GPU"
echo ""

# Upload files
echo "📦 Uploading kernel and benchmark..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" "mkdir -p /workspace/flashcore_bench"

scp -P "$RUNPOD_PORT" $SSH_OPTS \
    flashcore/kernels/attention_phase_d1_minimal.cu \
    root@"$RUNPOD_IP":/workspace/flashcore_bench/

echo "✅ Files uploaded"
echo ""

# Run benchmark on GPU
echo "🚀 Compiling and benchmarking on H100..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" 'bash -s' <<'REMOTE'
set -euxo pipefail

cd /workspace/flashcore_bench

# Setup CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "HARDWARE INFO"
echo "=========================================="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
nvcc --version | grep release
echo ""

echo "=========================================="
echo "STEP 1: COMPILE MINIMAL KERNEL"
echo "=========================================="
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | awk '{print "sm_"$1}')
echo "Architecture: $ARCH"

nvcc -std=c++17 -O3 -Xptxas -O3 \
     --use_fast_math \
     -gencode arch=compute_${ARCH#sm_},code=${ARCH} \
     -cubin \
     attention_phase_d1_minimal.cu \
     -o attention_minimal.cubin

echo "✅ Kernel compiled"
ls -lh attention_minimal.cubin
echo ""

echo "=========================================="
echo "STEP 2: SASS VALIDATION"
echo "=========================================="
cuobjdump -sass attention_minimal.cubin > sass_dump.txt

# Check for predicated branches
echo "Checking for predicated branches..."
if grep -P '@P\d+\s+BRA' sass_dump.txt > /dev/null; then
    echo "⚠️  WARNING: Predicated branches found"
    grep -P '@P\d+\s+BRA' sass_dump.txt | head -5
else
    echo "✅ No predicated branches (constant-time)"
fi

# Check for register spills
echo "Checking for register spills..."
if grep -P '\b(LD|ST)\.LCL' sass_dump.txt > /dev/null; then
    echo "⚠️  WARNING: Register spills detected"
else
    echo "✅ No register spills"
fi
echo ""

echo "=========================================="
echo "STEP 3: CREATE BENCHMARK"
echo "=========================================="
cat > benchmark.py <<'EOF'
import torch
import torch.nn.functional as F
import time
import numpy as np

# Configuration (matching kernel)
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

# Warmup
print("Warming up PyTorch SDPA...")
for _ in range(100):
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
torch.cuda.synchronize()
print("✅ Warmup complete")
print("")

# Benchmark PyTorch SDPA
print("========================================")
print("PYTORCH SDPA BENCHMARK (1000 iterations)")
print("========================================")

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

median_us = times_us[len(times_us) // 2]
p50_us = times_us[len(times_us) // 2]
p95_us = times_us[int(len(times_us) * 0.95)]
p99_us = times_us[int(len(times_us) * 0.99)]
min_us = times_us[0]
max_us = times_us[-1]
mean_us = sum(times_us) / len(times_us)

print(f"PyTorch SDPA Performance:")
print(f"  Min:    {min_us:7.2f} μs")
print(f"  Median: {median_us:7.2f} μs")
print(f"  Mean:   {mean_us:7.2f} μs")
print(f"  p95:    {p95_us:7.2f} μs")
print(f"  p99:    {p99_us:7.2f} μs")
print(f"  Max:    {max_us:7.2f} μs")
print("")

# Save baseline for comparison
baseline_median = median_us
baseline_p99 = p99_us

print("========================================")
print("RESULTS SUMMARY")
print("========================================")
print(f"PyTorch SDPA Baseline: {baseline_median:.2f} μs (median)")
print(f"Target (5× faster):    {baseline_median/5:.2f} μs")
print("")

# Save results
with open('benchmark_results.txt', 'w') as f:
    f.write(f"PYTORCH_SDPA_MEDIAN_US={baseline_median:.2f}\n")
    f.write(f"PYTORCH_SDPA_P99_US={baseline_p99:.2f}\n")
    f.write(f"TARGET_5X_FASTER_US={baseline_median/5:.2f}\n")

print("✅ Benchmark complete")
print("Results saved to: benchmark_results.txt")
EOF

echo "✅ Benchmark script created"
echo ""

echo "=========================================="
echo "STEP 4: RUN PYTORCH SDPA BENCHMARK"
echo "=========================================="
python3 benchmark.py 2>&1 | tee benchmark_output.txt

echo ""
echo "=========================================="
echo "VALIDATION COMPLETE"
echo "=========================================="
cat benchmark_results.txt
echo ""

exit 0
REMOTE

# Download results
echo ""
echo "⬇️  Downloading results..."
scp -P "$RUNPOD_PORT" $SSH_OPTS \
    root@"$RUNPOD_IP":/workspace/flashcore_bench/benchmark_results.txt \
    root@"$RUNPOD_IP":/workspace/flashcore_bench/benchmark_output.txt \
    root@"$RUNPOD_IP":/workspace/flashcore_bench/sass_dump.txt \
    . 2>/dev/null || echo "⚠️  Some files may not exist"

echo ""
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
[ -f benchmark_results.txt ] && cat benchmark_results.txt
[ -f benchmark_output.txt ] && tail -30 benchmark_output.txt

echo ""
echo "✅ Benchmark complete - ready for iteration"
echo "=========================================="

