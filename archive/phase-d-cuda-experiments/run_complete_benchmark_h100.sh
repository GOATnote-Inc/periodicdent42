#!/bin/bash
# Complete Benchmark: Compile + Run + Compare vs SDPA
set -euo pipefail

RUNPOD_IP="${1:-154.57.34.90}"
RUNPOD_PORT="${2:-36088}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "=========================================="
echo "COMPLETE KERNEL BENCHMARK ON H100"
echo "=========================================="
echo "Building standalone benchmark with actual kernel launch"
echo ""

# Upload
echo "📦 Uploading complete benchmark..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" "mkdir -p /workspace/benchmark_complete"

scp -P "$RUNPOD_PORT" $SSH_OPTS \
    flashcore/benchmark/attention_benchmark.cu \
    root@"$RUNPOD_IP":/workspace/benchmark_complete/

echo "✅ Uploaded"
echo ""

# Build and run
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" 'bash -s' <<'REMOTE'
set -euxo pipefail

cd /workspace/benchmark_complete

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "STEP 1: COMPILE COMPLETE BENCHMARK"
echo "=========================================="

nvcc -std=c++17 -O3 -Xptxas -O3 \
     --use_fast_math \
     -gencode arch=compute_90,code=sm_90 \
     attention_benchmark.cu \
     -o attention_benchmark

echo "✅ Compiled executable"
ls -lh attention_benchmark
echo ""

echo "=========================================="
echo "STEP 2: EXTRACT AND VALIDATE KERNEL SASS"
echo "=========================================="
cuobjdump -sass attention_benchmark > sass_complete.txt

BRANCH_COUNT=$(grep -cP '@P\d+\s+BRA' sass_complete.txt || echo "0")
echo "Predicated branches: $BRANCH_COUNT"

SPILL_COUNT=$(grep -cP '\b(LD|ST)\.LCL' sass_complete.txt || echo "0")
echo "Register spills: $SPILL_COUNT"

SMEM_COUNT=$(grep -c '\.shared\.' sass_complete.txt || echo "0")
echo "Shared memory instructions: $SMEM_COUNT"
echo ""

echo "=========================================="
echo "STEP 3: RUN KERNEL BENCHMARK"
echo "=========================================="
./attention_benchmark 2>&1 | tee benchmark_output.txt

echo ""
echo "=========================================="
echo "STEP 4: COMPARE VS PYTORCH SDPA"
echo "=========================================="

# Quick PyTorch comparison
python3 <<'EOF'
import torch
import torch.nn.functional as F

B, H, S, D = 1, 8, 512, 64
Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(100):
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
torch.cuda.synchronize()

# Benchmark
times = []
for _ in range(1000):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000.0)  # Convert to μs

times.sort()
median = times[len(times)//2]
p99 = times[int(len(times)*0.99)]

print(f"\nPyTorch SDPA:")
print(f"  Median: {median:7.2f} μs")
print(f"  p99:    {p99:7.2f} μs")

with open('sdpa_performance.txt', 'w') as f:
    f.write(f"SDPA_MEDIAN_US={median:.2f}\n")
    f.write(f"SDPA_P99_US={p99:.2f}\n")
EOF

echo ""
echo "=========================================="
echo "RESULTS COMPARISON"
echo "=========================================="
echo "Our Kernel:"
cat kernel_performance.txt

echo ""
echo "PyTorch SDPA:"
cat sdpa_performance.txt

echo ""
# Calculate speedup
python3 <<'EOF'
import re

# Read results
with open('kernel_performance.txt') as f:
    kernel_data = f.read()
    kernel_median = float(re.search(r'KERNEL_MEDIAN_US=([\d.]+)', kernel_data).group(1))

with open('sdpa_performance.txt') as f:
    sdpa_data = f.read()
    sdpa_median = float(re.search(r'SDPA_MEDIAN_US=([\d.]+)', sdpa_data).group(1))

speedup = sdpa_median / kernel_median
target_5x = sdpa_median / 5.0

print("========================================")
print("SPEEDUP ANALYSIS")
print("========================================")
print(f"PyTorch SDPA:    {sdpa_median:7.2f} μs (baseline)")
print(f"Our Kernel:      {kernel_median:7.2f} μs")
print(f"Speedup:         {speedup:.2f}×")
print(f"")
print(f"Target (5×):     {target_5x:7.2f} μs")
if kernel_median < target_5x:
    print(f"Status:          ✅ TARGET ACHIEVED!")
elif kernel_median < sdpa_median:
    print(f"Status:          ✅ FASTER (but not 5× yet)")
    print(f"Remaining:       {kernel_median / target_5x:.2f}× more speedup needed")
else:
    print(f"Status:          ⚠️  SLOWER (needs optimization)")
    print(f"Gap:             {kernel_median / sdpa_median:.2f}× slower")

with open('speedup_results.txt', 'w') as f:
    f.write(f"SDPA_MEDIAN_US={sdpa_median:.2f}\n")
    f.write(f"KERNEL_MEDIAN_US={kernel_median:.2f}\n")
    f.write(f"SPEEDUP={speedup:.2f}\n")
    f.write(f"TARGET_5X_US={target_5x:.2f}\n")
    if kernel_median < target_5x:
        f.write(f"STATUS=TARGET_ACHIEVED\n")
    elif kernel_median < sdpa_median:
        f.write(f"STATUS=FASTER\n")
    else:
        f.write(f"STATUS=SLOWER\n")
EOF

cat speedup_results.txt

exit 0
REMOTE

# Download results
echo ""
echo "⬇️  Downloading results..."
scp -P "$RUNPOD_PORT" $SSH_OPTS \
    root@"$RUNPOD_IP":/workspace/benchmark_complete/speedup_results.txt \
    root@"$RUNPOD_IP":/workspace/benchmark_complete/benchmark_output.txt \
    root@"$RUNPOD_IP":/workspace/benchmark_complete/sass_complete.txt \
    . 2>/dev/null || true

echo ""
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
[ -f speedup_results.txt ] && cat speedup_results.txt

echo ""
echo "✅ Complete benchmark finished"
echo "Results saved to: speedup_results.txt"

