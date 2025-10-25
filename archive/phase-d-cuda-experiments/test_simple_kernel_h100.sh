#!/bin/bash
# Test SIMPLE kernel - learn from failure, iterate
set -euo pipefail

RUNPOD_IP="${1:-154.57.34.90}"
RUNPOD_PORT="${2:-36088}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "=========================================="
echo "ITERATION: SIMPLE KERNEL (Learning from Failure)"
echo "=========================================="
echo "Previous: 40ms (1723× slower)"
echo "Goal: Beat PyTorch (< 24 μs)"
echo ""

ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" "mkdir -p /workspace/simple_test"

scp -P "$RUNPOD_PORT" $SSH_OPTS \
    flashcore/benchmark/attention_simple_fast.cu \
    root@"$RUNPOD_IP":/workspace/simple_test/

ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" 'bash -s' <<'REMOTE'
set -euxo pipefail

cd /workspace/simple_test

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "COMPILE SIMPLE KERNEL"
echo "=========================================="

nvcc -std=c++17 -O3 -Xptxas -O3 --use_fast_math \
     -gencode arch=compute_90,code=sm_90 \
     attention_simple_fast.cu -o simple_benchmark

echo "✅ Compiled ($(ls -lh simple_benchmark | awk '{print $5}'))"
echo ""

echo "=========================================="
echo "RUN SIMPLE KERNEL"
echo "=========================================="
./simple_benchmark 2>&1 | tee simple_output.txt

echo ""
echo "=========================================="
echo "COMPARE VS PYTORCH SDPA"
echo "=========================================="
python3 <<'EOF'
import torch
import torch.nn.functional as F
import re

# Quick SDPA benchmark
B, H, S, D = 1, 8, 512, 64
Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

for _ in range(100):
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
torch.cuda.synchronize()

times = []
for _ in range(1000):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000.0)

times.sort()
sdpa_median = times[len(times)//2]

# Load our kernel result
with open('simple_kernel_perf.txt') as f:
    data = f.read()
    kernel_median = float(re.search(r'SIMPLE_KERNEL_MEDIAN_US=([\d.]+)', data).group(1))

speedup = sdpa_median / kernel_median
target_5x = sdpa_median / 5.0

print("\n========================================")
print("RESULTS")
print("========================================")
print(f"PyTorch SDPA:     {sdpa_median:8.2f} μs")
print(f"Simple Kernel:    {kernel_median:8.2f} μs")
print(f"Speedup:          {speedup:8.2f}×")
print(f"")
print(f"Target (5×):      {target_5x:8.2f} μs")

if kernel_median < target_5x:
    print(f"Status:           ✅ TARGET ACHIEVED!")
    status = "TARGET_ACHIEVED"
elif kernel_median < sdpa_median:
    gap = kernel_median / target_5x
    print(f"Status:           ✅ FASTER than PyTorch!")
    print(f"Remaining:        {gap:.2f}× more speedup needed for 5× target")
    status = "FASTER"
else:
    gap = kernel_median / sdpa_median
    print(f"Status:           ⚠️  Still slower")
    print(f"Gap:              {gap:.2f}× slower than PyTorch")
    status = "SLOWER"

with open('simple_comparison.txt', 'w') as f:
    f.write(f"SDPA_MEDIAN_US={sdpa_median:.2f}\n")
    f.write(f"SIMPLE_KERNEL_MEDIAN_US={kernel_median:.2f}\n")
    f.write(f"SPEEDUP={speedup:.3f}\n")
    f.write(f"STATUS={status}\n")
    f.write(f"TARGET_5X_US={target_5x:.2f}\n")
EOF

cat simple_comparison.txt

exit 0
REMOTE

scp -P "$RUNPOD_PORT" $SSH_OPTS \
    root@"$RUNPOD_IP":/workspace/simple_test/simple_comparison.txt \
    . 2>/dev/null || true

echo ""
echo "=========================================="
echo "ITERATION COMPLETE"
echo "=========================================="
[ -f simple_comparison.txt ] && cat simple_comparison.txt

echo ""
echo "Next: If faster, optimize further. If slower, try different approach."

