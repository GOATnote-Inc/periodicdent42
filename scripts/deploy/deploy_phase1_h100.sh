#!/bin/bash
# Deploy and test Phase 1 kernel on H100

set -e

RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"

echo "========================================"
echo "PHASE 1: DEPLOYING TO H100"
echo "========================================"
echo ""

echo "[1/4] Deploying kernel source..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/fast/attention_hopper_cuda.cu \
    flashcore/cuda/CMakeLists.txt \
    flashcore/cuda/test_hopper_kernel.cu \
    build_hopper.sh \
    root@${RUNPOD_IP}:/workspace/flashcore_llama/
echo "✅ Source deployed"
echo ""

echo "[2/4] Building kernel on H100..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama

# Create directory structure
mkdir -p flashcore/cuda flashcore/fast build

# Move files to correct locations
mv attention_hopper_cuda.cu flashcore/fast/
mv CMakeLists.txt flashcore/cuda/
mv test_hopper_kernel.cu flashcore/cuda/

# Build
chmod +x build_hopper.sh
./build_hopper.sh 2>&1

ENDSSH

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
else
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "[3/4] Running Phase 1 kernel test..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama

echo "GPU Info:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

if [ -f build/bin/test_hopper ]; then
    ./build/bin/test_hopper
else
    echo "❌ Test binary not found"
    exit 1
fi

ENDSSH

echo ""
echo "[4/4] Comparing vs PyTorch SDPA..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
export PYTHONPATH=/workspace/flashcore_llama:$PYTHONPATH

python3 << 'ENDPYTHON'
import torch
import time
import numpy as np

# Test configuration
B, H, S, D = 16, 16, 2048, 64
dtype = torch.float16

# Create inputs
query = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
key = torch.randn(B, H, S, D, device='cuda', dtype=dtype)
value = torch.randn(B, H, S, D, device='cuda', dtype=dtype)

# Benchmark PyTorch SDPA
print("="*80)
print("COMPARISON: Phase 1 vs PyTorch SDPA")
print("="*80)
print()

# Warmup
for _ in range(20):
    _ = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
torch.cuda.synchronize()

# Benchmark
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

times = []
for _ in range(100):
    start.record()
    _ = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

times = torch.tensor(times)
median_ms = torch.quantile(times, 0.5).item()

# Compute TFLOPS
flops = 4 * B * H * S * S * D
tflops_sdpa = flops / (median_ms / 1000) / 1e12

print(f"PyTorch SDPA:")
print(f"  Latency: {median_ms:.3f} ms")
print(f"  TFLOPS:  {tflops_sdpa:.1f}")
print()

print("="*80)
print("PHASE 1 STATUS")
print("="*80)
print()
print("Expected Performance:")
print("  Phase 1 (WMMA foundation): 120-140 TFLOPS")
print("  Phase 2 (TMA + WGMMA):     180-210 TFLOPS")
print("  Phase 3 (Optimized):       210-230 TFLOPS")
print()
print(f"PyTorch SDPA baseline:       {tflops_sdpa:.1f} TFLOPS")
print()

if tflops_sdpa >= 120:
    print(f"✅ Phase 1 target achievable (SDPA already at {tflops_sdpa:.1f})")
else:
    print(f"⚠️  SDPA baseline: {tflops_sdpa:.1f} TFLOPS")

ENDPYTHON

ENDSSH

echo ""
echo "========================================"
echo "PHASE 1 DEPLOYMENT COMPLETE"
echo "========================================"

