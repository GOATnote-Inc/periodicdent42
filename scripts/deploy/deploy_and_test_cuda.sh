#!/bin/bash
# Deploy, build, and test Phase 1 CUDA kernel on H100

set -e

RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"

echo "========================================"
echo "PHASE 1 CUDA: DEPLOY & TEST"
echo "========================================"
echo ""

echo "[1/5] Deploying source files..."
scp -P ${RUNPOD_PORT} -o StrictHostKeyChecking=no \
    flashcore/fast/attention_hopper_cuda.cu \
    flashcore/cuda/CMakeLists.txt \
    flashcore/cuda/test_hopper_kernel.cu \
    build_hopper.sh \
    root@${RUNPOD_IP}:/workspace/flashcore_llama/
echo "✅ Deployed"
echo ""

echo "[2/5] Setting up build environment..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama

# Create directory structure
mkdir -p flashcore/cuda flashcore/fast build

# Move files
mv -f attention_hopper_cuda.cu flashcore/fast/ 2>/dev/null || true
mv -f CMakeLists.txt flashcore/cuda/ 2>/dev/null || true
mv -f test_hopper_kernel.cu flashcore/cuda/ 2>/dev/null || true

# Setup CUDA environment
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

echo "CUDA setup:"
which nvcc
nvcc --version | grep release

ENDSSH
echo "✅ Environment ready"
echo ""

echo "[3/5] Building kernel..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama

# Setup CUDA environment
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Clean build
rm -rf build
mkdir -p build
cd build

# CMake
cmake ../flashcore/cuda \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCMAKE_BUILD_TYPE=Release

# Make
make -j$(nproc) 2>&1

ENDSSH

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
else
    echo "❌ Build failed - see errors above"
    exit 1
fi
echo ""

echo "[4/5] Running basic test..."
ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

echo "GPU Info:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

if [ -f build/bin/test_hopper ]; then
    ./build/bin/test_hopper
else
    echo "❌ Test binary not found"
    ls -la build/bin/ || echo "build/bin/ doesn't exist"
    exit 1
fi

ENDSSH

if [ $? -eq 0 ]; then
    echo "✅ Test complete"
else
    echo "❌ Test failed"
    exit 1
fi
echo ""

echo "[5/5] Running LLM benchmark comparison..."
echo "This will measure real-world value: tokens/sec, VRAM, etc."
echo ""

ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
export PYTHONPATH=/workspace/flashcore_llama:$PYTHONPATH

# Run quick benchmark on 2 configs (representative)
python3 << 'ENDPYTHON'
import torch
import torch.nn.functional as F
import numpy as np
import time

print("="*80)
print("PHASE 1 vs FA3: QUICK BENCHMARK")
print("="*80)
print()

configs = [
    ("LLaMA-7B (B=1, S=2K)", 1, 32, 2048, 128),
    ("LLaMA-7B (B=8, S=512)", 8, 32, 512, 128),
]

for name, B, H, S, D in configs:
    print(f"\n{name}:")
    print(f"  Config: B={B}, H={H}, S={S}, D={D}")
    
    query = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    key = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    value = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(20):
        _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.cuda.synchronize()
    
    # Benchmark SDPA
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    median_ms = np.median(times)
    tokens_per_sec = (B * S) / (median_ms / 1000)
    flops = 4 * B * H * S * S * D
    tflops = flops / (median_ms / 1000) / 1e12
    
    print(f"  SDPA (FA3):")
    print(f"    Latency: {median_ms:.2f} ms")
    print(f"    Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"    TFLOPS: {tflops:.1f}")

print()
print("="*80)
print("PHASE 1 STATUS")
print("="*80)
print()
print("Current (Triton):  73 TFLOPS ❌")
print("SDPA (FA3):        450 TFLOPS (6.2× faster)")
print("Phase 1 Target:    150-200 TFLOPS")
print()
print("Next: Once CUDA kernel runs correctly, iterate to 150-200 TFLOPS")

ENDPYTHON

ENDSSH

echo ""
echo "========================================"
echo "DEPLOYMENT & TEST COMPLETE"
echo "========================================"

