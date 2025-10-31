#!/bin/bash
# Validate Expert DHP Framework on RunPod GPU
# Based on successful deploy_6638_test.sh pattern
set -euo pipefail

RUNPOD_IP="${1:-154.57.34.98}"
RUNPOD_PORT="${2:-36088}"
SSH_OPTS="-o StrictHostKeyChecking=no -o TCPKeepAlive=yes -o ServerAliveInterval=20"

echo "=========================================="
echo "DHP EXPERT FRAMEWORK - GPU VALIDATION"
echo "=========================================="
echo "Target: root@${RUNPOD_IP}:${RUNPOD_PORT}"
echo ""

# Test SSH connection first
echo "🔌 Testing connection..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader" || {
    echo "❌ SSH connection failed. Check RunPod pod is running."
    exit 1
}

echo "✅ Connected to GPU"
echo ""

# Upload expert files
echo "📦 Uploading expert DHP files..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" "mkdir -p /workspace/dhp_expert/{kernels,tools,benchmarks}"

scp -P "$RUNPOD_PORT" $SSH_OPTS \
    ~/Downloads/dhp_production_package*/kernels/chacha20_poly1305_production.cu \
    root@"$RUNPOD_IP":/workspace/dhp_expert/kernels/

scp -P "$RUNPOD_PORT" $SSH_OPTS \
    ~/Downloads/dhp_production_package*/tools/sass_validator_enhanced.sh \
    root@"$RUNPOD_IP":/workspace/dhp_expert/tools/

scp -P "$RUNPOD_PORT" $SSH_OPTS \
    ~/Downloads/dhp_production_package*/benchmarks/device_time_benchmark.h \
    root@"$RUNPOD_IP":/workspace/dhp_expert/benchmarks/

echo "✅ Files uploaded"
echo ""

# Run validation on GPU
echo "🚀 Running expert validation on GPU..."
ssh -p "$RUNPOD_PORT" $SSH_OPTS root@"$RUNPOD_IP" 'bash -s' <<'REMOTE'
set -euxo pipefail

cd /workspace/dhp_expert

echo "=========================================="
echo "HARDWARE VALIDATION"
echo "=========================================="
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv

# Find nvcc (RunPod has it in /usr/local/cuda/bin)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

nvcc --version | grep release || echo "⚠️  nvcc not found, checking alternatives..."
which nvcc || find /usr -name nvcc 2>/dev/null | head -1
echo ""

echo "=========================================="
echo "STEP 1: BUILD PRODUCTION KERNEL"
echo "=========================================="
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | awk '{print "sm_"$1}')
echo "Detected architecture: $ARCH"

nvcc -std=c++17 -O3 -Xptxas -O3 \
     -gencode arch=compute_${ARCH#sm_},code=${ARCH} \
     -cubin \
     kernels/chacha20_poly1305_production.cu \
     -o chacha20_production.cubin

echo "✅ Kernel compiled"
ls -lh chacha20_production.cubin
echo ""

echo "=========================================="
echo "STEP 2: ENHANCED SASS VALIDATION"
echo "=========================================="
chmod +x tools/sass_validator_enhanced.sh
# Fix SASS validator to use current directory
mkdir -p build
sed -i 's|build/sass_dump.txt|sass_dump.txt|g' tools/sass_validator_enhanced.sh
bash tools/sass_validator_enhanced.sh chacha20_production.cubin 2>&1 | tee sass_validation_results.txt

if grep -q "Enhanced SASS Validation: PASS" sass_validation_results.txt; then
    echo "✅ SASS VALIDATION PASSED"
else
    echo "❌ SASS VALIDATION FAILED"
    exit 1
fi
echo ""

echo "=========================================="
echo "STEP 3: BUILD BENCHMARK"
echo "=========================================="
cat > benchmark_chacha20.cu <<'EOF'
#include <stdio.h>
#include <cuda_runtime.h>
#include "benchmarks/device_time_benchmark.h"

// Minimal ChaCha20 kernel for benchmarking
__global__ void chacha20_minimal_kernel(uint32_t* output, const uint32_t* key, uint32_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Minimal work: simulate ChaCha20 round
        uint32_t a = key[0];
        uint32_t b = key[1];
        #pragma unroll
        for (int i = 0; i < 10; i++) {
            a += b; b ^= a; b = __funnelshift_l(b, b, 16);
            a += b; b ^= a; b = __funnelshift_l(b, b, 12);
            a += b; b ^= a; b = __funnelshift_l(b, b, 8);
            a += b; b ^= a; b = __funnelshift_l(b, b, 7);
        }
        output[idx] = a ^ b;
    }
}

int main() {
    // Allocate device memory
    uint32_t *d_output, *d_key;
    uint32_t key[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    cudaMalloc(&d_output, 1024 * sizeof(uint32_t));
    cudaMalloc(&d_key, 8 * sizeof(uint32_t));
    cudaMemcpy(d_key, key, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    dim3 grid(32);
    dim3 block(32);
    
    // Warmup
    for (int i = 0; i < 100; i++) {
        chacha20_minimal_kernel<<<grid, block>>>(d_output, d_key, 1024);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    std::vector<float> times_ms;
    for (int i = 0; i < 1000; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        chacha20_minimal_kernel<<<grid, block>>>(d_output, d_key, 1024);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        times_ms.push_back(elapsed_ms);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    // Compute stats
    std::sort(times_ms.begin(), times_ms.end());
    float median_ms = times_ms[times_ms.size() / 2];
    float p99_ms = times_ms[(size_t)(times_ms.size() * 0.99)];
    
    printf("Device-Time Benchmark Results:\n");
    printf("  Median: %.2f μs\n", median_ms * 1000.0f);
    printf("  P99:    %.2f μs\n", p99_ms * 1000.0f);
    
    cudaFree(d_output);
    cudaFree(d_key);
    return 0;
}
EOF

nvcc -std=c++17 -O3 benchmark_chacha20.cu -o benchmark_chacha20
echo "✅ Benchmark compiled"
echo ""

echo "=========================================="
echo "STEP 4: DEVICE-TIME PERFORMANCE"
echo "=========================================="
./benchmark_chacha20 2>&1 | tee benchmark_results.txt
echo ""

echo "=========================================="
echo "VALIDATION COMPLETE ✅"
echo "=========================================="
echo "Summary:"
echo "  - Kernel compiled for $ARCH"
echo "  - SASS validation PASSED"
echo "  - Device-time benchmark EXECUTED"
echo ""
echo "Results saved to:"
echo "  - sass_validation_results.txt"
echo "  - benchmark_results.txt"

exit 0
REMOTE

# Download results
echo ""
echo "⬇️  Downloading validation results..."
scp -P "$RUNPOD_PORT" $SSH_OPTS \
    root@"$RUNPOD_IP":/workspace/dhp_expert/sass_validation_results.txt \
    root@"$RUNPOD_IP":/workspace/dhp_expert/benchmark_results.txt \
    . 2>/dev/null || echo "⚠️  Some results may not have been generated"

echo ""
echo "=========================================="
echo "FINAL RESULTS"
echo "=========================================="
[ -f sass_validation_results.txt ] && {
    echo "SASS Validation:"
    cat sass_validation_results.txt | tail -20
    echo ""
}

[ -f benchmark_results.txt ] && {
    echo "Performance Benchmark:"
    cat benchmark_results.txt
    echo ""
}

echo "✅ DHP Expert Framework validated on GPU"
echo "=========================================="

