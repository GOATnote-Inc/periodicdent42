#!/bin/bash
# Measure Stage 1 baseline TFLOPS on H100

set -e

RUNPOD_IP="154.57.34.90"
RUNPOD_PORT="14727"

echo "========================================="
echo "MEASURING STAGE 1 BASELINE TFLOPS"
echo "========================================="
echo ""

ssh -p ${RUNPOD_PORT} -o StrictHostKeyChecking=no root@${RUNPOD_IP} << 'ENDSSH'
cd /workspace/flashcore_llama
export PYTHONPATH=/workspace/flashcore_llama:$PYTHONPATH

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "Running baseline benchmark (USE_WARP_SPEC=False)..."
echo ""

python3 << 'ENDPY'
import torch
import numpy as np
from flashcore.fast.attention_stage5_warpspec import attention_stage5, benchmark_stage5

# Test configuration (same as Stage 1 validation)
config = {
    'B': 16,
    'H': 16,
    'S': 2048,
    'D': 64,
    'use_warp_spec': False,  # Baseline (all warps as consumers)
    'num_producer_warps': 2,
    'use_fast_exp': False,
}

print("Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")
print()

# Benchmark
print("Benchmarking (100 iterations)...")
results = benchmark_stage5(config, warmup=20, iters=100)

# Compute TFLOPS
B, H, S, D = config['B'], config['H'], config['S'], config['D']
flops = 4 * B * H * S * S * D
tflops = flops / (results['p50'] / 1000) / 1e12

print()
print("="*60)
print("BASELINE PERFORMANCE (STAGE 1)")
print("="*60)
print(f"Median (p50):  {results['p50']:.3f} ms")
print(f"p95:           {results['p95']:.3f} ms")
print(f"p99:           {results['p99']:.3f} ms")
print(f"Mean:          {results['mean']:.3f} ms")
print(f"Std:           {results['std']:.3f} ms")
print()
print(f"TFLOPS:        {tflops:.1f}")
print()
print("Expected for Stage 1: 40-60 TFLOPS (unoptimized baseline)")
print("Target for Stage 2:   110 TFLOPS (warp-level sync)")
print("="*60)

ENDPY

ENDSSH

echo ""
echo "========================================="
echo "BASELINE MEASUREMENT COMPLETE"
echo "========================================="

