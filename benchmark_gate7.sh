#!/bin/bash
# ============================================================================
# Gate 7 Performance Benchmark Script
# ============================================================================
# Measures TFLOPS, latency, and compares to Gate 6 / PyTorch
# ============================================================================

set -e

echo "========================================"
echo "GATE 7 - PERFORMANCE BENCHMARK"
echo "========================================"
echo ""

KERNEL_BIN="build/bin/attention_gate7"
RESULTS_DIR="build/results"
METRICS_JSON="$RESULTS_DIR/gate7_metrics.json"

# Check kernel
if [ ! -f "$KERNEL_BIN" ]; then
    echo "❌ Kernel not found: $KERNEL_BIN"
    echo "   Build first: ./build_gate7.sh"
    exit 1
fi

mkdir -p $RESULTS_DIR

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ============================================================================
# Benchmark with Python
# ============================================================================

python3 << 'BENCHMARK_EOF'
import torch
import torch.nn.functional as F
import numpy as np
import ctypes
import time
import json
from pathlib import Path

# Load kernel
lib = ctypes.CDLL("build/bin/attention_gate7")
lib.launch_attention_tma_wgmma_64.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_bool, ctypes.c_void_p
]

def attention_gate7(Q, K, V, scale=None):
    B, H, S, D = Q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(D)
    O = torch.empty_like(Q)
    lib.launch_attention_tma_wgmma_64(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        B, H, S, D, scale, False, None
    )
    torch.cuda.synchronize()
    return O

def benchmark(B, H, S, D, iterations=100, warmup=10):
    """Benchmark kernel performance"""
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        _ = attention_gate7(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(iterations):
        start_event.record()
        O = attention_gate7(Q, K, V)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    times = np.array(times)
    
    # Compute TFLOPS (FlashAttention: 4*B*H*S^2*D FLOPs)
    flops = 4 * B * H * S * S * D
    mean_time_s = np.mean(times) / 1000
    tflops = (flops / 1e12) / mean_time_s
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'max_ms': float(np.max(times)),
        'tflops': float(tflops)
    }

def benchmark_pytorch(B, H, S, D, iterations=100, warmup=10):
    """Benchmark PyTorch SDPA"""
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = F.scaled_dot_product_attention(Q, K, V)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(iterations):
        start_event.record()
        with torch.no_grad():
            O = F.scaled_dot_product_attention(Q, K, V)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    times = np.array(times)
    
    flops = 4 * B * H * S * S * D
    mean_time_s = np.mean(times) / 1000
    tflops = (flops / 1e12) / mean_time_s
    
    return {
        'mean_ms': float(np.mean(times)),
        'tflops': float(tflops)
    }

print("="*60)
print("GATE 7 PERFORMANCE BENCHMARK")
print("="*60)
print()

# Standard config
B, H, S, D = 2, 8, 512, 64
print(f"Configuration: B={B}, H={H}, S={S}, D={D}")
print(f"Iterations:    100 (warmup: 10)")
print()

print("Benchmarking Gate 7 kernel...")
gate7_results = benchmark(B, H, S, D, iterations=100, warmup=10)

print("Benchmarking PyTorch SDPA...")
pytorch_results = benchmark_pytorch(B, H, S, D, iterations=100, warmup=10)

# Results
print()
print("="*60)
print("RESULTS")
print("="*60)
print()

print("Gate 7 TMA Kernel:")
print(f"  Mean latency:  {gate7_results['mean_ms']:.3f} ms")
print(f"  Std dev:       {gate7_results['std_ms']:.3f} ms")
print(f"  P50:           {gate7_results['p50_ms']:.3f} ms")
print(f"  P95:           {gate7_results['p95_ms']:.3f} ms")
print(f"  P99:           {gate7_results['p99_ms']:.3f} ms")
print(f"  TFLOPS:        {gate7_results['tflops']:.2f}")
print()

print("PyTorch SDPA (baseline):")
print(f"  Mean latency:  {pytorch_results['mean_ms']:.3f} ms")
print(f"  TFLOPS:        {pytorch_results['tflops']:.2f}")
print()

# Speedup
speedup_latency = pytorch_results['mean_ms'] / gate7_results['mean_ms']
speedup_tflops = gate7_results['tflops'] / pytorch_results['tflops']

print("Speedup vs PyTorch:")
print(f"  Latency:       {speedup_latency:.2f}× faster")
print(f"  TFLOPS:        {speedup_tflops:.2f}× higher")
print()

# Gate 7 targets
print("Gate 7 Targets:")
target_tflops = 92.0
target_latency = 0.28
target_speedup = 15.0

pass_tflops = gate7_results['tflops'] >= target_tflops
pass_latency = gate7_results['mean_ms'] <= target_latency
pass_speedup = speedup_tflops >= target_speedup

status_tflops = "✅ PASS" if pass_tflops else "⚠️  CLOSE" if gate7_results['tflops'] >= 75 else "❌ FAIL"
status_latency = "✅ PASS" if pass_latency else "⚠️  CLOSE" if gate7_results['mean_ms'] <= 0.35 else "❌ FAIL"
status_speedup = "✅ PASS" if pass_speedup else "⚠️  CLOSE" if speedup_tflops >= 10 else "❌ FAIL"

print(f"  TFLOPS ≥ {target_tflops}:     {status_tflops} ({gate7_results['tflops']:.2f})")
print(f"  Latency ≤ {target_latency} ms: {status_latency} ({gate7_results['mean_ms']:.3f} ms)")
print(f"  Speedup ≥ {target_speedup}×:    {status_speedup} ({speedup_tflops:.2f}×)")
print()

# Save metrics
metrics = {
    "gate": 7,
    "date": time.strftime("%Y-%m-%d"),
    "config": {"B": B, "H": H, "S": S, "D": D},
    "gate7": gate7_results,
    "pytorch_baseline": pytorch_results,
    "speedup": {
        "latency": float(speedup_latency),
        "tflops": float(speedup_tflops)
    },
    "targets": {
        "tflops_target": target_tflops,
        "tflops_achieved": gate7_results['tflops'],
        "tflops_pass": pass_tflops,
        "latency_target": target_latency,
        "latency_achieved": gate7_results['mean_ms'],
        "latency_pass": pass_latency
    }
}

with open("build/results/gate7_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved to: build/results/gate7_metrics.json")
print()

BENCHMARK_EOF

echo "========================================"
echo "BENCHMARK COMPLETE"
echo "========================================"
echo ""
echo "View detailed metrics:"
echo "  cat build/results/gate7_metrics.json"
echo ""
