#!/usr/bin/env python3
# Copyright 2025 GOATnote Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlashCore Quick Start Example
==============================

This script demonstrates basic usage of the FlashCore attention kernel.

Requirements:
- NVIDIA GPU with CUDA support
- PyTorch 2.0+
- Triton 2.1+
"""

import torch
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flashcore.fast.attention_production import attention


def main():
    print("=" * 70)
    print("FlashCore Quick Start Example")
    print("=" * 70)
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This example requires a CUDA-capable GPU.")
        return 1
    
    print(f"✅ CUDA available")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Configuration
    batch_size = 16
    num_heads = 8
    seq_len = 512
    head_dim = 64
    
    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads:  {num_heads}")
    print(f"  Seq length: {seq_len}")
    print(f"  Head dim:   {head_dim}")
    print()
    
    # Create input tensors
    print("Creating input tensors...")
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device='cuda', dtype=torch.float16)
    k = q.clone()
    v = q.clone()
    print(f"  Shape: {q.shape}")
    print()
    
    # Run attention
    print("Running FlashCore attention...")
    output = attention(q, k, v)
    print(f"✅ Success! Output shape: {output.shape}")
    print()
    
    # Benchmark
    print("Benchmarking performance...")
    
    # Warmup
    for _ in range(100):
        _ = attention(q, k, v)
    torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(500):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = attention(q, k, v)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # Convert to μs
    
    # Statistics
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    p99 = times[int(len(times) * 0.99)]
    
    per_seq_p50 = p50 / batch_size
    per_seq_p95 = p95 / batch_size
    per_seq_p99 = p99 / batch_size
    
    print()
    print("Performance Results (500 trials):")
    print(f"  Total latency:")
    print(f"    P50: {p50:.2f} μs")
    print(f"    P95: {p95:.2f} μs")
    print(f"    P99: {p99:.2f} μs")
    print()
    print(f"  Per-sequence latency:")
    print(f"    P50: {per_seq_p50:.2f} μs/seq")
    print(f"    P95: {per_seq_p95:.2f} μs/seq")
    print(f"    P99: {per_seq_p99:.2f} μs/seq")
    print()
    
    # Target check
    target_us = 5.0
    if per_seq_p50 < target_us:
        print(f"✅ Target achieved! {per_seq_p50:.2f} < {target_us} μs/seq")
        print(f"   ({target_us / per_seq_p50:.2f}× faster than target)")
    else:
        print(f"⚠️  Target not met: {per_seq_p50:.2f} > {target_us} μs/seq")
        print(f"   (Try increasing batch size for better amortization)")
    print()
    
    # Correctness check
    print("Verifying correctness vs PyTorch SDPA...")
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    max_diff = torch.max(torch.abs(output - ref)).item()
    
    if max_diff < 0.004:  # Within FP16 tolerance
        print(f"✅ Correctness verified! Max difference: {max_diff:.6f}")
    else:
        print(f"⚠️  Large difference detected: {max_diff:.6f}")
    print()
    
    print("=" * 70)
    print("🎉 FlashCore quick start complete!")
    print()
    print("Next steps:")
    print("  - Check docs/getting-started/ for more examples")
    print("  - Run expert_validation.py for comprehensive benchmarks")
    print("  - Explore flashcore/ source code")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())

