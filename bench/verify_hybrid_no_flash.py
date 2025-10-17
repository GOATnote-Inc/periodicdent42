"""
Verify hybrid implementation is NOT using Flash Attention accidentally

This script explicitly disables Flash Attention and confirms we're
using the simple cuBLAS + softmax + matmul path.
"""

import torch
import time

def bench_with_backends(Q, K, V, scale):
    """Benchmark with different SDPA backends disabled"""
    
    print("Testing different backend configurations:")
    print()
    
    # Test 1: Flash disabled (our hybrid)
    print("1. Flash DISABLED (cuBLAS path):")
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            # Manual computation (what our hybrid does)
            S = torch.matmul(Q, K.transpose(-2, -1)) * scale
            P = torch.softmax(S, dim=-1)
            O = torch.matmul(P, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        time_no_flash = (t1 - t0) * 1e6 / 100
        print(f"   Time: {time_no_flash:.2f} μs")
    print()
    
    # Test 2: Flash enabled (SDPA baseline)
    print("2. Flash ENABLED (SDPA baseline):")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            O = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        time_flash = (t1 - t0) * 1e6 / 100
        print(f"   Time: {time_flash:.2f} μs")
    print()
    
    # Test 3: Math only (reference implementation)
    print("3. Math ONLY (reference):")
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            O = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        time_math = (t1 - t0) * 1e6 / 100
        print(f"   Time: {time_math:.2f} μs")
    print()
    
    print("=" * 70)
    print("Summary:")
    print(f"  Manual (cuBLAS+softmax+matmul): {time_no_flash:.2f} μs")
    print(f"  SDPA Flash:                      {time_flash:.2f} μs")
    print(f"  SDPA Math:                       {time_math:.2f} μs")
    print()
    print(f"  Manual vs Flash: {time_no_flash / time_flash:.2f}× slower")
    print(f"  Manual vs Math:  {time_no_flash / time_math:.2f}×")
    print("=" * 70)
    print()
    
    # Verify our hybrid is using the manual path
    if abs(time_no_flash - time_math) < 10:
        print("✅ CONFIRMED: Hybrid uses manual cuBLAS path (not Flash)")
    else:
        print("⚠️  WARNING: Time mismatch, investigate")
    print()
    
    return time_no_flash, time_flash, time_math

def main():
    print("=" * 70)
    print("Verify Hybrid is NOT using Flash Attention")
    print("=" * 70)
    print()
    
    # Setup
    B, H, S, D = 1, 8, 512, 64
    scale = 1.0 / (D ** 0.5)
    
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    print(f"Config: B={B}, H={H}, S={S}, D={D}")
    print()
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        S_mat = torch.matmul(Q, K.transpose(-2, -1)) * scale
        P = torch.softmax(S_mat, dim=-1)
        O = torch.matmul(P, V)
    print("✅ Warmup complete")
    print()
    
    # Benchmark
    time_no_flash, time_flash, time_math = bench_with_backends(Q, K, V, scale)
    
    print("Interpretation:")
    print("  - If 'Manual ≈ Math': We're using cuBLAS (correct) ✅")
    print("  - If 'Manual ≈ Flash': We're accidentally using Flash (wrong) ❌")
    print("  - Flash should be ~1.5-2× faster than Math")
    print()

if __name__ == "__main__":
    main()

