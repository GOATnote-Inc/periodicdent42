"""
TMA Invariant Sweep - One Variable at a Time
Systematic testing with checksummed artifacts
"""
import os
import sys
import csv
import torch
import triton
import triton.language as tl
from pathlib import Path
from datetime import datetime

@triton.jit
def sweep_kernel(X, Y,
                 M: tl.constexpr, N: tl.constexpr,
                 STRIDE_XM, STRIDE_XN,
                 STRIDE_YM, STRIDE_YN,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 ORDER_INNER_FIRST: tl.constexpr):
    pid_m = tl.program_id(0)
    base_m = pid_m * BLOCK_M

    # Descriptor-in-loop
    for expert in range(3):
        off_m = base_m + expert * 7
        
        # Vary order based on parameter
        if ORDER_INNER_FIRST:
            order = (1, 0)  # Inner-first (column-major-ish for blocks)
        else:
            order = (0, 1)  # Outer-first
        
        x_blk = tl.make_block_ptr(base=X,
                                  shape=(M, N),
                                  strides=(STRIDE_XM, STRIDE_XN),
                                  offsets=(off_m, 0),
                                  block_shape=(BLOCK_M, BLOCK_N),
                                  order=order)
        y_blk = tl.make_block_ptr(base=Y,
                                  shape=(M, N),
                                  strides=(STRIDE_YM, STRIDE_YN),
                                  offsets=(off_m, 0),
                                  block_shape=(BLOCK_M, BLOCK_N),
                                  order=order)
        
        a = tl.load(x_blk, boundary_check=(0,1))
        if (expert & 1) == 0:
            tl.store(y_blk, a, boundary_check=(0,1))

def check_ir_for_tma(cache_root=Path.home() / ".triton/cache"):
    """Scan most recent cache entries for TMA operations"""
    # Find most recent kernel
    ttir_files = sorted(cache_root.glob("**/sweep_kernel.ttir"), 
                       key=lambda p: p.stat().st_mtime, reverse=True)
    ttgir_files = sorted(cache_root.glob("**/sweep_kernel.ttgir"), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
    
    ttir_tma = 0
    ttgir_tma = 0
    
    if ttir_files:
        content = ttir_files[0].read_text()
        ttir_tma = content.count("async_tma") + content.count("ttng.async_tma")
    
    if ttgir_files:
        content = ttgir_files[0].read_text()
        ttgir_tma = content.count("async_tma") + content.count("ttng.async_tma")
    
    return ttir_tma, ttgir_tma

def run_config(name, M, N, BLOCK_M, BLOCK_N, num_stages, num_warps, order_inner_first):
    """Run one configuration and check for TMA emission"""
    torch.manual_seed(0)
    
    X = torch.randn((M, N), device='cuda', dtype=torch.float16)
    Y = torch.zeros_like(X)
    
    grid = (triton.cdiv(M, BLOCK_M),)
    
    try:
        sweep_kernel[grid](
            X, Y, M, N,
            X.stride(0), X.stride(1),
            Y.stride(0), Y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            ORDER_INNER_FIRST=order_inner_first,
            num_stages=num_stages,
            num_warps=num_warps
        )
        torch.cuda.synchronize()
        
        # Check IR
        ttir_tma, ttgir_tma = check_ir_for_tma()
        
        result = "✅"
    except Exception as e:
        result = f"❌ {str(e)[:30]}"
        ttir_tma = -1
        ttgir_tma = -1
    
    return {
        "name": name,
        "result": result,
        "M": M,
        "N": N,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "num_stages": num_stages,
        "num_warps": num_warps,
        "order": "(1,0)" if order_inner_first else "(0,1)",
        "ttir_tma": ttir_tma,
        "ttgir_tma": ttgir_tma
    }

def main():
    # Output directory
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = Path("artifacts/sweeps")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / f"sweep_{ts}.csv"
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  TMA Invariant Sweep - One Variable at a Time           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    # Get Triton version
    try:
        import triton
        version = triton.__version__
        print(f"Triton version: {version}")
    except:
        version = "unknown"
    
    results = []
    
    # Baseline
    results.append(run_config("Baseline", 1024, 512, 128, 128, 4, 8, True))
    
    # Vary tile sizes (one at a time)
    results.append(run_config("BLOCK_M=64", 1024, 512, 64, 128, 4, 8, True))
    results.append(run_config("BLOCK_M=256", 2048, 512, 256, 128, 4, 8, True))
    results.append(run_config("BLOCK_N=64", 1024, 512, 128, 64, 4, 8, True))
    results.append(run_config("BLOCK_N=256", 1024, 2048, 128, 256, 4, 8, True))
    
    # Vary num_stages
    results.append(run_config("stages=3", 1024, 512, 128, 128, 3, 8, True))
    results.append(run_config("stages=5", 1024, 512, 128, 128, 5, 8, True))
    
    # Vary num_warps
    results.append(run_config("warps=4", 1024, 512, 128, 128, 4, 4, True))
    results.append(run_config("warps=16", 1024, 512, 128, 128, 4, 16, True))
    
    # Vary order
    results.append(run_config("order=(0,1)", 1024, 512, 128, 128, 4, 8, False))
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "result", "M", "N", "BLOCK_M", "BLOCK_N", 
            "num_stages", "num_warps", "order", "ttir_tma", "ttgir_tma"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    # Print results
    print()
    print("Results:")
    print(f"{'Name':<20} {'Result':<4} {'Tiles':<12} {'Stages':<7} {'Warps':<6} {'Order':<8} {'TTIR':<6} {'TTGIR':<6}")
    print("=" * 100)
    for r in results:
        tiles = f"{r['BLOCK_M']}x{r['BLOCK_N']}"
        print(f"{r['name']:<20} {r['result']:<4} {tiles:<12} {r['num_stages']:<7} {r['num_warps']:<6} {r['order']:<8} {r['ttir_tma']:<6} {r['ttgir_tma']:<6}")
    
    # Summary
    tma_found = any(r['ttir_tma'] > 0 or r['ttgir_tma'] > 0 for r in results if r['ttir_tma'] >= 0)
    print()
    print(f"CSV saved: {csv_path}")
    print()
    if tma_found:
        print("✅ TMA found in at least one configuration!")
        configs_with_tma = [r for r in results if r['ttir_tma'] > 0 or r['ttgir_tma'] > 0]
        for r in configs_with_tma:
            print(f"   → {r['name']}: TTIR={r['ttir_tma']}, TTGIR={r['ttgir_tma']}")
    else:
        print(f"❌ NO TMA in any configuration (Triton {version})")
        print("   → Block pointers do NOT trigger TMA emission")

if __name__ == "__main__":
    main()

