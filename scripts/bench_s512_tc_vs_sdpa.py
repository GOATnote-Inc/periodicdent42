#!/usr/bin/env python3
"""
S=512 narrow benchmark: TC vs SDPA (canon_3 shape)
Generates JSON artifact for evidence
"""
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = False


def bench(fn, Q, K, V, s, c, warmup=20, n=200):
    """Benchmark a function with warmup"""
    for _ in range(warmup):
        fn(Q, K, V, s, c)
    torch.cuda.synchronize()
    
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn(Q, K, V, s, c)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 90))


def main():
    out_dir = Path("cudadent42/artifacts/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to import TC kernel
    try:
        from cudadent42.bench.fa_tc_s512 import flash_attention_tc_s512_forward as tc_fwd
        tc_available = True
    except (ImportError, AttributeError) as e:
        print(f"⚠️  TC kernel not available: {e}")
        tc_available = False
    
    # Canon_3: B=2, H=8, S=512, D=64, non-causal
    B, H, S, D = 2, 8, 512, 64
    
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    scale = 1.0 / (D ** 0.5)
    
    # Benchmark SDPA
    sdpa = lambda q, k, v, sc, ca: F.scaled_dot_product_attention(
        q, k, v, is_causal=ca, scale=sc
    )
    
    print("Benchmarking SDPA...")
    p50_sdpa, p90_sdpa = bench(sdpa, Q, K, V, scale, False)
    
    # Benchmark TC (if available)
    if tc_available:
        print("Benchmarking TC (config 64x64)...")
        tc = lambda q, k, v, sc, ca: tc_fwd(q, k, v, softmax_scale=sc, is_causal=ca, config_id=1)
        p50_tc, p90_tc = bench(tc, Q, K, V, scale, False)
    else:
        p50_tc, p90_tc = None, None
    
    # Build result
    res = {
        "canon_3": {
            "shape": {"B": B, "H": H, "S": S, "D": D},
            "causal": False,
            "sdpa": {
                "p50_ms": p50_sdpa,
                "p90_ms": p90_sdpa
            }
        }
    }
    
    if tc_available:
        res["canon_3"]["tc"] = {
            "p50_ms": p50_tc,
            "p90_ms": p90_tc,
            "speedup_vs_sdpa": p50_sdpa / p50_tc if p50_tc else None
        }
    
    # Write JSON
    out_file = out_dir / "tc_vs_sdpa_s512.json"
    out_file.write_text(json.dumps(res, indent=2))
    
    # Print summary
    print("\n" + "=" * 70)
    print("S=512 BENCHMARK RESULTS (canon_3: B=2, H=8, S=512, D=64)")
    print("=" * 70)
    print(f"SDPA:  p50={p50_sdpa:.3f}ms  p90={p90_sdpa:.3f}ms")
    
    if tc_available:
        print(f"TC:    p50={p50_tc:.3f}ms  p90={p90_tc:.3f}ms")
        speedup = p50_sdpa / p50_tc
        indicator = "✅ FASTER" if speedup > 1.0 else "❌ SLOWER"
        print(f"Speedup: {speedup:.3f}× {indicator}")
    else:
        print("TC:    Not available (module not compiled)")
    
    print(f"\n📊 Results saved to: {out_file}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

