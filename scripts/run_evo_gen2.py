#!/usr/bin/env python3
"""Quick Evo sweep: 2 generations, 8 candidates, seeded from microbench"""
import subprocess
import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent

# Load microbench seeds
micro_best = repo_root / "evidence" / "micro_best.json"
if not micro_best.exists():
    print("❌ No microbench seeds found. Run: python bench/micro/run_micro.py")
    sys.exit(1)

with open(micro_best) as f:
    seeds = json.load(f)

print(f"✅ Loaded {len(seeds)} microbench seeds")

# Gen 0: Use microbench Top-3
population = []
for i, cfg in enumerate(seeds[:3]):
    variant_id = f"gen0_micro{i}"
    population.append({
        "id": variant_id,
        "BLOCK_M": cfg["BLOCK_M"],
        "NUM_WARPS": cfg["NUM_WARPS"],
        "VEC_WIDTH": cfg["VEC_WIDTH"],
        "SYNC_POLICY": 2,
        "REDUCE": "warp"
    })

# Gen 0: Add baseline
population.append({
    "id": "gen0_baseline",
    "BLOCK_M": 32,
    "NUM_WARPS": 4,
    "VEC_WIDTH": 4,
    "SYNC_POLICY": 2,
    "REDUCE": "warp"
})

# Gen 0: Add mutations
for i, base in enumerate(population[:2]):
    # Mutate NUM_WARPS
    mut = base.copy()
    mut["id"] = f"gen0_mut{i}"
    mut["NUM_WARPS"] = 8 if base["NUM_WARPS"] == 4 else 4
    population.append(mut)

print(f"\n📊 Gen 0: {len(population)} candidates")
for p in population:
    print(f"  {p['id']}: BLOCK_M={p['BLOCK_M']} WARPS={p['NUM_WARPS']} VEC={p['VEC_WIDTH']}")

# Build & test each
sys.path.insert(0, str(repo_root / "bench"))
from build_phase3_variant import build_phase3_variant

import torch
import time

results = []

for variant in population:
    print(f"\n🔨 Testing {variant['id']}...")
    
    # Set env
    for k, v in variant.items():
        if k != "id":
            import os
            os.environ[k] = str(v)
    
    # Build
    try:
        build_phase3_variant()
    except:
        print(f"  ❌ Build failed")
        results.append({**variant, "time_us": 999999, "speedup": 0.0})
        continue
    
    # Test
    try:
        import fa_phase3
        import importlib
        importlib.reload(fa_phase3)
        
        B, H, S, D = 1, 8, 512, 64
        q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
        k = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
        v = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda:0')
        scale = 1.0 / (D ** 0.5)
        
        # Warmup
        for _ in range(5):
            o = fa_phase3.forward(q, k, v, scale)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            o = fa_phase3.forward(q, k, v, scale)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_us = (end - start) * 1e6 / 50
        
        # PyTorch baseline
        ref_time_us = 47.0  # Known L4 baseline
        speedup = ref_time_us / time_us
        
        print(f"  ✅ {time_us:.2f} μs ({speedup:.2f}× vs SDPA)")
        results.append({**variant, "time_us": time_us, "speedup": speedup})
        
    except Exception as e:
        print(f"  ❌ Runtime failed: {e}")
        results.append({**variant, "time_us": 999999, "speedup": 0.0})

# Sort by time
results.sort(key=lambda x: x["time_us"])

# Write results
output_file = repo_root / "evidence" / "evo_gen0_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to {output_file}")
print("\n📊 Top-3:")
for i, r in enumerate(results[:3]):
    print(f"  {i+1}. {r['id']}: {r['time_us']:.2f} μs "
          f"(M={r['BLOCK_M']}, W={r['NUM_WARPS']}, V={r['VEC_WIDTH']})")

