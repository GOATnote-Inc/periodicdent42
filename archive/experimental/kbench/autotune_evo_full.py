#!/usr/bin/env python3
"""
Stage-5 EvoEngineer-Full Autotune
===================================
Elite preservation (K=3) over configuration grid.
Two-layer traverse: macro variants (WS, Persist, tiles) + micro optimizations.

Aligned with EvoEngineer methodology (Table 3, Sec. 4.1-4.3)
"""

import itertools
import json
import subprocess
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ====================
# Configuration Grid
# ====================

# Macro knobs (Layer 1: algorithmic choices)
MACRO_KNOBS = {
    "USE_WARP_SPECIALIZATION": [0, 1],
    "USE_PERSISTENT_CTA": [0, 1],
    "TILE_M": [32],  # Keep constant initially
    "TILE_N": [32],  # Keep constant initially
}

# Micro knobs (Layer 2: fine-tuning per macro)
MICRO_KNOBS = {
    "NUM_PRODUCER_WARPS": [1, 2],
    "USE_FAST_EXP": [0, 1],
}

# Generate full configuration grid
CFG = []
for ws, persist, tm, tn in itertools.product(
    MACRO_KNOBS["USE_WARP_SPECIALIZATION"],
    MACRO_KNOBS["USE_PERSISTENT_CTA"],
    MACRO_KNOBS["TILE_M"],
    MACRO_KNOBS["TILE_N"],
):
    for prod, fast_exp in itertools.product(
        MICRO_KNOBS["NUM_PRODUCER_WARPS"],
        MICRO_KNOBS["USE_FAST_EXP"],
    ):
        CFG.append((ws, persist, tm, tn, prod, fast_exp))

print(f"Configuration grid size: {len(CFG)} configs")

# ====================
# Hyperparameters
# ====================

ELITES = 3          # Elite population size (K)
ITERS = 60          # Performance benchmark iterations
WARMUP = 20         # Warmup iterations
SHAPES = "mission"  # Focus on mission shape for autotune


# ====================
# Helper Functions
# ====================

def config_flags(c):
    """Convert config tuple to NVCC flags."""
    WS, PERSIST, TM, TN, PROD, FAST = c
    return (
        f"-DUSE_CP_ASYNC=1 "
        f"-DUSE_WMMA_PV=1 "
        f"-DUSE_WARP_SPECIALIZATION={WS} "
        f"-DNUM_PRODUCER_WARPS={PROD} "
        f"-DUSE_PERSISTENT_CTA={PERSIST} "
        f"-DTILE_M={TM} "
        f"-DTILE_N={TN} "
        f"-DUSE_FAST_EXP={FAST}"
    )


def build(flags):
    """
    Build kernel with specified flags.
    Returns True if successful, False otherwise.
    """
    env = os.environ.copy()
    env["NVCCFLAGS"] = f"-O3 -arch=sm_89 --use_fast_math {flags}"
    
    result = subprocess.run(
        ["python", "-m", "tasks.fp8_sdpa_stage_c_wmma.build"],
        env=env,
        capture_output=True,
        text=True,
    )
    
    # Check for PTXAS gate violations
    if result.returncode != 0:
        return False
    
    # Parse PTXAS output for resource usage
    for line in result.stdout.split("\n"):
        if "ptxas info" in line and "Used" in line:
            # Example: "ptxas info    : Used 96 registers, 37120 bytes smem"
            if "spill" in line.lower():
                print(f"  ⚠️  Spills detected: {line.strip()}")
                return False  # Reject configs with spills
            if "registers" in line:
                regs = int(line.split("Used")[1].split("registers")[0].strip())
                if regs > 120:
                    print(f"  ⚠️  Too many registers: {regs}")
                    return False
    
    return True


def run_bench():
    """
    Run benchmark script and parse results.
    Returns (ok, data) tuple.
    """
    out = "kbench/results_tune.json"
    result = subprocess.run(
        [
            "python",
            "scripts/bench_sdpa.py",
            "--iters",
            str(ITERS),
            "--warmup",
            str(WARMUP),
            "--shapes",
            SHAPES,
            "--out",
            out,
        ],
        capture_output=True,
        text=True,
    )
    
    ok = result.returncode == 0
    data = []
    
    if ok and os.path.exists(out):
        try:
            data = json.load(open(out))
        except:
            ok = False
    
    return ok, data


def score(report):
    """
    Score a configuration based on performance and correctness.
    Returns p50 latency (lower is better) or infinity if correctness fails.
    """
    if not report:
        return float("inf")
    
    # Extract mission shape results
    mission = [x for x in report if x["shape"] == "mission"]
    if not mission:
        return float("inf")
    
    mission = mission[0]
    
    # Gate: Correctness
    if not mission.get("correctness_pass", False):
        return float("inf")  # Reject incorrect configs
    
    # Gate: Performance (primary metric)
    return mission["p50_us"]


# ====================
# Main Search Loop
# ====================

def main():
    print("=" * 80)
    print("EvoEngineer-Full Autotune (Elite K={})".format(ELITES))
    print("=" * 80)
    print(f"Search space: {len(CFG)} configurations")
    print(f"Shapes: {SHAPES}")
    print(f"Iterations: {ITERS} (warmup: {WARMUP})")
    print("=" * 80)
    print()
    
    elites = []
    
    for i, config in enumerate(CFG):
        print(f"[{i+1}/{len(CFG)}] Config: {config}")
        flags = config_flags(config)
        print(f"  Flags: {flags}")
        
        # Gate 1: Compile
        print("  Gate 1: Compile...", end=" ", flush=True)
        if not build(flags):
            print("FAIL (PTXAS violation)")
            continue
        print("PASS")
        
        # Gate 2: Correctness + Performance
        print("  Gate 2: Benchmark...", end=" ", flush=True)
        ok, report = run_bench()
        if not ok:
            print("FAIL (benchmark error)")
            continue
        
        cfg_score = score(report)
        if cfg_score == float("inf"):
            print("FAIL (correctness gate)")
            continue
        
        print(f"PASS (p50={cfg_score:.2f} μs)")
        
        # Update elite population
        elites.append({
            "config": config,
            "flags": flags,
            "report": report,
            "score": cfg_score,
        })
        
        # Retain top-K elites
        elites = sorted(elites, key=lambda x: x["score"])[:ELITES]
        
        # Save elite population
        with open("kbench/elite.json", "w") as f:
            json.dump(elites, f, indent=2)
        
        # Print current elite leaderboard
        print(f"  Elite Top-{ELITES}:")
        for j, e in enumerate(elites):
            print(f"    {j+1}. {e['score']:.2f} μs — {e['config']}")
        print()
    
    # ====================
    # Final Results
    # ====================
    print("=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)
    
    if not elites:
        print("❌ No valid configurations found!")
        sys.exit(1)
    
    print(f"Final Elite Top-{ELITES}:")
    for i, e in enumerate(elites):
        print(f"  {i+1}. p50={e['score']:.2f} μs — {e['config']}")
    
    print()
    print("Winner:")
    winner = elites[0]
    print(f"  Config: {winner['config']}")
    print(f"  Score:  {winner['score']:.2f} μs")
    print(f"  Flags:  {winner['flags']}")
    print()
    print("📁 Elite population saved to: kbench/elite.json")
    print("✅ Autotune complete!")


if __name__ == "__main__":
    main()

