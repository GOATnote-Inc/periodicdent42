#!/usr/bin/env python3
"""
Robust-kbench runner for FP8 SDPA Stage-C WMMA kernel.

Usage:
    python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small,mission --seeds 0,1,2
    python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 5 --profile
"""

import argparse
import json
import datetime
import sys
from pathlib import Path
import torch
import math

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tasks.fp8_sdpa_stage_c_wmma.build import build_extension, capture_build_metadata
from tasks.fp8_sdpa_stage_c_wmma.func_forward import (
    forward_ref,
    forward_kernel,
    quantize_sim_fp8_per_head,
    validate_correctness,
)

def load_config():
    """Load configuration from config_forward.json."""
    config_path = Path(__file__).parent / "config_forward.json"
    with open(config_path) as f:
        return json.load(f)

def setup_determinism(seed: int):
    """Set up deterministic execution."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_inputs(B, H, S, D, seed, device="cuda", dtype=torch.float16):
    """Generate random FP16 inputs for given shape and seed."""
    setup_determinism(seed)
    Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    K = torch.randn(B, H, S, D, device=device, dtype=dtype)
    V = torch.randn(B, H, S, D, device=device, dtype=dtype)
    return Q, K, V

def run_correctness_check(ext, shape_config, seed, config):
    """Run correctness check for one (shape, seed) pair."""
    B = shape_config["B"]
    H = shape_config["H"]
    S = shape_config["S"]
    D = shape_config["D"]
    name = shape_config["name"]
    
    # Generate inputs
    Q, K, V = generate_inputs(B, H, S, D, seed)
    scale = 1.0 / math.sqrt(D)
    
    # Reference
    ref = forward_ref(Q, K, V, scale)
    
    # Quantize
    Q_q, Q_s = quantize_sim_fp8_per_head(Q)
    K_q, K_s = quantize_sim_fp8_per_head(K)
    V_q, V_s = quantize_sim_fp8_per_head(V)
    
    # Kernel
    out = forward_kernel(Q_q, K_q, V_q, Q_s, K_s, V_s, scale, ext)
    
    # Validate
    tol = config["tolerance"]
    metrics = validate_correctness(
        ref, out,
        atol=tol["atol"],
        rtol=tol["rtol"],
        pct_bad_max=tol["pct_bad_max"],
    )
    
    return {
        "shape": name,
        "B": B, "H": H, "S": S, "D": D,
        "seed": seed,
        **metrics,
    }

def run_performance_baseline(ext, shape_config, seed, iters, config):
    """Run performance baseline for one (shape, seed) pair."""
    B = shape_config["B"]
    H = shape_config["H"]
    S = shape_config["S"]
    D = shape_config["D"]
    name = shape_config["name"]
    
    timing_cfg = config.get("timing", {})
    warmup_iters = timing_cfg.get("warmup_iters", 100)
    timed_iters = iters if iters > 0 else timing_cfg.get("timed_iters", 500)
    
    # Generate inputs
    Q, K, V = generate_inputs(B, H, S, D, seed)
    scale = 1.0 / math.sqrt(D)
    
    # Quantize
    Q_q, Q_s = quantize_sim_fp8_per_head(Q)
    K_q, K_s = quantize_sim_fp8_per_head(K)
    V_q, V_s = quantize_sim_fp8_per_head(V)
    
    # Warmup
    for _ in range(warmup_iters):
        _ = forward_kernel(Q_q, K_q, V_q, Q_s, K_s, V_s, scale, ext)
    torch.cuda.synchronize()
    
    # Timed runs with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(timed_iters):
        start_event.record()
        _ = forward_kernel(Q_q, K_q, V_q, Q_s, K_s, V_s, scale, ext)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event) * 1000)  # Convert ms to μs
    
    times = torch.tensor(times)
    p50 = torch.median(times).item()
    p90 = torch.quantile(times, 0.9).item()
    mean = times.mean().item()
    std = times.std().item()
    
    return {
        "shape": name,
        "B": B, "H": H, "S": S, "D": D,
        "seed": seed,
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "p50_us": p50,
        "p90_us": p90,
        "mean_us": mean,
        "std_us": std,
        "iters_per_sec": 1e6 / mean,  # Convert μs to iters/s
    }

def main():
    parser = argparse.ArgumentParser(description="FP8 SDPA Stage-C WMMA robust-kbench runner")
    parser.add_argument("--shapes", type=str, default="small,mission", help="Comma-separated shape names")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--iters", type=int, default=0, help="Performance iterations (0 = skip perf)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--no-build", action="store_true", help="Skip build (use cached)")
    parser.add_argument("--profile", action="store_true", help="Run with profiling markers")
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Parse shapes and seeds
    shape_names = [s.strip() for s in args.shapes.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Filter shapes
    shapes_to_test = [s for s in config["shapes"] if s["name"] in shape_names]
    if not shapes_to_test:
        print(f"❌ No matching shapes for: {shape_names}")
        return 1
    
    print(f"\n{'='*80}")
    print("FP8 SDPA Stage-C WMMA — Robust-kbench Runner")
    print(f"{'='*80}")
    print(f"  Shapes: {', '.join(shape_names)}")
    print(f"  Seeds:  {', '.join(map(str, seeds))}")
    print(f"  Device: {args.device}")
    print(f"  Perf:   {'Yes' if args.iters > 0 else 'Correctness only'}")
    print(f"{'='*80}\n")
    
    # Build extension
    if not args.no_build:
        ext = build_extension(verbose=False)
    else:
        print("⚠️  Skipping build (--no-build), using cached extension")
        from torch.utils.cpp_extension import load
        ext = load(name="sdpa_fp8_stage_c_wmma", sources=[], verbose=False)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("results") / "fp8_wmma_baseline" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Capture build metadata
    build_meta = capture_build_metadata(output_dir)
    
    # Run correctness checks
    print(f"\n{'='*80}")
    print("CORRECTNESS VALIDATION")
    print(f"{'='*80}\n")
    
    correctness_results = []
    all_pass = True
    
    for shape in shapes_to_test:
        for seed in seeds:
            result = run_correctness_check(ext, shape, seed, config)
            correctness_results.append(result)
            
            status = "✅ PASS" if result["pass"] else "❌ FAIL"
            print(f"[{shape['name']:8s}] seed={seed}: "
                  f"max_err={result['max_abs_err']:.4f}, "
                  f"mean_err={result['mean_abs_err']:.4f}, "
                  f"%bad={result['pct_bad']:.1f}% "
                  f"{status}")
            
            if not result["pass"]:
                all_pass = False
                gates = result["gates"]
                print(f"  Gates: max_abs_err={gates['max_abs_err_pass']}, "
                      f"mean_abs_err={gates['mean_abs_err_pass']}, "
                      f"pct_bad={gates['pct_bad_pass']}")
    
    # Save correctness results
    with open(output_dir / "correctness_summary.json", "w") as f:
        json.dump({
            "results": correctness_results,
            "all_pass": all_pass,
            "config": config["tolerance"],
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    if all_pass:
        print("✅ ALL CORRECTNESS CHECKS PASSED!")
    else:
        print("❌ SOME CHECKS FAILED - see details above")
    print(f"{'='*80}\n")
    
    # Run performance baseline if requested
    if args.iters > 0:
        print(f"\n{'='*80}")
        print(f"PERFORMANCE BASELINE (iters={args.iters})")
        print(f"{'='*80}\n")
        
        perf_results = []
        for shape in shapes_to_test:
            for seed in seeds:
                result = run_performance_baseline(ext, shape, seed, args.iters, config)
                perf_results.append(result)
                
                print(f"[{shape['name']:8s}] seed={seed}: "
                      f"p50={result['p50_us']:.2f}μs, "
                      f"p90={result['p90_us']:.2f}μs, "
                      f"std={result['std_us']:.2f}μs")
        
        # Save performance results
        with open(output_dir / "perf_baseline.json", "w") as f:
            json.dump({
                "results": perf_results,
                "config": config["timing"],
            }, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✅ Performance baseline saved")
        print(f"{'='*80}\n")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Output dir: {output_dir}")
    print(f"  Files:")
    print(f"    - build_meta.json")
    print(f"    - correctness_summary.json")
    if args.iters > 0:
        print(f"    - perf_baseline.json")
    print(f"{'='*80}\n")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())

