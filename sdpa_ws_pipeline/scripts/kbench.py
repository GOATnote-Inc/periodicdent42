#!/usr/bin/env python3
import os, sys, json, math, time, random, argparse, importlib, statistics, subprocess, pathlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
ART_DIR = ROOT / "artifacts" / "bench"; ART_DIR.mkdir(parents=True, exist_ok=True)

# ---- Determinism (best-effort for perf harness) ----
def set_seeds(seed:int=12345):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # may be ignored by some kernels, but we log it

def percentile(a, p):
    return float(np.percentile(np.asarray(a, dtype=np.float64), p))

def bootstrap_ci(samples, stat_fn, iters=1000, alpha=0.05):
    rng = np.random.default_rng(2024)
    n = len(samples)
    boots = []
    arr = np.asarray(samples, dtype=np.float64)
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        boots.append(stat_fn(arr[idx]))
    lo = np.percentile(boots, 100 * (alpha/2))
    hi = np.percentile(boots, 100 * (1 - alpha/2))
    return float(lo), float(hi)

def time_callable(fn, warmup=5, iters=100):
    # CUDA timing with events (us)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record(); end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # to microseconds
    return times

def sdpa_baseline(Q, K, V, scale, backend: str):
    assert backend in ("math","flash","default")
    if backend == "math":
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    elif backend == "flash":
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    else:
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)  # default
    with ctx:
        # PyTorch SDPA expects [B*H, S, D]
        q = Q.reshape(-1, Q.shape[2], Q.shape[3])
        k = K.reshape(-1, K.shape[2], K.shape[3])
        v = V.reshape(-1, V.shape[2], V.shape[3])
        O = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale)
        return O.reshape_as(Q)

def load_candidate(name:str):
    # maps to kernels/<name>/impl.py with run(), get_config()
    sys.path.insert(0, str(ROOT))
    mod = importlib.import_module(f"kernels.{name}.impl")
    return mod

def gen_inputs(shape, dtype=torch.float16, device="cuda"):
    B,H,S,D = shape
    Q = torch.randn(B,H,S,D, device=device, dtype=dtype)
    K = torch.randn(B,H,S,D, device=device, dtype=dtype)
    V = torch.randn(B,H,S,D, device=device, dtype=dtype)
    return Q,K,V

def run_variant(name, shape, warmup, iters, dtype=torch.float16):
    mod = load_candidate(name)
    cfg = mod.get_config()
    Q,K,V = gen_inputs(shape, dtype=dtype)
    scale = 1.0 / math.sqrt(shape[3])

    # Reference output from PyTorch SDPA (baseline B = fastest native)
    ref = sdpa_baseline(Q, K, V, scale, backend="flash")

    def call():
        out = mod.run(Q, K, V, scale)
        # ensure kernel finished; some extensions return async tensors
        if out.is_cuda: torch.cuda.synchronize()

    times = time_callable(call, warmup, iters)

    # Single correctness check (mean/max abs error) vs baseline B
    out = mod.run(Q, K, V, scale); torch.cuda.synchronize()
    mean_abs = float((out - ref).abs().mean().item())
    max_abs  = float((out - ref).abs().max().item())

    result = {
        "variant": cfg["name"],
        "description": cfg.get("description",""),
        "shape": {"B":shape[0], "H":shape[1], "S":shape[2], "D":shape[3]},
        "dtype": str(dtype).replace("torch.",""),
        "warmup": warmup, "iters": iters,
        "p50_us": percentile(times, 50), "p90_us": percentile(times, 90),
        "median_ci95_us": bootstrap_ci(times, lambda x: float(np.percentile(x,50))),
        "mean_us": float(np.mean(times)),
        "mean_ci95_us": bootstrap_ci(times, np.mean),
        "samples": len(times),
        "mean_abs_err": mean_abs, "max_abs_err": max_abs,
        "atol": 6e-2, "rtol": 1e-3,
        "correctness_pass": (max_abs <= 6e-2),
    }
    return result

def run_baselines(shape, warmup, iters, dtype=torch.float16):
    Q,K,V = gen_inputs(shape, dtype=dtype)
    scale = 1.0 / math.sqrt(shape[3])

    def mk_runner(backend):
        return lambda: sdpa_baseline(Q,K,V,scale,backend=backend)

    results = {}
    for tag,backend in [("baseline_a","math"),("baseline_b","flash")]:
        times = time_callable(mk_runner(backend), warmup, iters)
        results[tag] = {
            "variant": f"PyTorch SDPA ({backend})",
            "shape": {"B":shape[0], "H":shape[1], "S":shape[2], "D":shape[3]},
            "dtype": str(dtype).replace("torch.",""),
            "warmup": warmup, "iters": iters,
            "p50_us": percentile(times, 50), "p90_us": percentile(times, 90),
            "median_ci95_us": bootstrap_ci(times, lambda x: float(np.percentile(x,50))),
            "mean_us": float(np.mean(times)),
            "mean_ci95_us": bootstrap_ci(times, np.mean),
            "samples": len(times)
        }
        (ART_DIR / f"{tag}.json").write_text(json.dumps(results[tag], indent=2))
    return results

def add_speedups(cand, base_a, base_b):
    cand["speedup_vs_baseline_a"] = base_a["p50_us"] / cand["p50_us"]
    cand["speedup_vs_baseline_b"] = base_b["p50_us"] / cand["p50_us"]
    return cand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="2,8,512,64")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--variants", nargs="+", default=[
        "candidate_triton_flashlike",  # Stage-2 control
        "candidate_cuda_stub",         # WS P=1
        "candidate_triton_ws"          # WS P=2
    ])
    args = ap.parse_args()

    set_seeds(args.seed)
    shape = tuple(map(int, args.shape.split(",")))

    # Baselines
    bases = run_baselines(shape, args.warmup, args.iters)
    base_a, base_b = bases["baseline_a"], bases["baseline_b"]

    # Candidates
    for name in args.variants:
        res = run_variant(name, shape, args.warmup, args.iters)
        res = add_speedups(res, base_a, base_b)
        # map names to the required filenames
        alias = {
            "candidate_triton_flashlike":"candidate_stage2",
            "candidate_cuda_stub":"candidate_ws_p1",
            "candidate_triton_ws":"candidate_ws_p2"
        }
        out = ART_DIR / f"{alias.get(name,name)}.json"
        out.write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required"
    main()
