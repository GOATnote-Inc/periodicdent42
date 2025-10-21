#!/usr/bin/env python3
"""
kbench.py — Hardened benchmarking harness for SDPA kernels
=========================================================
- Deterministic seeds
- Warmups + N iterations
- p50/p90 + 95% CI
- Wall-clock latency and throughput (approx. tokens/s and TFLOP/s)
- Records environment (GPU, driver, CUDA, PyTorch, git SHA)
- Compares against baselines:
  A) PyTorch SDPA (default backend)
  B) PyTorch SDPA (fastest native backend: flash -> mem_efficient -> math)
- Numerical checks vs baseline A
- Emits JSON to artifacts/bench/<run_id>.json
"""
import argparse, os, time, json, math, statistics, subprocess, sys, hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sdpa_ref(Q, K, V, scale):
    # PyTorch SDPA in FP16 (baseline A: default backend selection)
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale)

def sdpa_fastest(Q, K, V, scale):
    # Prefer FLASH; if not available, fallback to mem_efficient; else math
    try:
        # Flash only
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale)
    except Exception:
        pass
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale)
    except Exception:
        pass
    # Fallback: math
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale)

def run_timed(fn, *tensors, iters=1, warmup=0):
    # CUDA event timing
    times = []
    # Warmup
    for _ in range(warmup):
        out = fn(*tensors)
    torch.cuda.synchronize()
    # Timed
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*tensors)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        times.append(ms * 1000.0)  # μs
    return times, out

def stats(times_us):
    p50 = statistics.median(times_us)
    p90 = statistics.quantiles(times_us, n=10)[8] if len(times_us) >= 10 else p50
    mean = statistics.mean(times_us)
    stdev = statistics.stdev(times_us) if len(times_us) > 1 else 0.0
    # 95% CI for mean (normal approx)
    ci95 = 1.96 * (stdev / math.sqrt(max(1, len(times_us))))
    return dict(p50_us=p50, p90_us=p90, mean_us=mean, stdev_us=stdev, ci95_us=ci95)

def approx_flops(B,H,S,D):
    # Attention FLOPs (QK^T + softmax + PV). Approximate:
    # QK^T: B*H*S*S*D*2; PV: B*H*S*S*D*2 → ~4*B*H*S*S*D FLOPs
    return 4.0*B*H*S*S*D

def approx_tokens(B,H,S):
    # Tokens processed ~ B*S per head batch; report Q tokens/s aggregate
    return B*S

def load_candidate(backend: str):
    """
    Dynamically load candidate backend. Expected shape/order: (B,H,S,D) x 3 -> (B,H,S,D)
    The module must expose a `run(Q,K,V,scale)` function.
    """
    if backend == "baseline_a":
        return sdpa_ref
    if backend == "baseline_b":
        return sdpa_fastest

    # Search under kernels/<backend>/impl.py
    mod_path = Path("kernels")/backend/"impl.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Candidate backend '{backend}' not found at {mod_path}")
    import importlib.util, importlib.machinery
    spec = importlib.util.spec_from_file_location(f"kernels.{backend}.impl", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "run"):
        raise AttributeError(f"{mod_path} must define run(Q,K,V,scale)")
    return mod.run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", type=str, default="baseline_a",
                    help="baseline_a | baseline_b | candidate_triton_ws | candidate_triton_flashlike | candidate_cuda_stub | <custom>")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", type=str, default="mission",
                    help="small|mission|long or B,H,S,D as CSV")
    ap.add_argument("--out", type=str, default=None, help="JSON output path")
    ap.add_argument("--check", action="store_true", help="Run numerical check vs baseline A (within tolerances)")
    args = ap.parse_args()

    # Resolve shape
    if args.shape in ("small","mission","long"):
        table = {
            "small":   (1,8,128,64),
            "mission": (2,8,512,64),
            "long":    (2,8,2048,64),
        }
        B,H,S,D = table[args.shape]
    else:
        B,H,S,D = [int(x) for x in args.shape.split(",")]
    device = "cuda"
    dtype = torch.float16
    set_seed(args.seed)

    # Make tensors
    Q = torch.randn(B,H,S,D, dtype=dtype, device=device)
    K = torch.randn(B,H,S,D, dtype=dtype, device=device)
    V = torch.randn(B,H,S,D, dtype=dtype, device=device)
    scale = 1.0/math.sqrt(D)

    # Baseline A (for correctness and SDPA baseline)
    ref = sdpa_ref(Q,K,V,scale).detach()

    # Candidate
    run_fn = load_candidate(args.backend)
    times_us, cand = run_timed(run_fn, Q,K,V,scale, iters=args.iters, warmup=args.warmup)

    st = stats(times_us)

    # Throughput & TFLOPs/s (approx from mean latency per iter)
    total_tokens = approx_tokens(B,H,S) * args.iters
    avg_s = st["mean_us"]/1e6
    tput_tokens_per_s = (approx_tokens(B,H,S) / (st["p50_us"]/1e6))  # p50-based throughput
    tflops = (approx_flops(B,H,S,D) / 1e12) / (st["p50_us"]/1e6)

    # Baseline timings for speedup vs baseline A/B
    # We keep short runs for baselines to keep total runtime controlled
    torch.cuda.synchronize()
    baseA_t, _ = run_timed(sdpa_ref, Q,K,V,scale, iters=min( max(10, args.iters//5), 50 ), warmup=10)
    baseA_p50 = statistics.median(baseA_t)
    baseB_t, _ = run_timed(sdpa_fastest, Q,K,V,scale, iters=min( max(10, args.iters//5), 50 ), warmup=10)
    baseB_p50 = statistics.median(baseB_t)

    speedup_vs_A = baseA_p50 / st["p50_us"]
    speedup_vs_B = baseB_p50 / st["p50_us"]

    # Numerical checks
    max_err = float((cand.float()-ref.float()).abs().max().item())
    mean_err = float((cand.float()-ref.float()).abs().mean().item())
    # tolerance defaults
    atol = 6e-2
    rtol = 2e-2
    correctness = (max_err <= atol) and (mean_err <= rtol)

    # Environment
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "sm": list(torch.cuda.get_device_capability(0)),
        "driver": None,
        "git_sha": os.environ.get("GIT_SHA", None),
        "backend": args.backend,
    }

    run_id = f"{args.backend}_{args.shape}_{args.iters}i_{args.warmup}w"
    out_path = args.out or f"artifacts/bench/{run_id}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "ts": time.time(),
        "backend": args.backend,
        "shape": dict(B=B,H=H,S=S,D=D,name=args.shape),
        "iters": args.iters,
        "warmup": args.warmup,
        "latency": st,
        "baseline_p50_us": {"A": baseA_p50, "B": baseB_p50},
        "speedup": {"vs_A": speedup_vs_A, "vs_B": speedup_vs_B},
        "throughput_tokens_per_s_p50": tput_tokens_per_s,
        "approx_tflops_p50": tflops,
        "numeric": {"max_abs_err": max_err, "mean_abs_err": mean_err, "atol": 0.06, "rtol": 0.02, "pass": correctness},
        "env": env,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({
        "p50_us": st["p50_us"],
        "speedup_vs_A": speedup_vs_A,
        "speedup_vs_B": speedup_vs_B,
        "numeric_pass": correctness,
        "out": out_path,
    }, indent=2))

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Please run on a CUDA-enabled machine.", file=sys.stderr)
        sys.exit(1)
    main()
