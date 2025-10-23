#!/usr/bin/env python3
"""Build and benchmark the FlashCore WMMA kernels."""

import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent.parent


def build_extension(verbose: bool = False):
    sources = [
        ROOT / "flashcore" / "flashcore_bindings.cpp",
        ROOT / "flashcore" / "flashcore_unified.cu",
        ROOT / "flashcore" / "flashcore_fused.cu",
        ROOT / "flashcore" / "flashcore_fused_phase2.cu",
        ROOT / "flashcore" / "flashcore_v8_dynamic_smem.cu",
        ROOT / "flashcore" / "flashcore_v9_warp_spec.cu",
        ROOT / "flashcore" / "flashcore_v10_3stage.cu",
        ROOT / "flashcore" / "flashcore_v9_1_verified.cu",
    ]
    extra_include_paths = [str(ROOT / "flashcore")]
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-use_fast_math",
        "--generate-code=arch=compute_89,code=sm_89",
        "-maxrregcount=96",  # Allow more registers
        "-Xptxas=-dlcm=ca",  # Cache-all mode for L2
    ]

    return load(
        name="flashcore_wmma",
        sources=[str(s) for s in sources],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        verbose=verbose,
    )


def _benchmark_kernel(fn, *tensors, iters: int = 200, warmup: int = 20):
    for _ in range(warmup):
        fn(*tensors)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    for _ in range(iters):
        start.record()
        fn(*tensors)
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)
    return (total_ms / iters) * 1000.0  # Convert to microseconds


def benchmark_qkt(module, B: int, H: int, S: int, D: int, iters: int):
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    scale = 1.0 / (D ** 0.5)
    avg_us = _benchmark_kernel(lambda: module.qkt(q, k, scale), iters=iters)
    return avg_us


def benchmark_pv(module, B: int, H: int, S: int, D: int, iters: int):
    p = torch.randn(B, H, S, S, device="cuda", dtype=torch.float16)
    p = torch.softmax(p.float(), dim=-1).to(torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    avg_us = _benchmark_kernel(lambda: module.pv(p, v), iters=iters)
    return avg_us


def main():
    parser = argparse.ArgumentParser(description="FlashCore WMMA benchmark")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--B", type=int, default=1, help="Batch size")
    parser.add_argument("--H", type=int, default=16, help="Number of heads")
    parser.add_argument("--S", type=int, default=128, help="Sequence length")
    parser.add_argument("--D", type=int, default=64, help="Head dimension")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose build logs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark FlashCore kernels")

    torch.cuda.init()
    module = build_extension(verbose=args.verbose)

    print("Running FlashCore WMMA benchmarks...")
    print(f"Configuration: B={args.B}, H={args.H}, S={args.S}, D={args.D}, iters={args.iters}")

    qkt_us = benchmark_qkt(module, args.B, args.H, args.S, args.D, args.iters)
    print(f"[QK^T] Average latency: {qkt_us:.2f} µs")

    pv_us = benchmark_pv(module, args.B, args.H, args.S, args.D, args.iters)
    print(f"[P·V] Average latency: {pv_us:.2f} µs")


if __name__ == "__main__":
    main()

