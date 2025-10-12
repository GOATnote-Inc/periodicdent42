# CUDA Reboot Initiative

A clean-slate consolidation of the most effective CUDA kernels produced in the periodicdent42 program. This branch resets the layout near the repository root so the GPU work can evolve independently while retaining the proven FlashAttention-Science and fused Mixture-of-Experts (MoE) kernels.

## Goals

- Provide a minimal, auditable entry point for CUDA kernel research.
- Curate the production-ready FlashAttention-Science and fused MoE kernels with focused documentation.
- Ship reproducible benchmarks against state-of-the-art (SOTA) industry baselines to substantiate performance claims.

## What's Inside

```
cuda_reboot/
├── README.md                     # This overview
├── REPRODUCIBILITY.md            # Exact environment + runbook for rebuilding the kernels
├── kernels/
│   ├── FLASH_ATTENTION.md        # Design + references to canonical implementation
│   └── FUSED_MOE.md              # MoE kernel architecture + tuning guide
└── benchmarks/
    ├── README.md                 # Benchmark methodology and baseline definitions
    ├── run_attention_benchmarks.py
    ├── run_moe_benchmarks.py
    └── results/
        ├── flash_attention_h100.json
        └── fused_moe_h100.json
```

The actual CUDA source continues to live under `cudadent42/`. These reboot docs and harnesses point directly to the canonical implementations while making it simple to rerun tests from a pristine clone.

## Quick Start

1. Set up the environment following [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md).
2. Build the CUDA extensions in `cudadent42/` (exact commands provided in the reproducibility guide).
3. Run `python cuda_reboot/benchmarks/run_attention_benchmarks.py --preset hopper-h100` to generate fresh attention benchmark numbers.
4. Optionally run `python cuda_reboot/benchmarks/run_moe_benchmarks.py --preset hopper-h100` for the fused MoE benchmark suite.
5. Inspect JSON outputs under `cuda_reboot/benchmarks/results/` or pass `--output` to create timestamped reports.

## Relationship to Prior Work

- The FlashAttention-Science kernel is implemented in `cudadent42/python/flashmoe_science/ops.py` with CUDA sources under `cudadent42/kernels/attention/`.
- The fused MoE kernel lives in the same package and reuses the dispatcher defined in `cudadent42/kernels/moe/`.
- Previous documentation scattered across multiple session summaries has been condensed into the two design memos found in `cuda_reboot/kernels/`.

This branch/folder can now serve as the canonical launchpad for further CUDA R&D.
