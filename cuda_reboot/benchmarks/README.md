# Benchmark Methodology

This directory provides auditable scripts and results comparing the FlashAttention-Science and fused MoE kernels against contemporary baselines.

## Baselines

| Kernel | Baselines |
|--------|-----------|
| FlashAttention-Science | 1. PyTorch 2.2 scaled dot product attention (cuDNN backend)  
| | 2. `flash-attn` 2.3.3 (Hopper-optimized FA2) |
| Fused MoE | 1. PyTorch reference MoE implemented with `torch.einsum` + `torch.topk`  
| | 2. DeepSpeed MoE (v0.12) fused experts |

## Scripts

- `run_attention_benchmarks.py` — Produces latency, throughput, and memory stats for attention kernels.
- `run_moe_benchmarks.py` — Measures tokens/sec and memory usage for MoE kernels.

Both scripts:

- Use `torch.cuda` events for precise GPU timing with warmup + measurement phases.
- Emit structured JSON for reproducibility and markdown tables for quick inspection.
- Support presets (`hopper-h100`, `ampere-a100`) that standardize tensor shapes.
- Accept `--baseline` filters to skip unavailable baselines.

## Results Files

- `results/flash_attention_h100.json` — Captured on 2025-10-11 using H100 SXM.
- `results/fused_moe_h100.json` — Same environment, includes DeepSpeed comparison.

New runs can be appended by passing `--output` to the scripts; filenames include timestamps unless a path is provided.
