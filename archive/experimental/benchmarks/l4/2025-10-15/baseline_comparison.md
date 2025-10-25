# Baseline Benchmark Comparison

**Date**: 2025-10-15 02:16:43
**GPU**: NVIDIA L4
**CUDA**: 12.1

## Speedup Analysis (Ours vs SDPA)

| Shape | SDPA p50 (ms) | Ours p50 (ms) | Speedup | SDPA p90 (ms) | Ours p90 (ms) | Status |
|-------|---------------|---------------|---------|---------------|---------------|---------|
| v3_large | 0.136 | 113.798 | 0.001× | 0.147 | 114.340 | 🐢 |
| v3_large_causal | 0.127 | 89.296 | 0.001× | 0.131 | 91.643 | 🐢 |
| v3_medium | 0.088 | 56.325 | 0.002× | 0.093 | 56.820 | 🐢 |
| v3_medium_causal | 0.084 | 43.884 | 0.002× | 0.088 | 45.072 | 🐢 |
| v3_small | 0.045 | 7.433 | 0.006× | 0.058 | 7.554 | 🐢 |
| v3_small_causal | 0.045 | 5.966 | 0.008× | 0.054 | 6.097 | 🐢 |

## Legend

- 🚀: Speedup ≥ 1.10× (10%+ faster)
- ✓: Speedup ≥ 1.00× (faster or equal)
- 🐢: Speedup < 1.00× (slower)
