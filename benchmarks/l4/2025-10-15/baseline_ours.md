# Baseline Benchmark Report: ours

**Date**: 2025-10-15 02:16:43
**Warmups**: 10
**Iterations**: 50

| Name | B | H | S | D | Dtype | Causal | p50 (ms) | p90 (ms) | TFLOP/s |
|------|---|---|---|---|-------|--------|----------|----------|---------|
| v3_small | 1 | 8 | 512 | 64 | float16 | False | 7.433 | 7.554 | 0.07 |
| v3_small_causal | 1 | 8 | 512 | 64 | float16 | True | 5.966 | 6.097 | 0.09 |
| v3_medium | 4 | 16 | 512 | 64 | float16 | False | 56.325 | 56.820 | 0.08 |
| v3_medium_causal | 4 | 16 | 512 | 64 | float16 | True | 43.884 | 45.072 | 0.10 |
| v3_large | 8 | 16 | 512 | 64 | float16 | False | 113.798 | 114.340 | 0.08 |
| v3_large_causal | 8 | 16 | 512 | 64 | float16 | True | 89.296 | 91.643 | 0.10 |
