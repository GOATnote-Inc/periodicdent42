# FlashAttention-Science Kernel Digest

**Source of truth:** `cudadent42/kernels/attention/` (C++/CUDA) with Python bindings in `cudadent42/python/flashmoe_science/ops.py`.

## Design Highlights

- **Warp Specialization:** Separates load/store and math warps to hide memory latency on Hopper.
- **Asynchronous Pipelines:** Uses `cuda::pipeline` with `memcpy_async` to double-buffer tiles (supports 128 × 128 × 64 tile).
- **Mixed Precision:** FP8 inputs with BF16 accumulation and FP32 softmax for numerical stability.
- **Periodic-Pattern Tiling:** Aligns block scheduling with periodic table embeddings to maximize cache reuse.
- **Autotuning Hooks:** Kernel exposes `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, and staging parameters via template instantiation for easy tuning.

## Usage

```python
from flashmoe_science import flash_attention_science
output = flash_attention_science(Q, K, V, causal=True)
```

- Accepts FP16/BF16 tensors on CUDA.
- Optional `softmax_scale` overrides default `1/sqrt(head_dim)`.

## Key Files

| Path | Purpose |
|------|---------|
| `cudadent42/kernels/attention/include/flash_attention_science.h` | Host/device interface + launch params |
| `cudadent42/kernels/attention/src/flash_attention_science.cu` | Forward kernel implementation |
| `cudadent42/kernels/attention/src/flash_attention_backward.cu` | Backward kernel |
| `cudadent42/tests/test_attention_correctness.py` | Numerical validation |

## Benchmark Summary (H100, CUDA 12.3)

| Baseline | Latency (ms) | Speedup |
|----------|--------------|---------|
| PyTorch SDPA (cuDNN) | 3.28 ± 0.04 | 2.35× |
| flash-attn 2.3.3 | 1.62 ± 0.02 | 1.19× |
| FlashAttention-Science | **1.36 ± 0.01** | — |

Benchmarks capture 4 × 8 heads, 2048 sequence length, 64 head dimension, BF16 precision. Full raw samples: [`../benchmarks/results/flash_attention_h100.json`](../benchmarks/results/flash_attention_h100.json).

## Future Work

- Extend kernels to support 8k contexts with paged KV cache.
- Integrate automatic heuristics for tile selection based on SM count.
- Explore Blackwell-specific optimizations (Tensor Memory Accelerator).
