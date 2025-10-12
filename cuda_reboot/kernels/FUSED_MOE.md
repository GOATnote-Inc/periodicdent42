# Fused Mixture-of-Experts Kernel Digest

**Source of truth:** `cudadent42/kernels/moe/` with Python wrappers in `cudadent42/python/flashmoe_science/ops.py:fused_moe`.

## Architecture Overview

1. **Routing Radix Sort**
   - Top-k selection performed via radix sort on routing weights.
   - Produces contiguous token blocks per expert to maximize memory coalescing.
2. **FP8 Expert GEMM**
   - Converts expert weights to FP8 (E4M3) with BF16 accumulation using Tensor Cores.
   - Supports fallback to BF16 on Ampere.
3. **Single-Kernel Fusion**
   - Dispatch, GEMM, and combine executed in one kernel launch to minimize synchronization overhead.
4. **Load Balancing**
   - Per-expert occupancy histogram ensures no expert exceeds 1.2× average load.
   - Optional temperature annealing for routing weights to reduce stragglers.

## Usage

```python
from flashmoe_science import fused_moe
output = fused_moe(tokens, expert_weights, routing_weights, top_k=8)
```

- `tokens`: `[batch, seq_len, hidden_dim]`
- `expert_weights`: `[num_experts, hidden_dim, expert_dim]`
- `routing_weights`: `[batch * seq_len, num_experts]`

## Key Files

| Path | Purpose |
|------|---------|
| `cudadent42/kernels/moe/include/fused_moe.h` | Host launch helpers |
| `cudadent42/kernels/moe/src/fused_moe_dispatch.cu` | Token routing + fusion kernel |
| `cudadent42/tests/test_warp_specialized.py` | End-to-end MoE tests |

## Benchmark Summary (H100, CUDA 12.3)

Configuration: 256 experts, top-8 routing, 4096 hidden dim, batch 16 × sequence 128.

| Baseline | Tokens/sec | Speedup |
|----------|------------|---------|
| PyTorch MoE (torch.compile) | 18.2k ± 0.3k | 3.8× |
| DeepSpeed MoE (v0.12) | 42.5k ± 0.5k | 1.6× |
| FlashMoE-Science Fused | **68.1k ± 0.6k** | — |

Raw timing samples: [`../benchmarks/results/fused_moe_h100.json`](../benchmarks/results/fused_moe_h100.json).

## Future Work

- Integrate with DTensor for hybrid tensor + expert parallelism.
- Auto-tune radix parameters for diverse batch sizes.
- Add BF16-without-FP8 path for legacy GPUs.
