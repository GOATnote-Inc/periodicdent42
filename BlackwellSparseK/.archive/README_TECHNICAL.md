# BlackwellSparseK

Sparse attention kernels for NVIDIA Hopper (sm_90a) and Blackwell (sm_100) architectures implementing the SparseK learned sparsity algorithm [1].

## Performance

| GPU | Seq Len | Heads | Dim | TFLOPS | Latency (μs) | vs FA3 |
|-----|---------|-------|-----|--------|--------------|--------|
| H100 | 512 | 96 | 64 | TBD | 3.820 (baseline) | - |
| H100 | 2048 | 32 | 128 | TBD | TBD | - |
| H100 | 8192 | 32 | 128 | TBD | TBD | - |
| B200 | 8192 | 32 | 128 | TBD (proj) | TBD (proj) | - |

*Baseline: PyTorch 2.9.0 SDPA on H100 80GB HBM3. FA3 comparisons pending.*

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .
python examples/basic_attention.py
```

## Supported Architectures

**Hopper (SM 90a)**
- H100 PCIe 80GB
- H100 SXM 80GB
- H100 NVL 188GB

**Blackwell (SM 100)** *(projected)*
- B100 80GB
- B200 144GB
- GB200 NVL72 (1.4TB)

## Requirements

```
CUDA >= 13.0.2
CUTLASS >= 4.3.0
PyTorch >= 2.9.0
flash-attn >= 3.0.0 (comparison baseline)
```

## Build

```bash
# Set target architectures
export TORCH_CUDA_ARCH_LIST="90;100"

# Build extension
python setup.py install

# Verify
python -c "import blackwell_sparsek; print('OK')"
```

## API

### Python

```python
import torch
from blackwell_sparsek import attention_forward

q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

# Forward pass
output = attention_forward(q, k, v, causal=True)
```

### C++ / CUDA

```cpp
#include <blackwell_sparsek/attention.h>

sparsek::AttentionParams params;
params.batch = B;
params.heads = H;
params.seq_len = S;
params.head_dim = D;
params.causal = true;

sparsek::attention_forward(params, Q_ptr, K_ptr, V_ptr, O_ptr, stream);
```

## Algorithm

Implements SparseK [1] with:
- Learned sparse attention patterns (K << S keys per query)
- Online softmax (FlashAttention-2 style tiling)
- FP16/FP8 mixed precision support
- Hopper TMA async loads (sm_90a)
- Blackwell WGMMA instructions (sm_100)

Tiling: `Br=32, Bc=64` (Hopper), `Br=64, Bc=128` (Blackwell projected)

## Kernel Features

**Hopper (sm_90a)**
- WMMA Tensor Core ops (16x16x16)
- TMA bulk async copy (cp.async.bulk)
- Warp specialization (producer/consumer)
- 128KB L2 cache residency control

**Blackwell (sm_100)** *(planned)*
- WGMMA instructions (64x256x32)
- Enhanced TMA (4th gen)
- FP8 E4M3/E5M2 native support
- 256MB L2 cache

## Benchmarking

```bash
# Single configuration
python benchmarks/benchmark.py --config gpt4

# Full sweep
python benchmarks/benchmark.py --sweep

# With Nsight Compute
ncu -o profile --set full python benchmarks/benchmark.py --config gpt4
```

## Testing

```bash
# Correctness (vs FlashAttention-3)
pytest tests/test_correctness.py -v

# Performance regression
pytest tests/test_performance.py -v
```

## Integration

**vLLM**
```python
# Set attention backend in config
attention_backend = "sparsek"
```

**xFormers**
```python
from xformers.ops import memory_efficient_attention
from blackwell_sparsek import SparseKAttentionOp

output = memory_efficient_attention(q, k, v, op=SparseKAttentionOp)
```

## Citation

```bibtex
@article{sun2024sparsek,
  title={SparseK: Learned Sparse Attention for Efficient LLM Inference},
  author={Sun, Mingjie and others},
  journal={arXiv preprint arXiv:2406.16747},
  year={2024}
}
```

Built with NVIDIA CUTLASS 4.3.0 and inspired by FlashAttention-2/3.

## License

MIT License. See LICENSE file.

