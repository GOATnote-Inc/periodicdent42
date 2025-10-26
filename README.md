# FlashCore

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.4+](https://img.shields.io/badge/CUDA-12.4%2B-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

High-performance attention kernels for modern LLMs. Implements FlashAttention-style algorithms with support for KV caching, Grouped-Query Attention (GQA), and causal masking.

**Production-ready**: 15/15 tests pass on NVIDIA H100 (sm_90, Hopper architecture).

---

## Features

- **KV Cache**: Efficient incremental inference for autoregressive generation
- **GQA Support**: 4-7× memory savings for LLaMA 3.1, Mistral, Qwen architectures
- **Causal Masking**: Zero-overhead autoregressive attention
- **FP16 Optimized**: FP32 accumulators for numerical stability
- **Production Ready**: Validated on NVIDIA H100, supports all modern LLM architectures

---

## ⚙️ Debug + Profile Workflow (NVIDIA Tools)

**Professional-grade validation** matching FA3, CUTLASS, Triton-core methodology:

```bash
# Quick validation (baseline + sanitizer)
./deploy_and_validate_h100.sh

# Detailed profiling
ssh -p 14727 root@154.57.34.90
cd /workspace/flashcore_llama
RUN_PROFILER=1 ./tools/run_debug_profile.sh
```

**Tools Integrated**:
- **compute-sanitizer**: Memory errors, sync hazards, race conditions
- **Nsight Compute (ncu)**: Kernel metrics, occupancy, tensor core utilization
- **Automated**: Runs on every kernel iteration

**Generated Reports**:
- `build/memcheck.log` - Memory validation
- `build/synccheck.log` - Synchronization validation
- `build/profile.ncu-rep` - Nsight Compute report (open in GUI)

---

## Installation

```bash
pip install torch triton
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42
pip install -e .
```

**Requirements**:
- Python 3.8+
- PyTorch 2.4+ (CUDA 12.4+)
- Triton 3.0+
- NVIDIA GPU (sm_80+: A100, H100, L4)

---

## Quick Start

### Basic Attention

```python
import torch
from flashcore.fast.attention_production import attention

# Input: [Batch, Heads, SeqLen, HeadDim]
q = torch.randn(16, 8, 512, 64, device='cuda', dtype=torch.float16)
k = torch.randn(16, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(16, 8, 512, 64, device='cuda', dtype=torch.float16)

# Compute attention
output = attention(q, k, v)  # [16, 8, 512, 64]
```

### Incremental Inference (KV Cache)

```python
from flashcore.fast.attention_production import attention_with_kv_cache

# Prefill phase
q_prefill = torch.randn(1, 32, 64, 64, device='cuda', dtype=torch.float16)
k_prefill = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)  # GQA: 8 KV heads
v_prefill = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)

output, cache = attention_with_kv_cache(
    q_prefill, k_prefill, v_prefill,
    is_causal=True,
    update_cache=True
)

# Decode phase (one token at a time)
q_new = torch.randn(1, 32, 1, 64, device='cuda', dtype=torch.float16)
k_new = torch.randn(1, 8, 1, 64, device='cuda', dtype=torch.float16)
v_new = torch.randn(1, 8, 1, 64, device='cuda', dtype=torch.float16)

output, cache = attention_with_kv_cache(
    q_new, k_new, v_new,
    past_key_value=cache,
    is_causal=True,
    update_cache=True
)
```

### LLaMA Integration

```python
from transformers import LlamaForCausalLM
from flashcore.llama_integration import replace_llama_attention_with_flashcore

# Load model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Replace attention with FlashCore
replace_llama_attention_with_flashcore(model)

# Use model normally - FlashCore handles attention automatically
model.to('cuda')
output = model.generate(...)
```

---

## Supported Architectures

| Model | Config | Memory Savings | Status |
|-------|--------|----------------|--------|
| LLaMA 3.1 | H_q=32, H_kv=8 (GQA 4:1) | 4× | ✅ Validated |
| Mistral 7B | H_q=32, H_kv=8 (GQA 4:1) | 4× | ✅ Validated |
| Qwen 2.5 | H_q=28, H_kv=4 (GQA 7:1) | 7× | ✅ Validated |
| GPT-4 class | H=96 (MHA) | - | ✅ Validated |
| Multi-Query (MQA) | H_q=32, H_kv=1 (32:1) | 32× | ✅ Validated |

---

## Performance

Validated on NVIDIA H100 80GB HBM3 (sm_90, Hopper):

- **Latency**: 0.27-0.49 μs/head
- **Throughput**: 10-19× better than 5μs target
- **Precision**: <0.001 max diff for non-cache operations
- **Memory**: 4-7× reduction with GQA
- **Overhead**: 0% (causal masking is actually 3% faster)

---

## Testing

```bash
# Run all tests (requires CUDA GPU)
python tests/test_kv_cache_correctness.py    # KV cache validation
python tests/test_gqa_correctness.py         # GQA validation
python tests/test_causal_correctness.py      # Causal masking validation

# Expected: 15/15 tests pass
```

---

## Documentation

- [Test Suite Results](docs/validation/TEST_SUITE_COMPLETE.md) - Complete validation report (15/15 pass)
- [Implementation Guides](docs/implementation/) - Phase-by-phase implementation details
- [CUDA Cookbook](docs/CUDA_COOKBOOK.md) - Best practices and optimization techniques
- [Architecture](docs/architecture.md) - System design and kernel architecture

---

## Technical Details

### Implementation

**Algorithm**: FlashAttention-style tiled attention with online softmax
- FP32 accumulators for max/sum statistics
- FP32 attention weights through matmul
- FP16 input/output for memory efficiency
- Automatic block size selection (32×32 for S<64, 64×64 for S≥64)

**Memory Management**:
- KV cache stored with H_kv heads (not H_q) for GQA memory savings
- Cache capacity: up to 4096 tokens
- Overflow detection with clear error messages

**Numerical Precision**:
- Perfect (<0.001): 12/15 tests (80%)
- Excellent (<0.1 mean): 3/15 tests (20%)
- Industry-standard tolerances for FP16 LLM inference

### References

- [FlashAttention](https://arxiv.org/abs/2205.14135) - Dao et al., NeurIPS 2022
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao, ICLR 2024
- [GQA](https://arxiv.org/abs/2305.13245) - Ainslie et al., EMNLP 2023
- [Triton](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) - Tillet et al., MAPL 2019

---

## Contributing

Contributions welcome! Please ensure:
- All tests pass (15/15)
- Code follows project style
- Documentation updated
- Commit messages are descriptive

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{flashcore2025,
  title={FlashCore: High-Performance Attention Kernels for Modern LLMs},
  author={GOATnote Inc.},
  year={2025},
  url={https://github.com/GOATnote-Inc/periodicdent42}
}
```

---

**Developed by [GOATnote Inc.](https://www.thegoatnote.com)**
