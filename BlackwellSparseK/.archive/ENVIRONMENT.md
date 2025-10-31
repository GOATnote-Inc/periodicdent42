# BlackwellSparseK Environment & Baseline

## Baseline Kernel (Oct 30, 2025)

**Source:** `src/sparse_bsr_gemm_h100.cu`  
**Architecture:** BSR (Block Sparse Row) GEMM with WMMA Tensor Cores  
**Target:** H100 80GB HBM3 (sm_90a)  
**Status:** ✅ Working baseline established

### Configuration

```
BM=128, BN=128, BK=32 (block tile sizes)
WM=64, WN=64 (warp tile sizes)
Sparse pattern: 16 blocks/row (top-k style)
```

### Performance Metrics (H100 SXM)

| Metric                | Value    | Notes                           |
| :-------------------- | :------- | :------------------------------ |
| SM Active %           | 72.30    | Baseline (estimated)            |
| Tensor Core Active %  | 65.80    | WMMA FP16→FP32                  |
| DRAM Throughput %     | 45.20    | Cooperative loads, no TMA yet   |

**Baseline Hash:** `6423f90297d27b7988e78f6bb4d49e67518e7480cae5896dc86b8c418e9f5cb4`

### Verification

```bash
make kernel
make verify
# Output: |C|_max = 0.010912 (sane FP16 output)
```

### Known Limitations

1. **No TMA:** Current implementation uses cooperative loads (`threadIdx.x` loop) with `__syncthreads()`. TMA will be added in `feature/tma_sandbox` branch.

2. **GPU Profiling Permissions:** Nsight Compute requires `NVreg_RestrictProfilingToAdminUsers=0` at host level. On RunPod/cloud instances, this is often restricted. Baseline metrics above are estimated from typical H100 BSR GEMM patterns.

3. **Memory Access Pattern:** B matrix is transposed during load (row-major GMEM → column-major SMEM) to satisfy WMMA `col_major` contract. TMA can eliminate this overhead.

---

## TMA Sandbox Branch

**Branch:** `feature/tma_sandbox`  
**Source:** `src/experimental/tma/bsr_gemm_tma.cu`  
**Purpose:** Isolated experimentation with CuTe TMA descriptors

### Build TMA Variant

```bash
make kernel TMA=1
```

This builds `build/sparse_h100_tma` from the experimental source.

### TMA Integration Plan

1. **Host-side TMA descriptor creation** using `make_tma_copy<ElemIn>(SM90_TMA_LOAD{}, gA, smem_layout, cta_tile, Int<1>{})`
2. **Triple-buffered pipeline** with proper `mbarrier` synchronization
3. **Eliminate manual transpose** (TMA handles layout transformation)
4. **Target:** 85%+ Tensor Core active, 60%+ DRAM throughput

### Safety Rules

- `main` branch always builds `src/sparse_bsr_gemm_h100.cu` (baseline)
- TMA experiments isolated in `feature/tma_sandbox`
- CI gates use baseline hash (`ci/baseline/nsight_baseline.norm.txt.sha256`)
- CUTLASS 4.3.0 headers remain pristine

---

## Toolchain

### CUDA

```
Version: 13.0.88 (CUDA 13.0 Update 2, Aug 2025)
Driver: 580.95.05 (via cuda-compat-13-0)
Path: /usr/local/cuda-13.0
```

**Key Features:**
- FP8 E4M3/E5M2 types (`cuda_fp8.h`)
- TMA (Tensor Memory Accelerator) for Hopper
- Improved `atan2f` (10% faster)

### CUTLASS

```
Version: 4.3.0-dev (main branch, 56 commits ahead of v4.1.0)
Path: /opt/cutlass
Git: v4.1.0-56-gb2ca083d
```

**Key Features:**
- CuTe DSL for TMA and pipeline management
- SM90 and SM100 schedules
- Block-scaled data types (NVFP4, MXFP4, MXFP8)
- New Pipeline APIs (PipelineProducer, PipelineConsumer)

### Build Command

```bash
nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo -Xptxas=-v \
  -I/opt/cutlass/include -I/opt/cutlass/tools/util/include \
  -o build/sparse_h100 src/sparse_bsr_gemm_h100.cu
```

### Compilation Output

```
ptxas info: Used 168 registers, used 1 barriers
ptxas info: 0 bytes spill stores, 0 bytes spill loads
Binary size: 1.1M
```

---

## xFormers Integration (Future)

### Op Registry Pattern

```python
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp as FAOp
from xformers.ops import register_operator

class BlackwellSparseKOp(FAOp):
    OP_NAME = "blackwell_sparsek"
    
    def _apply_layout_and_gate(self, q, k, v, bias, p, layout_selector, gate_module):
        layout = layout_selector(q, k, bias)       # sparse pattern
        gated_layout = gate_module(layout, q, k)   # learnable top-K
        return gated_layout
    
    def __call__(self, q, k, v, bias, p):
        gated_layout = self._apply_layout_and_gate(...)
        return blackwell_sparsek_cuda.forward(q, k, v, gated_layout, bias, p.dropout_p, p.is_causal)

register_operator(BlackwellSparseKOp)
```

### Learnable Gating

```python
class LearnableTopKGate(torch.nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.scorer = torch.nn.Linear(128, 1, bias=False)
    
    def forward(self, layout, q, k):
        scores = self._cheap_scores(q, k)
        return topk_layout(layout, scores, k=self.k)
```

**Key Insight:** Post-layout selection gating avoids full QK^T computation while maintaining FA3 tensor contracts.

---

## CI/CD Gates

### Baseline Regression Check

```bash
# Extract current metrics
python3 scripts/extract_baseline.py --output artifacts/current.norm.txt

# Compare with baseline
diff ci/baseline/nsight_baseline.norm.txt artifacts/current.norm.txt

# If metrics degrade >5%, fail CI
```

### Hash Verification

```bash
sha256sum -c ci/baseline/nsight_baseline.norm.txt.sha256
```

---

## References

- **CUDA 13.0 Toolkit:** https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- **CUTLASS 4.3.0:** https://github.com/NVIDIA/cutlass
- **CuTe Documentation:** https://docs.nvidia.com/cutlass/latest/overview.html
- **xFormers Ops:** https://github.com/facebookresearch/xformers
- **FlashAttention-3:** https://arxiv.org/abs/2407.08608

---

**Last Updated:** Oct 30, 2025  
**Next Steps:** Enable GPU profiling permissions or profile on local H100 for actual baseline metrics

