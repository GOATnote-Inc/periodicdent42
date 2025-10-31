# BlackwellSparseK: Block-Sparse Attention for H100

**High-performance block-sparse attention kernel with CUDA 13.0.2 + CUTLASS 4.3.0 CuTe DSL**

---

## 🎯 Features

- **Block-Sparse (BSR) layout** - Memory-efficient sparse attention
- **TMA async copy** - Hopper TMA with 3-stage pipeline
- **WMMA Tensor Cores** - 16×16×16 tiles, FP16→FP32 accumulation
- **sm_90a optimized** - H100 architecture-specific
- **Containerized** - Reproducible builds with Docker

---

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Warp Active** | ≥ 85% | Pipeline overlap |
| **Memory Stall** | ≤ 5% | TMA benefit |
| **Tensor Core** | ≥ 70% | WMMA utilization |
| **Bank Conflicts** | ~0 | Coalesced access |

---

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA runtime
- NVIDIA H100 GPU
- Driver ≥ 570 (for CUDA 13.0.2 compat layer)

### Build and Run
```bash
# Build container with kernel
make build

# Run kernel
make run

# Profile with Nsight Compute
make ncu

# Full verification (preflight + run)
make verify
```

---

## 📁 Project Structure

```
BlackwellSparseK/
├── .cursor/              # Cursor IDE guardrails
│   ├── rules.md          # Hard constraints (CUDA 13.0.2, sm_90a)
│   └── config.json       # Enforcement rules
├── Dockerfile            # CUDA 13.0.2 + CUTLASS 4.3.0
├── Makefile              # Build/run/profile targets
├── scripts/
│   └── preflight.sh      # Validation checks
└── src/
    └── sparse_bsr_gemm_h100.cu   # BSR + TMA kernel (sm_90a)
```

---

## 🔧 Development

### Container Environment
- **Base**: nvidia/cuda:13.0.2-devel-ubuntu22.04
- **CUTLASS**: v4.3.0 (CuTe headers only)
- **Tools**: Nsight Compute CLI, nvcc

### Compiler Flags
```bash
nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo -Xptxas -v \
     -I/opt/cutlass/include \
     -o sparse_h100 src/sparse_bsr_gemm_h100.cu
```

### Guardrails
See `.cursor/rules.md` for development constraints:
- ✅ Edit kernel code (must compile in container)
- ✅ Improve TMA/CuTe wiring
- ✅ Add Nsight metrics
- ❌ Change CUDA/CUTLASS versions
- ❌ Add Triton/PyTorch dependencies
- ❌ Break containerization

---

## 📈 Benchmarking

### Nsight Compute Metrics
```bash
make ncu
```

Captures:
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `smsp__stall_memory_dependency.avg.pct`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`

### Export Report
```bash
docker run --gpus all --rm sparsek-h100 \
  ncu --export /workspace/report.ncu-rep \
  --set full ./sparse_h100
```

---

## 🎓 Technical Details

### Kernel Configuration
```
Block Size:  128×128 CTA (256 threads)
Warp Tile:   64×64 (4 warps)
WMMA Tile:   16×16×16 (FP16 input, FP32 accumulator)
Pipeline:    3-stage TMA overlapped
Shared Mem:  ~36 KB (A/B tiles × 3 stages)
```

### BSR Format
```cpp
struct BSR {
    int* row_ptr;      // Row pointers [num_rows+1]
    int* col_idx;      // Column indices [nnz_blocks]
    half* vals;        // Values [nnz_blocks * BM * BN]
};
```

### TMA Pipeline
```cpp
using Pipe = PipelineTmaAsync<STAGES=3>;

// Producer (load)
pipe.producer_acquire(write_stage);
copy(tmaA, gA, sA, pipe, write_stage);
copy(tmaB, gB, sB, pipe, write_stage);
pipe.producer_commit(write_stage);

// Consumer (compute)
pipe.consumer_wait(read_stage);
// WMMA compute on smem[read_stage]
pipe.consumer_release(read_stage);
```

---

## 🏆 Related Work

- **CUTLASS**: NVIDIA's CUDA Templates for Linear Algebra ([GitHub](https://github.com/NVIDIA/cutlass))
- **CuTe**: CUTLASS Utilities for Tensor Expressions (CUTLASS 3.x+)
- **FlashAttention**: Memory-efficient exact attention ([Paper](https://arxiv.org/abs/2205.14135))
- **SparseK**: Learnable block-sparse attention ([Paper](https://arxiv.org/abs/2406.16747))

---

## 📄 License

MIT License with Ethical Use Clause

Copyright (c) 2025 BlackwellSparseK Contributors

---

## 🤝 Contributing

This project enforces strict versioning:
- CUDA 13.0.2 (no exceptions)
- CUTLASS 4.3.0 (CuTe only)
- sm_90a H100 target

See `.cursor/rules.md` for contribution guidelines.

---

## 📧 Contact

For questions or issues, see project documentation in `docs/` or create an issue.

---

**Status**: Guardrail kit deployed, kernel implementation pending  
**Last Updated**: October 30, 2025
