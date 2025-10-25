# 🎯 Kernel Optimization Focus - October 16, 2025

## Current Status: Ready for Systematic Optimization

### What We Have
**Working Kernel**: `cudadent42/bench/kernels/fa_s512_v3.cu`
- **Current Performance**: 38.00 μs (B=2, H=8, S=512, D=64)
- **PyTorch SDPA Baseline**: 47.10 μs (slow/unoptimized)
- **Current Standing**: ✅ **Already 21% faster than PyTorch!**

### Target Hardware
**NVIDIA L4** (Ada Lovelace, sm_89)
- **SMEM**: 48 KB max (NOT 64 KB - critical constraint)
- **Tensor Cores**: 242 TFLOPS @ FP16 accumulation, 121 TFLOPS @ FP32
- **L2 Cache**: 48 MB (massive - can fit entire KV cache)
- **Memory Bandwidth**: 300 GB/s

### Known Issues from Past Sessions
(See `docs/archive/session_logs/` for full details)

1. **SMEM Overflow** (`ITER1_CRITICAL_FINDINGS.md`)
   - Using `float` for intermediate results → 69 KB SMEM (exceeds 48 KB limit)
   - Solution: Use `half` for S_smem, QK_smem
   
2. **fa_s512.cu Fundamentally Broken** (`ITER1_CRITICAL_FINDINGS.md`)
   - Misaligned address errors for all batch sizes
   - Documented baseline of 321 μs was never actually measured
   - DO NOT use as optimization baseline

3. **Bank Conflicts** (`L4_ROADMAP_RECONCILED.md`)
   - HEAD_DIM=64 × 2 bytes = 128 bytes = 32 banks
   - Without swizzling: 32-way serialization (32× slower SMEM access)
   - Solution: XOR swizzling or padding

4. **FP32 Accumulation Penalty** (`L4_ROADMAP_RECONCILED.md`)
   - Ada Tensor Cores: FP16 accumulation = 242 TFLOPS
   - FP32 accumulation = 121 TFLOPS (50% slower!)
   - Solution: Use `wmma::fragment<..., half>` for accumulators

### Optimization Strategy (EvoEngineer-Insight)

**Phase 1: SMEM Optimization** (Target: 32-35 μs)
- [ ] Convert S_smem to `half` (if not already done)
- [ ] Add XOR swizzling for K/V SMEM layout
- [ ] Verify SMEM usage < 48 KB with `ptxas --verbose`

**Phase 2: Vectorized I/O** (Target: 28-32 μs)
- [ ] Use `uint4` for 128-bit loads (8×fp16 at once)
- [ ] Ensure 16-byte alignment
- [ ] Coalesced global memory access

**Phase 3: Tensor Core Integration** (Target: 15-25 μs)
- [ ] WMMA for Q·K^T (FP16 accumulation)
- [ ] Warp tiling (2×2 tiles = 32×32 output per warp)
- [ ] Verify `sm__inst_executed_pipe_tensor` > 0 in Nsight

**Phase 4: L2 Cache Persistence** (Target: 12-18 μs)
- [ ] Pin K/V in L2 using `cudaStreamSetAttribute`
- [ ] 48 MB L2 can fit entire attention for small batches

### Quick Start Commands

```bash
# Navigate to kernel directory
cd ~/periodicdent42/cudadent42/bench

# Build current kernel
python build_v3_release.py

# Run quick benchmark (< 60s)
cd ../../scripts
python bench_v3_quick.py

# Profile with Nsight (on GPU)
cd ~/periodicdent42
make profile  # Generates artifacts/profile/latest.ncu-rep
```

### Success Metrics

| Metric | Current | Target (Phase 1) | Target (Phase 3) |
|--------|---------|------------------|------------------|
| Latency (p50) | 38.00 μs | 32-35 μs | 15-25 μs |
| vs PyTorch | 1.24× faster | 1.35-1.47× faster | 1.88-3.14× faster |
| SMEM Usage | ~41 KB | < 48 KB | < 48 KB |
| TC Utilization | 0% | 0% | > 50% |

### Critical Rules

1. **ONE change at a time** - Measure before next change
2. **Correctness first** - `torch.allclose(atol=1e-3, rtol=1e-3)` must pass
3. **No regressions** - Performance must improve or stay neutral
4. **Document everything** - Update comments for each optimization
5. **L4-specific** - Don't assume Ampere (sm_80) behavior

### Files to Focus On

```
cudadent42/bench/kernels/fa_s512_v3.cu       # Primary kernel (805 lines)
cudadent42/bench/kernels/fa_s512_v3_bindings.cpp  # PyBind11 interface
cudadent42/bench/build_v3_release.py          # Build script
scripts/bench_v3_quick.py                      # Quick benchmark
.cursor/rules/kernel_optimization.md          # AI rules for edits
```

### References

- **L4 Optimization Guide**: `docs/archive/session_logs/L4_ROADMAP_RECONCILED.md`
- **Critical Findings**: `docs/archive/session_logs/ITER1_CRITICAL_FINDINGS.md`
- **EvoEngineer Paper**: Summary in session logs
- **PyTorch SDPA Baseline**: 47.10 μs (verified Oct 16, 2025)

---

**Last Updated**: 2025-10-16 15:05 PDT  
**Status**: ✅ Repository clean, GPU stopped, ready to optimize  
**Next Action**: Phase 1 - SMEM optimization (convert to `half`, add swizzling)
