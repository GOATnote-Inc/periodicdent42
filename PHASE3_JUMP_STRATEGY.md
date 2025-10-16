# Phase 3 Jump Strategy - Tensor Cores First

**Decision Date**: October 16, 2025  
**Rationale**: PyTorch SDPA baseline is 47.10μs (SLOW category), indicating massive optimization headroom. Skip scalar/memory phases and go directly to WMMA with safety guardrails.

**References**:
- EvoEngineer Framework: https://arxiv.org/html/2510.03760v1
- KernelBench Dataset: https://github.com/ScalingIntelligence/KernelBench

---

## Go/No-Go Checklist ✅

| Requirement | Status | Notes |
|-------------|--------|-------|
| **FP16 inputs** | ✅ | Q, K, V are `torch.float16` |
| **FP32 accum** | ✅ | WMMA accumulator for Q·K^T uses `float` |
| **Shapes divisible by 16** | ✅ | S=512, D=64 (both 16×n) |
| **USE_WMMA toggle** | ✅ | Build flag exists, easy fallback |
| **Causal mask** | ✅ | Pre-scale + mask before softmax |

**Verdict**: ✅ All green → Proceed with Phase 3

---

## Minimal WMMA Configuration (Ada sm_89)

### Tile Configuration
```
CTA Tile:
- M (query rows):  128
- N (key columns): 64
- K (head dim step): 32

Threading:
- Threads/CTA: 256 (8 warps)
- Target occupancy: 6-10 CTAs/SM

Memory:
- STAGES: 2 (double buffer for K, V)
- SMEM budget: ~24 KB (well under 48 KB limit)
  * Q: 128×32×2B = 8 KB
  * K: 32×64×2B = 4 KB  
  * V: 32×64×2B = 4 KB
  * ×2 stages = 24 KB + epilogue scratch
```

### WMMA Details (m16n16k16)
- **Fragment type**: `wmma::fragment<matrix_a, 16, 16, 16, half, row_major>`
- **Accumulator**: `wmma::fragment<accumulator, 16, 16, 16, float>` (FP32 for Ada 2× throughput)
- **Load from**: Shared memory (NOT registers - previous bug source)

### Compiler Flags
```bash
-DUSE_WMMA=1
-DTILE_M=128
-DTILE_N=64
-DTILE_K=32
-DSTAGES=2
-DACCUM_F32=1
-arch=sm_89
```

---

## Implementation Order (3 Commits)

### Commit 1: Q·K^T WMMA Microkernel
**Files**: `cudadent42/bench/kernels/fa_s512_v3_wmma.cu`

**Components**:
1. SMEM tile loaders with XOR swizzle (bank conflict mitigation)
2. WMMA fragment loading (`load_matrix_sync`)
3. WMMA matmul (`mma_sync`) with FP32 accumulator
4. Loop over K dimension (Kstep=32)

**Success Criteria**:
- Compiles without "local memory" warnings
- Nsight: `sm__inst_executed_pipe_tensor.pct_of_peak ≥ 40%`
- Nsight: Bank conflicts < 5%

### Commit 2: Rowwise Softmax (Warp-Cooperative)
**Components**:
1. Warp shuffle for rowwise max
2. Exp computation
3. Warp shuffle for sum
4. Probability normalization

**Success Criteria**:
- Numerical correctness: `torch.allclose(atol=1e-2, rtol=1e-2)`
- No NaN/Inf outputs

### Commit 3: P·V with cp.async (Optional First Pass)
**Components**:
1. FMA epilogue for P·V (simple)
2. cp.async for K/V double buffering (STAGES=2)
3. Commit groups + wait_group(1)

**Success Criteria**:
- Nsight: `dram__throughput.pct_of_peak ≥ 60%`
- Performance: < 25μs (2× faster than PyTorch 47μs)

---

## Nsight Gates (Accept/Reject)

| Metric | Target | Command |
|--------|--------|---------|
| **Tensor Core utilization** | ≥ 40% | `ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` |
| **DRAM bandwidth** | ≥ 60% | `ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| **Bank conflicts** | < 5% | `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` |
| **Occupancy** | ≥ 4 CTAs/SM | `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active` |
| **Register usage** | ≤ 64/thread | Check compilation output |
| **SMEM usage** | ≤ 48 KB/CTA | Check compilation output |

**Fallback Rule**: If any gate fails → flip `USE_WMMA=0` and continue without blocking branch.

---

## EvoEngineer-Insight Configuration

Following EvoEngineer framework (Table 3 from paper):

| Component | Our Implementation |
|-----------|-------------------|
| **Task Context (I1)** | Shapes (S=512, B=2, H=8, D=64), Tiles (M=128, N=64, K=32), Constraints (SMEM ≤48KB, Regs ≤64) |
| **Historical Solutions (I2)** | ❌ SKIP (no previous working kernel) |
| **Optimization Insights (I3)** | ✅ Nsight metrics, register pressure warnings, bank conflict analysis |
| **Population Strategy** | Single best solution (EvoEngineer-Insight) |

**Expected Performance** (from EvoEngineer Table 4):
- **Speedup**: 1.47-1.60× over baseline
- **Validity**: 58-63% (acceptable for exploration)
- **Token usage**: Medium (balanced)

---

## Quick Run Commands

### Build and Test WMMA
```bash
# On GPU instance
cd ~/periodicdent42

# Enable WMMA
export USE_WMMA=1 TILE_M=128 TILE_N=64 TILE_K=32 STAGES=2
python3 cudadent42/bench/build_v3_release.py

# Run benchmark
python3 -c "
import torch
import cudadent42_ops
B, H, S, D = 2, 8, 512, 64
q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
k, v = q.clone(), q.clone()
out = cudadent42_ops.flash_attention_s512_v3_forward(q, k, v, config_id=0)
print(f'✓ Output shape: {out.shape}, dtype: {out.dtype}')
"

# Nsight profiling
ncu --set full -o v3_wmma_profile python3 bench_v3.py
```

### Fallback to Scalar
```bash
export USE_WMMA=0
python3 cudadent42/bench/build_v3_release.py
# Same benchmark command
```

---

## Risk Mitigation

### Known Failure Modes

1. **Register blow-up** (occupancy < 4 CTAs/SM)
   - **Fix**: Narrow fragment live ranges, use `__restrict__`, avoid large scratch arrays
   - **Detection**: Compilation output shows regs/thread > 64

2. **Bank conflicts** (SMEM throughput < 60%)
   - **Fix**: XOR swizzle: `((row >> 2) ^ (col >> 4)) & 0x7`
   - **Detection**: Nsight shows `shared_load_transactions_per_request > 5.0`

3. **IO-bound** (low tensor utilization despite WMMA)
   - **Fix**: Add cp.async (STAGES=2), check L2 hit rate
   - **Detection**: Nsight shows `dram__throughput > 80%` AND tensor utilization < 40%

### Emergency Fallback
```bash
# If WMMA is fundamentally broken
export USE_WMMA=0
# Continue with scalar baseline (slower but correct)
```

---

## Success Criteria (Phase 3 Complete)

| Metric | Target | Baseline (PyTorch) |
|--------|--------|--------------------|
| **Latency (p50)** | < 25 μs | 47.10 μs |
| **Speedup** | ≥ 2× | 1.0× (baseline) |
| **Correctness** | `torch.allclose(atol=1e-2)` | Reference |
| **Tensor Core %** | ≥ 40% | 0% (PyTorch uses Flash Attention v1) |
| **Occupancy** | ≥ 4 CTAs/SM | N/A |

**Final Gate**: If latency < 25μs AND correctness passes → **Phase 3 SUCCESS** ✅

---

## Timeline

| Task | Duration | Owner |
|------|----------|-------|
| Build system updates | 30 min | Agent |
| WMMA microkernel (Q·K^T) | 2-3 hours | EvoEngineer-Insight (Claude) |
| Softmax implementation | 1 hour | Agent |
| P·V epilogue | 1 hour | Agent |
| cp.async integration | 2 hours | Agent |
| Nsight validation | 1 hour | Agent |
| **Total** | **7-9 hours** | |

---

## References

1. **EvoEngineer Paper**: https://arxiv.org/html/2510.03760v1
   - Section 4.2: EvoEngineer-Insight configuration (Task + Insights, no Historical)
   - Table 4: Expected 1.47-1.60× speedup, 58-63% validity

2. **KernelBench Dataset**: https://github.com/ScalingIntelligence/KernelBench
   - Evaluation methodology for correctness (5 random inputs)
   - Performance benchmarking with `torch.utils.benchmark.Timer`

3. **NVIDIA WMMA Documentation**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
   - Ada (sm_89) m16n16k16 fragment types
   - FP32 accumulation for 2× throughput

4. **Phase 0 Baseline**: `phase0_baseline_results.txt`
   - PyTorch SDPA: 47.10 μs (SLOW category)
   - L4 utilization: 9.4% (massive headroom)

---

**Status**: ✅ Ready to implement. All guardrails in place. Fallback strategy defined.  
**Next**: Generate WMMA microkernel using EvoEngineer-Insight prompt.

