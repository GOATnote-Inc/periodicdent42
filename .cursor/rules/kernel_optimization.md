# Cursor Rules: CUDA Kernel Optimization

## Context: We are optimizing FlashAttention kernels for NVIDIA L4 (Ada, sm_89)

### Primary Target
**File**: `cudadent42/bench/kernels/fa_s512_v3.cu` (805 lines)
- Has region markers: `BEGIN KERNEL_TRAITS`, `BEGIN COMPUTE_BLOCK`, etc.
- **Current baseline**: 38.00 μs (already 21% faster than PyTorch SDPA @ 47.10 μs)
- **Goal**: Systematic optimization using EvoEngineer-Insight methodology

### L4-Specific Constraints (sm_89)
1. **Shared Memory**: 48 KB max per block (not 64 KB like Ampere)
2. **Tensor Cores**: FP16 accumulation for 2× throughput (avoid FP32)
3. **Bank Conflicts**: HEAD_DIM=64 creates 32-way conflicts without swizzling
4. **L2 Cache**: 48 MB (leverage with `cudaStreamSetAttribute`)

### Optimization Priorities
1. **SMEM usage** - Stay under 48 KB (use `half` for intermediate results)
2. **Vectorized loads** - 128-bit (uint4) for 8×fp16 loads
3. **XOR swizzling** - Avoid bank conflicts for column-major access
4. **WMMA (Tensor Cores)** - Use FP16 accumulation, not FP32
5. **Warp-level primitives** - Prefer shuffle over atomics

### Edit Guidelines
- **Make ONE change at a time** - Verify correctness before next optimization
- **Use region markers** - Edit within `BEGIN X` / `END X` blocks only
- **Preserve comments** - Keep "Target", "SMEM Budget", "Registers" annotations
- **Add profiling hooks** - Use `#ifdef DEBUG_KERNEL` for conditional prints
- **Document changes** - Update comments for non-obvious optimizations

### Testing Protocol
```bash
# After EVERY edit:
cd cudadent42/bench
python build_v3_release.py              # Compile
python -c "import flash_attention_s512_v3; print('OK')"  # Verify import
cd ../../scripts
python bench_v3_quick.py                # Benchmark (< 60s)
```

### Success Criteria
- ✅ Compiles without warnings
- ✅ No CUDA errors (`compute-sanitizer` clean)
- ✅ Numerically correct (`torch.allclose` with atol=1e-3, rtol=1e-3)
- ✅ Performance improvement OR neutral (no regressions)

### Red Flags (Stop and Ask)
- ❌ SMEM usage > 48 KB
- ❌ Register usage > 128 (kills occupancy)
- ❌ Performance regression > 5%
- ❌ Correctness test fails
- ❌ Warp divergence introduced

### Output Format for Kernel Edits
```
## Change: [One-line description]

**Motivation**: [Why this optimization matters for L4]

**Implementation**: [What you changed]

**Expected Impact**: [Latency, SMEM, registers, bandwidth]

**Verification**: [How to test]
```

### Do NOT
- Do NOT edit `ext/flash-attention-2/**` (external submodule)
- Do NOT modify working kernels without backup
- Do NOT introduce CUDA 12.3+ features (target: CUDA 12.2)
- Do NOT assume Ampere (sm_80) behavior (we are sm_89 Ada)

---

**Focus**: Systematic, measurable, reversible optimizations for L4 Ada architecture.

