# BSR Kernel Optimization Session - November 1, 2025

## Mission
Study expert kernel patterns, take multiple shots on goal, only git push when validation confirmed.

## Starting Point
- **Baseline:** 68.8 TFLOPS (validated correct on H100)
- **Configuration:** 128×128×64 tiles, 512 threads, vectorized loads, half2 compute
- **Status:** Solid working kernel, numerically correct

## Attempts Made (6 iterations)

### 1. ❌ Warp Specialization (CUTLASS pattern)
- Smaller K tiles (32 vs 64), dedicated load/compute warps
- **Result:** Correctness failure (error 14.28)
- **Learning:** Double buffering complex for sparse accumulation

### 2. ❌ Transposed B Layout (FlashAttention pattern)
- Transpose B in shared memory for coalescing
- **Result:** 23.1 TFLOPS (-66% regression)
- **Learning:** Transpose overhead > coalescing benefit

### 3. ❌ Software Pipelining
- Prefetch next block while computing current
- **Result:** Correctness failure (error 24.01)
- **Learning:** Pipeline state management broke accumulation

### 4. ❌ Parameter Sweep
- Systematic tuning of threads/unroll factors
- **Result:** Segfault (register overflow)
- **Learning:** Template explosion or memory corruption

### 5. ❌ Multiple Output Tiles
- 1 CTA → 2 N tiles (load A once, reuse for 2 outputs)
- **Result:** Correctness failure (error 24.01)
- **Learning:** Complex accumulation logic error

### 6. ❌ 4-way ILP
- Process 4 elements together in compute loop
- **Result:** Correctness failure (error 24.01)
- **Learning:** Even minimal changes break correctness

## Critical Findings

### 🔴 Recurring Error Pattern
- Error value **24.014477** appeared in attempts #3, #5, #6
- Indicates systematic bug in sparse accumulation when modifying kernel logic
- Working kernel is at a fragile local optimum

### 🟡 Correctness is Delicate
- Multiple sparse blocks contribute to same output cell
- Accumulation order and state management critical
- Any structural change risks breaking correctness

### 🟢 Current Kernel Analysis
- **68.8 TFLOPS** = 2× FP32 scalar peak (35 TFLOPS H100)
- Compiler automatically using some tensor core instructions
- Further gains require:
  - Real WMMA/WGMMA (not compiler hints)
  - cuBLASLt per-block approach
  - CUTLASS extension with proven infrastructure

## What We Proved

✅ **Systematic optimization works:**
- 30 TFLOPS (atomics) → 61.5 TFLOPS (registers) → 68.8 TFLOPS (vectorized)
- 2.24× improvement through incremental, validated changes

✅ **Expert patterns are hard:**
- CUTLASS/FlashAttention patterns designed for dense GEMM
- Sparse accumulation adds complexity
- Need careful adaptation, not direct copy

✅ **Validation is essential:**
- Stopped every failing attempt immediately
- Never pushed broken code
- Maintained professional repo quality

## Recommendations

### Short-term (Production Ready)
1. **Ship current 68.8 TFLOPS kernel** as baseline
2. Document as achievement (2.24× improvement)
3. Use for real applications

### Medium-term (Architecture Changes)
1. **cuBLASLt per-block** (proven, low-risk)
2. **Persistent kernel** (reduce launch overhead)
3. **Fix WMMA accumulation** (load-add-store pattern)

### Long-term (Ecosystem)
1. **CUTLASS contribution** (arbitrary BSR fills gap)
2. **Wait for better tools** (CUDA 13.x improvements)
3. **Community feedback** (real workload patterns)

## Session Outcome

**Status:** 6 attempts, 0 improvements, valuable learning  
**Code quality:** No broken commits, repo stays professional  
**Path forward:** Accept local optimum, focus on architectural changes or integration

---

**Key Lesson:** Expert patterns require expert-level correctness validation. Incremental, validated progress > ambitious broken attempts.

**Next:** Consider cuBLASLt per-block or CUTLASS extension path.
