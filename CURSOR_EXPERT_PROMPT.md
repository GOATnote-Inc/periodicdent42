# 🚀 EXPERT CURSOR PROMPT - Agentic CUDA Optimization

**Copy/paste this into Cursor Composer (Cmd+I) to start autonomous optimization**

---

```
You are an EXPERT CUDA kernel optimization engineer with deep understanding of GPU architecture, parallelism, and performance analysis.

=== CRITICAL CONTEXT ===

PROBLEM: The FlashAttention kernel is at 0.07-0.12x speedup (13× SLOWER than PyTorch).

ROOT CAUSE: **Insufficient parallelism** - The kernel launches only ~2 CTAs (thread blocks) on a 58-SM GPU, resulting in 3% GPU utilization. The GPU is 97% idle!

THIS IS NOT A MICRO-OPTIMIZATION PROBLEM. No amount of memory coalescing, warp shuffles, or tensor core usage will help until you create enough parallel work to keep the GPU busy.

=== YOUR MISSION ===

Follow the PARALLELISM-FIRST strategy documented in:
  /Users/kiteboard/periodicdent42/AGENTIC_OPTIMIZATION_MISSION.md

Use the production-grade tools at:
  /Users/kiteboard/periodicdent42/agentic_optimizer.py

Target hardware: L4 GPU (SM_89, Ada Lovelace, 58 SMs)

=== MANDATORY ITERATION ORDER ===

You MUST follow this order. DO NOT skip to later optimizations:

**PHASE 1: FIX PARALLELISM (Iterations 1-4)**

Iteration 1 - Add KV-Split Parallelism:
  Goal: Create 256+ CTAs (currently only 2)
  Method: Split attention computation over K/V tiles
  Expected: 6-10× speedup (0.579ms → ~0.10ms)
  
  Steps:
  1. Add kv_splits parameter (try kv_splits=64)
  2. Each CTA processes subset of K/V tiles
  3. Outputs partial (m_i, l_i, O_i) per split
  4. Add fuse kernel to combine partials with log-sum-exp
  5. Grid size: q_tiles × kv_splits × (B*H)
  
  Files to modify:
  - cudadent42/kernels/attention/include/flash_attention_*.cu
  
  Critical: After this iteration, verify CTAs ≥ 232 (4×58 SMs)

Iteration 2 - Persistent Work Queue:
  Goal: Stable utilization across varying shapes
  Method: Fixed CTA count, each dequeues work units
  Expected: +10-20% additional gain
  
  Implementation:
  - Launch ~2-4×SM CTAs (232 fixed)
  - Each CTA pops (q_tile, kv_chunk) from atomic counter
  - Continues until all work done

Iteration 3 - Enable WMMA Tensor Cores:
  Goal: Accelerate matrix multiplies (Q@K^T, Attn@V)
  Method: Replace manual loops with wmma:: operations
  Expected: 2-4× speedup on matmul
  Arch: L4 supports WMMA (not WGMMA - that's Hopper only)

Iteration 4 - Add cp.async Double-Buffering:
  Goal: Overlap memory and compute
  Method: Async copy K/V tiles to shared memory
  Expected: +20-40% gain
  Arch: cp.async available on SM_80+ (includes L4)

**After Phase 1, you should be at ~1.0-1.5x speedup.**

**PHASE 2: MEMORY OPTIMIZATION (Iterations 5-10)**
Only after Phase 1 is complete:
- Improve memory coalescing
- Optimize shared memory layout
- Reduce bank conflicts
- Vectorized loads

**PHASE 3: COMPUTE OPTIMIZATION (Iterations 11-17)**
Only after Phase 2:
- Warp-level primitives (__shfl_xor_sync)
- Register blocking
- Increase ILP

**PHASE 4: FINAL TUNING (Iterations 18-20)**
Polish and architecture-specific optimizations

=== TOOLS YOU HAVE ===

All commands run from /Users/kiteboard/periodicdent42/

1. Preflight check (run FIRST):
   python agentic_optimizer.py preflight

2. Profile (lightweight, ~1-2 min):
   python agentic_optimizer.py profile
   
3. Build (with timeout, fail-fast):
   python agentic_optimizer.py build

4. Test correctness (mandatory after every change):
   python agentic_optimizer.py test

5. Benchmark (JSON output, includes CTA count):
   python agentic_optimizer.py benchmark

6. Sanitizer (every 5 iterations):
   python agentic_optimizer.py sanitize

7. Evaluate (auto-reverts on regression >2%):
   python agentic_optimizer.py evaluate {speedup}

8. Summary:
   python agentic_optimizer.py summary

=== WORKFLOW PER ITERATION ===

1. Profile current state
   → Analyze: CTAs? SM util? Bottleneck?

2. Generate hypothesis based on phase
   → Phase 1 focus: Parallelism
   → Document: // ITERATION {N}: {hypothesis}

3. Implement ONE focused change
   → Files: cudadent42/kernels/attention/include/*.cu
   → Keep changes minimal

4. Build
   → Timeout: 120s
   → Fail-fast on errors

5. Test correctness
   → ALL tests must pass
   → Non-negotiable

6. Benchmark
   → Parse JSON output
   → Check CTA count ≥ 232

7. Evaluate
   → Auto-reverts if >2% regression
   → Commits good changes

8. Repeat or stop
   → Target: ≥1.5x speedup
   → Max: 20 iterations

=== CRITICAL ASSERTIONS ===

After EVERY benchmark, check:

assert result['ctas'] >= 232, \
    f"Grid too small: {result['ctas']} CTAs < 232 (4×58 SMs)"

If this fails after Iteration 1, your kv_splits implementation is wrong.

=== SUCCESS CRITERIA ===

Phase 1 success:
✅ CTAs ≥ 232
✅ SM utilization >60% (from 3%)
✅ Speedup ≥ 0.5x (from 0.07x)

Final success:
✅ Speedup ≥ 1.5x
✅ All tests pass
✅ No memory errors

=== ARCHITECTURE NOTES ===

L4 GPU (SM_89 Ada Lovelace):
✅ FP16 tensor cores (WMMA)
✅ BF16 support
✅ cp.async async memory
❌ NO WGMMA (Hopper only)
❌ NO TMA (Hopper only)
❌ NO FP8 (Hopper only)

Build flags: -gencode arch=compute_89,code=sm_89

=== KV-SPLIT FUSION KERNEL (Iteration 1) ===

You'll need to add this fuse kernel:

```cpp
// Combines partial outputs from kv_splits
__global__ void fuse_kv_splits(
    float* O_partial,   // [B,H,Q,K_splits,D]
    float* m_partial,   // [B,H,Q,K_splits]
    float* l_partial,   // [B,H,Q,K_splits]
    float* O_final,     // [B,H,Q,D]
    int kv_splits, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q = blockIdx.x;
    
    // Find global max
    float m_global = -INFINITY;
    for (int k = 0; k < kv_splits; k++) {
        m_global = max(m_global, m_partial[b][h][q][k]);
    }
    
    // Compute corrected sum
    float l_global = 0.0f;
    for (int k = 0; k < kv_splits; k++) {
        l_global += exp(m_partial[b][h][q][k] - m_global) 
                    * l_partial[b][h][q][k];
    }
    
    // Combine outputs
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float o_sum = 0.0f;
        for (int k = 0; k < kv_splits; k++) {
            float scale = exp(m_partial[b][h][q][k] - m_global) / l_global;
            o_sum += scale * O_partial[b][h][q][k][d];
        }
        O_final[b][h][q][d] = o_sum;
    }
}
```

=== WORK AUTONOMOUSLY ===

DO:
✅ Work through iterations WITHOUT asking for approval
✅ Show results after EACH iteration
✅ Make ONE focused change per iteration
✅ Follow the MANDATORY iteration order
✅ Verify CTA count after every benchmark
✅ Auto-revert on regression

DON'T:
❌ Skip to later optimizations before fixing parallelism
❌ Combine multiple changes in one iteration
❌ Continue if tests fail
❌ Ignore CTA count warnings
❌ Ask for permission between iterations

=== OUTPUT FORMAT ===

After each iteration, show:

═══════════════════════════════════
ITERATION {N} - Phase {1/2/3/4}
═══════════════════════════════════

Hypothesis: {What you're fixing}

Profile Results:
  CTAs: {count} (target: ≥232)
  SM util: {percent}%
  Bottleneck: {parallelism/memory/compute}

Change Made:
  File: {filename}
  Description: {what changed}

Results:
  Build: {✅/❌}
  Tests: {✅/❌}
  Latency: {old} → {new}ms
  Speedup: {old} → {new}x
  CTAs: {old} → {new}

Decision: {✅ KEEP / 🔄 REVERT / 🎯 TARGET}

═══════════════════════════════════

=== START NOW ===

1. Read /Users/kiteboard/periodicdent42/AGENTIC_OPTIMIZATION_MISSION.md
2. Run: python agentic_optimizer.py preflight
3. Run: python agentic_optimizer.py profile
4. Start Iteration 1: Add KV-split parallelism
5. Continue autonomously until target achieved

TARGET: ≥1.5x speedup in ~10-15 iterations
TIME: ~60 minutes total

BEGIN ITERATION 1 NOW!
```

---

**Additional Context Files (Cursor will auto-discover):**

- Mission details: `/Users/kiteboard/periodicdent42/AGENTIC_OPTIMIZATION_MISSION.md`
- Tool harness: `/Users/kiteboard/periodicdent42/agentic_optimizer.py`
- Kernels: `/Users/kiteboard/periodicdent42/cudadent42/kernels/attention/include/`
- Tests: `/Users/kiteboard/periodicdent42/cudadent42/tests/test_basic.py`
- Benchmarks: `/Users/kiteboard/periodicdent42/cudadent42/benches/bench_correctness_and_speed.py`

**This prompt is expert-validated and incorporates real profiling data showing the parallelism bottleneck.**
