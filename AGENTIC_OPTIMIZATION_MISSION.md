# CUDA Kernel Agentic Optimization Mission - EXPERT VALIDATED

**CRITICAL**: This is a **parallelism-first** optimization mission. Your kernel is at 0.07-0.12x because it launches **~2 CTAs on a 58-SM GPU** (3% utilization). No amount of micro-optimization helps until you fix grid decomposition.

---

## 🚨 THE REAL PROBLEM (Read This First!)

**Status**: 0.579ms @ S=128 vs PyTorch 0.043ms (13.5× slower)

**Root Cause**: Grid decomposition creates too few thread blocks (CTAs):
- Current: S=128, B=1, H=1 → grid ~(1,1,2) → **2 CTAs**
- L4 GPU: **58 SMs** 
- Utilization: **2/58 = 3.4%** 🔥
- **Result**: GPU is 96% idle!

**Target**: ≥4×SM CTAs = **≥232 CTAs** to keep L4 busy

**Why your earlier fixes didn't work**: Fixing THREADS_PER_BLOCK (128→384) only affects per-CTA efficiency. When you only have 2 CTAs, the other 56 SMs sit idle. You must create more CTAs.

---

## 🎯 Mission: Fix Parallelism FIRST, Then Optimize

### **Iteration Priority (MANDATORY ORDER):**

#### **Phase 1: Parallelism (Iterations 1-4) - MUST DO FIRST**
These iterations fix GPU utilization from 3% → 60%+

**Iteration 1: Add KV-Split Parallelism**
- **Problem**: Too few CTAs when (B,H,S) are small
- **Solution**: Split attention over K/V tiles
- **Change**: Add `kv_splits` parameter (e.g., 64 splits)
- **Math**: `CTAs = q_tiles * kv_splits * (B * H)`
  - Before: 4 q_tiles → 4 CTAs
  - After: 4 q_tiles × 64 kv_splits → 256 CTAs ✅
- **Result**: Expect 5-10× speedup (0.579ms → ~0.10ms)
- **Implementation**:
  ```cpp
  // Each CTA computes attention on subset of K/V tiles
  // Produces partial (m_i, l_i, O_i) outputs
  // Fuse kernel combines partials with log-sum-exp trick
  ```

**Iteration 2: Add Persistent Work Queue**
- **Problem**: Static grid doesn't adapt to varying shapes
- **Solution**: Launch ~2-4×SM CTAs, each dequeues work units
- **Change**: CTAs pop (q_tile, kv_chunk) from atomic counter
- **Result**: Stable utilization across all shapes
- **Implementation**:
  ```cpp
  __global__ void persistent_attention(...) {
    while (true) {
      int work_id = atomicAdd(&global_work_counter, 1);
      if (work_id >= total_work_units) break;
      // Decode work_id → (q_tile, kv_split)
      // Process attention tile
    }
  }
  ```

**Iteration 3: Enable Tensor Cores (WMMA)**
- **Problem**: Running on SIMT path (slow FP16 ops)
- **Solution**: Use WMMA for matrix multiplies (Q@K^T, Attn@V)
- **Change**: Replace manual loops with `wmma::` ops
- **Result**: 2-4× speedup on matrix ops
- **Arch**: L4 (SM_89) supports WMMA, not WGMMA

**Iteration 4: Add Async Memory Copy (cp.async)**
- **Problem**: Synchronous loads waste compute cycles
- **Solution**: Use `cp.async` for K/V tile loads
- **Change**: Double-buffer K/V tiles in shared memory
- **Result**: Overlap compute and memory (20-40% gain)
- **Arch**: Available on SM_80+ (includes L4)

**After Phase 1**: Expect 0.579ms → ~0.05-0.10ms (10-12× improvement)

#### **Phase 2: Memory Optimization (Iterations 5-10)**
Now that GPU is busy, optimize memory access

**Iteration 5-7: Coalescing & Tiling**
- Improve memory access patterns
- Optimize shared memory layout
- Reduce bank conflicts

**Iteration 8-10: Advanced Memory**
- Vectorized loads (128-bit)
- Shared memory padding
- L2 cache optimization

#### **Phase 3: Compute Optimization (Iterations 11-17)**
**Iteration 11-13: Warp-Level Ops**
- `__shfl_xor_sync` for reductions
- Register blocking
- Increase ILP

**Iteration 14-17: Advanced Patterns**
- Warp-specialized roles
- Online softmax refinement
- Custom reduction patterns

#### **Phase 4: Final Tuning (Iterations 18-20)**
- Split-K optimization (if needed)
- Thread block tuning
- Architecture-specific paths

---

## 🛠️ Tools & Commands (Updated for Production)

### **Profiling (Lightweight, Fast)**
```bash
python agentic_optimizer.py profile
```

Uses **minimal Nsight metrics** (not `--set full`):
- SM throughput (compute bound?)
- Memory throughput (memory bound?)
- Warp occupancy
- Global load/store sectors

**Why not `--set full`?** Too slow (10-20 min), huge files, stalls agents.

### **Building**
```bash
python agentic_optimizer.py build
```

With timeout (120s max) and fail-fast on errors.

### **Benchmarking (JSON Output)**
```bash
python agentic_optimizer.py benchmark
```

Emits **machine-parsable JSON**:
```json
{"speedup_vs_torch": 1.23, "latency_ms": 0.91, "ctas": 256, "sm_count": 58}
```

No more fragile regex parsing!

### **Correctness Checks**
```bash
python agentic_optimizer.py test
# Plus every 5 iterations:
python agentic_optimizer.py sanitize  # compute-sanitizer
```

### **Evaluation**
```bash
python agentic_optimizer.py evaluate {speedup}
```

Auto-reverts on regression (>2% slowdown).

---

## 📊 Success Metrics (Updated)

### **Phase 1 Success (Parallelism Fixed)**
✅ CTAs ≥ 232 (4×SM for L4)
✅ SM utilization >60% (from 3%)
✅ Speedup: 0.07x → ~0.5x (7× gain)
✅ Latency: 0.579ms → ~0.10ms

### **Phase 2 Success (Memory Optimized)**
✅ Memory throughput >40%
✅ L2 hit rate >70%
✅ Speedup: ~0.5x → ~0.8x

### **Final Target (All Optimizations)**
✅ Speedup ≥ 1.5x vs PyTorch
✅ Latency < 0.03ms @ S=128
✅ All correctness tests pass
✅ No CUDA errors

**Realistic L4 target**: 0.2-0.4× PyTorch SDPA is **good** for a learning kernel. Hitting 1.0× is possible but requires expert tuning. **2.5×** is unrealistic on L4.

---

## 🔧 Critical Implementation Details

### **KV-Split Fusion Kernel (Iteration 1)**

After each CTA computes partial attention, you need to combine results:

```cpp
// Each CTA writes partial outputs:
// O_partial[b,h,q_tile,kv_split,d]  - partial output
// m_partial[b,h,q_tile,kv_split]     - partial max
// l_partial[b,h,q_tile,kv_split]     - partial sum

// Fuse kernel (1 CTA per (b,h,q_tile)):
__global__ void fuse_kv_splits(
    float* O_partial,      // [B,H,Q,K_splits,D]
    float* m_partial,      // [B,H,Q,K_splits]
    float* l_partial,      // [B,H,Q,K_splits]
    float* O_final,        // [B,H,Q,D]
    int kv_splits, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q = blockIdx.x;
    
    // Step 1: Find global max
    float m_global = -INFINITY;
    for (int k = 0; k < kv_splits; k++) {
        m_global = max(m_global, m_partial[b,h,q,k]);
    }
    
    // Step 2: Compute corrected sum
    float l_global = 0.0f;
    for (int k = 0; k < kv_splits; k++) {
        l_global += exp(m_partial[b,h,q,k] - m_global) * l_partial[b,h,q,k];
    }
    
    // Step 3: Combine outputs (per thread handles some columns)
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float o_sum = 0.0f;
        for (int k = 0; k < kv_splits; k++) {
            float scale = exp(m_partial[b,h,q,k] - m_global) / l_global;
            o_sum += scale * O_partial[b,h,q,k,d];
        }
        O_final[b,h,q,d] = o_sum;
    }
}
```

### **Persistent Kernel Pattern (Iteration 2)**

```cpp
__global__ void persistent_attention(
    float* Q, float* K, float* V, float* O,
    int* work_counter,  // atomic counter
    int total_work_units
) {
    // Each CTA loops until all work done
    while (true) {
        int work_id = atomicAdd(work_counter, 1);
        if (work_id >= total_work_units) break;
        
        // Decode work_id → (b, h, q_tile, kv_split)
        int kv_split = work_id % kv_splits;
        int rem = work_id / kv_splits;
        int q_tile = rem % q_tiles;
        rem = rem / q_tiles;
        int h = rem % num_heads;
        int b = rem / num_heads;
        
        // Process this attention tile
        // ... (normal FlashAttention logic)
    }
}

// Launch with fixed CTA count
dim3 grid(num_persistent_ctas);  // e.g., 4*SM = 232
persistent_attention<<<grid, block>>>(Q, K, V, O, work_counter, total_work);
```

### **Launch Bounds (All Iterations)**

```cpp
// Force 384 threads/block, allow ≥2 CTAs/SM
__launch_bounds__(384, 2)
__global__ void flash_attention(...) {
    // kernel code
}
```

---

## ⚠️ Critical Constraints

### **Mandatory Checks (Before Each Iteration)**

1. **GPU Readiness**
   ```bash
   nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
   python -c "import torch; assert torch.cuda.is_available()"
   ```

2. **CTA Count Assertion**
   ```python
   # In harness after benchmark
   assert result['ctas'] >= 4 * result['sm_count'], \
       f"Grid too small: {result['ctas']} CTAs < 4×{result['sm_count']} SMs"
   ```

3. **Correctness (Every Iteration)**
   ```bash
   python tests/test_basic.py  # All must pass
   ```

4. **Memory Safety (Every 5 Iterations)**
   ```bash
   compute-sanitizer --tool=memcheck python benches/bench_correctness_and_speed.py
   ```

### **Timeouts (Prevent Hangs)**
- Build: 120 seconds
- Benchmark: 300 seconds
- Profile: 300 seconds
- Total session: 60 minutes

### **Auto-Revert Policy**
If speedup **regresses >2%**:
```bash
git restore -SW :/ && git clean -fd
# Log regression, try different approach
```

### **One Change Per Iteration**
- ✅ ONE optimization focus
- ✅ Clear hypothesis documented
- ✅ Measurable improvement expected
- ❌ NO combined changes

---

## 📈 Expected Progression

```
Iteration 0 (Baseline):
  CTAs: 2, SM util: 3%, Latency: 0.579ms, Speedup: 0.07x
  
Iteration 1 (KV-splits):
  CTAs: 256, SM util: 65%, Latency: 0.095ms, Speedup: 0.45x
  Gain: 6× improvement! 🎉
  
Iteration 2 (Persistent):
  CTAs: 232, SM util: 70%, Latency: 0.085ms, Speedup: 0.51x
  Gain: +13% improvement
  
Iteration 3 (WMMA):
  CTAs: 232, SM util: 75%, Latency: 0.042ms, Speedup: 1.02x
  Gain: 2× improvement! 🎉
  
Iteration 4 (cp.async):
  CTAs: 232, SM util: 78%, Latency: 0.035ms, Speedup: 1.23x
  Gain: +20% improvement
  
Iterations 5-10 (Memory opts):
  Latency: 0.035ms → 0.028ms, Speedup: 1.23x → 1.54x
  Gain: +25% cumulative
  
🎯 TARGET ACHIEVED at Iteration 10!
```

---

## 🎓 L4-Specific Guidance

**Architecture**: Ada Lovelace, SM_89
- ✅ FP16 Tensor Cores (WMMA)
- ✅ BF16 support (emulated TC)
- ✅ cp.async for async copies
- ❌ No WGMMA (Hopper only)
- ❌ No TMA (Hopper only)
- ❌ No FP8 (Hopper only)

**Memory**: 300 GB/s bandwidth
- Primary bottleneck for attention
- Focus on coalescing and L2 utilization
- Async copy helps overlap

**Compute**: 242 TFLOPS FP16
- Secondary consideration after memory
- WMMA gives 2-4× gain over SIMT

---

## 📝 Iteration Log Template

```markdown
### Iteration {N} - {Timestamp}

**Phase**: {Parallelism/Memory/Compute/Tuning}

**Hypothesis**: {What bottleneck are you fixing?}

**Profiling Results**:
- CTAs: {count} (target: ≥232)
- SM utilization: {percent}%
- Memory throughput: {percent}%
- Bottleneck: {memory/compute/parallelism}

**Change Made**:
```cpp
// File: {filename}
// Lines: {range}
// Description: {what changed}
{code snippet}
```

**Build**: {✅ Success / ❌ Failed}
**Tests**: {✅ Passed / ❌ Failed}
**Sanitizer**: {✅ Clean / ⚠️ Warnings / ❌ Errors}

**Results**:
- Latency: {old_ms}ms → {new_ms}ms ({change}%)
- Speedup: {old_x}x → {new_x}x
- CTAs: {old_count} → {new_count}
- SM util: {old}% → {new}%

**Decision**: {✅ KEEP (commit) / 🔄 REVERT / 🎯 TARGET HIT}

**Next Hypothesis**: {If continuing, what to try next}
```

---

## 🚀 Cursor Agent Instructions

**For Cursor to execute this mission:**

```
You are an expert CUDA kernel optimization engineer.

CRITICAL: This is a PARALLELISM-FIRST mission.

Current problem: Kernel launches only 2 CTAs on 58-SM GPU (3% utilization).
No micro-optimization helps until you fix grid decomposition.

Follow this file: AGENTIC_OPTIMIZATION_MISSION.md

MANDATORY ITERATION ORDER:
1. Add KV-split parallelism (expect 6× gain)
2. Add persistent work queue
3. Enable WMMA tensor cores (expect 2-4× gain)
4. Add cp.async double-buffering
5-10. Memory optimizations
11-20. Compute optimizations

Use tools:
- python agentic_optimizer.py profile  (lightweight, fast)
- python agentic_optimizer.py build    (120s timeout)
- python agentic_optimizer.py benchmark (emits JSON)
- python agentic_optimizer.py test     (correctness)
- python agentic_optimizer.py evaluate {speedup}

Target: ≥1.5x speedup, ≥232 CTAs, >60% SM utilization

START with Iteration 1 (KV-splits) NOW.
Work autonomously. Show results after each iteration.
```

---

## 📚 References

**Expert Analysis**: See companion documents
- Parallelism strategy (Document 22)
- Profiling improvements (Document 23)

**CUDA Documentation**:
- [WMMA Programming Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma)
- [Async Copy (cp.async)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)
- [Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)

**Architecture**:
- L4 GPU: SM_89, Ada Lovelace, 58 SMs
- Thread blocks: Target 232+ for good utilization

---

**Last Updated**: October 12, 2025 (Expert Validated)
**Status**: PRODUCTION READY
**Priority**: PARALLELISM FIRST!
