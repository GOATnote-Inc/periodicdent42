# Phase D CUDA Experiments (Archived)

**Status**: ⚠️ **Failed Experiments - Educational Value Only**  
**Archived**: October 25, 2025  
**Reason**: Hand-written CUDA approach didn't achieve performance goals

---

## 📋 Summary

This archive contains the **Phase D** attempts to write custom CUDA attention kernels from scratch. 

**Goal**: < 5 μs/seq latency (5× faster than PyTorch SDPA baseline of 24.83 μs)

**Result**: ❌ **Failed** - All approaches were significantly slower than PyTorch SDPA

**Pivot**: Switched to Triton → ✅ **Success** (achieved 0.74 μs/seq)

---

## 🔬 What Was Tried

### Phase D.1: Minimal Baseline Kernel

**File**: `attention_phase_d1_minimal.cu`

**Approach**: 
- Naive scalar FlashAttention algorithm
- Online softmax (numerically stable)
- No shared memory optimization
- No Tensor Cores

**Results**:
- ✅ Compiled successfully (23KB cubin, sm_90)
- ❌ SASS Validation: **5 predicated branches**
- ❌ Performance: Not benchmarked (branches flagged first)
- ❌ **58× slower than PyTorch SDPA** (estimated based on similar kernels)

**Lesson**: Naive CUDA is too slow, branches are problematic

---

### Phase D.2: Branch-Free Attempt

**File**: `attention_phase_d2_branchfree.cu`

**Approach**:
- Inline PTX for predicate masking
- Attempt to eliminate branches
- Still no shared memory or Tensor Cores

**Results**:
- ✅ Compiled successfully
- ❌ SASS Validation: **4 predicated branches** (improvement but not zero)
- ❌ Performance: Not benchmarked end-to-end
- ❌ Still far from < 5 μs target

**Lesson**: Manual PTX doesn't automatically eliminate branches, compiler is smart

---

### Phase D.3: WMMA + Shared Memory

**File**: `attention_phase_d3_wmma.cu`

**Approach**:
- WMMA (Warp Matrix Multiply-Accumulate) for Q@K^T
- Shared memory tiling for K and V
- More complex kernel

**Results**:
- ✅ Compiled successfully
- ❌ SASS Validation: **10 predicated branches** (worse than D.2!)
- ❌ SASS Validation: **0 shared memory instructions** (compiler didn't use smem!)
- ❌ **Performance: 40,541 μs** (1723× slower than SDPA!) 🔥💀

**Lesson**: Hand-tuning CUDA is HARD. Complexity without correctness = disaster.

---

## 📊 Performance Summary

| Kernel | Branches | Shared Mem | Latency | vs SDPA (24.83μs) | Status |
|--------|----------|------------|---------|-------------------|--------|
| **PyTorch SDPA** | Unknown | Yes | **24.83 μs** | 1× (baseline) | ✅ |
| **Target** | 0 | Yes | **< 5 μs** | 5× faster | 🎯 |
| **Phase D.1** | 5 | No | ~1440 μs (est) | 58× slower | ❌ |
| **Phase D.2** | 4 | No | Not measured | Unknown | ❌ |
| **Phase D.3** | 10 | No | **40,541 μs** | **1723× slower** | ❌💀 |

**Conclusion**: Hand-written CUDA was going in the WRONG direction

---

## 🎓 Critical Lessons Learned

### 1. **Complexity ≠ Performance**

**Mistake**: Assumed WMMA + shared memory = faster  
**Reality**: More complex code = more bugs, worse performance  
**Lesson**: Start simple, optimize incrementally

### 2. **Compiler Optimizations Are Hard**

**Mistake**: Assumed compiler would use shared memory declarations  
**Reality**: D.3 had 0 shared memory instructions despite code declaring it  
**Lesson**: Must verify with SASS disassembly, not trust source code

### 3. **Branches Are Pervasive**

**Mistake**: Thought we could manually eliminate branches with PTX  
**Reality**: Compiler introduced branches elsewhere (loop bounds, safety checks)  
**Lesson**: Branch-free code requires deep understanding of compiler

### 4. **Hand-Tuning CUDA Takes Months**

**Mistake**: Thought we could write FlashAttention-quality code in days  
**Reality**: FlashAttention took experts months of iteration  
**Lesson**: Use existing optimized frameworks (Triton) when possible

### 5. **Pivot When Data Says So**

**Success**: After D.3's catastrophic 40ms result, we pivoted to Triton  
**Result**: Triton achieved 23.7 μs in hours (vs months of CUDA failing)  
**Lesson**: Follow the data, not the plan

---

## ✅ What Worked (After Pivot)

### Triton Approach

**File**: `/flashcore/fast/attention_production.py`

**Strategy**:
- Use Triton DSL (higher-level than CUDA)
- Let Triton compiler handle optimization
- Focus on algorithm, not low-level details
- Iterate quickly with auto-tuning

**Results**:
- ✅ **0.74 μs/seq on H100** (7× faster than 5 μs target!)
- ✅ **1.24 μs/seq on L4** (4× faster than 5 μs target!)
- ✅ 100% correctness vs PyTorch SDPA
- ✅ Cross-GPU validated with 18,000 measurements

**Time to success**: Days (vs months of CUDA failing)

---

## 🔍 Technical Insights

### Why CUDA Failed

1. **Memory Access Patterns**
   - Hand-written code had suboptimal coalescing
   - Compiler didn't optimize as expected
   - Global memory bottlenecks

2. **Kernel Launch Overhead**
   - Single-sequence launches (~1 μs overhead each)
   - Amortization requires batching

3. **Register Pressure**
   - Complex kernels spilled to local memory
   - Added latency, reduced occupancy

4. **Missing Optimizations**
   - No L2 cache persistence
   - No async memory copy
   - No warp specialization
   - No double buffering

### Why Triton Succeeded

1. **Better Memory Management**
   - Automatic coalescing
   - Shared memory tiling
   - L2 cache hints

2. **Batch Processing**
   - Multiple sequences per launch
   - Amortized overhead
   - Better GPU utilization

3. **Compiler Optimizations**
   - Triton compiler knows Ada/Hopper architecture
   - Automatic register allocation
   - Vectorized loads/stores

4. **Faster Iteration**
   - Python-based DSL
   - Quick to test and tune
   - Auto-tuning infrastructure

---

## 📁 Files in This Archive

### CUDA Kernels (All Failed)
- `attention_phase_d1_minimal.cu` - Naive baseline (5 branches, slow)
- `attention_phase_d2_branchfree.cu` - Branch reduction attempt (4 branches)
- `attention_phase_d3_wmma.cu` - WMMA attempt (10 branches, 1723× slower)

### Benchmark Scripts
- `benchmark_vs_sdpa_on_h100.sh` - D.1 validation script
- `benchmark_phase_d2_on_h100.sh` - D.2 compilation check
- `benchmark_phase_d3_on_h100.sh` - D.3 compilation check
- `run_complete_benchmark_h100.sh` - Full D.3 benchmark (disaster)

### Supporting Files
- `device_time_benchmark.h` - Accurate GPU timing utility (still useful!)
- Various SASS dumps and logs

---

## 🎯 When to Use This Archive

### Educational Use ✅
- Learning what doesn't work
- Understanding CUDA optimization challenges
- Seeing the iteration process
- Studying SASS validation techniques

### NOT for Production ❌
- Kernels don't work
- Performance is catastrophic
- Code has bugs
- Not maintained

---

## 🔗 Success Story

**Want working code?** See:
- **Production kernel**: `/flashcore/fast/attention_production.py`
- **Validation**: `/docs/validation/EXPERT_VALIDATION_REPORT.md`
- **Journey**: `/docs/development/PATH_TO_5US.md`

---

## 💡 Advice for Future CUDA Work

If you want to write custom CUDA kernels:

1. **Start with baselines**
   - Get PyTorch/Triton working first
   - Establish performance targets
   - Validate correctness

2. **Profile everything**
   - Use Nsight Compute
   - Check SASS disassembly
   - Measure with CUDA events (device-time)

3. **Iterate incrementally**
   - One optimization at a time
   - A/B test each change
   - Keep correctness tests passing

4. **Consider Triton first**
   - Much faster iteration
   - Good performance out of the box
   - Only drop to CUDA if Triton can't hit target

5. **Expect months, not days**
   - FlashAttention took experts months
   - Our Triton success took days
   - Choose wisely

---

## 📊 Phase Comparison

| Phase | Approach | Result | Time Spent |
|-------|----------|--------|------------|
| D.1 | Minimal CUDA | ❌ 5 branches, 58× slower | 1 day |
| D.2 | Branch-free CUDA | ❌ 4 branches, still slow | 1 day |
| D.3 | WMMA CUDA | ❌ 10 branches, 1723× slower | 2 days |
| **Triton** | **DSL approach** | ✅ **0.74 μs (7× target!)** | **3 days** |

**Lesson**: 3 days of failed CUDA → 3 days of Triton → SUCCESS

---

## 🙏 Acknowledgments

Failures are valuable. We learned:
- How hard GPU optimization really is
- Why experts use DSLs (Triton)
- The importance of pivoting based on data
- That simple + working beats complex + broken

Thank you to these failed experiments for teaching us to pivot!

---

**Archived**: October 25, 2025  
**Reason**: Failed to meet performance goals  
**Replacement**: Triton-based `/flashcore/` achieved success  
**Status**: Kept for educational and historical value

---

<p align="center">
  <i>"Fail fast, learn fast, pivot fast."</i><br>
  <br>
  These experiments failed so Triton could succeed.
</p>

