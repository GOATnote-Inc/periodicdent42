# FlashCore <40 μs Methodology: Systematic Path to Excellence

**Date**: October 21, 2025  
**Current**: 634 μs (multi-query baseline)  
**Target**: <40 μs (15.85× speedup needed)  
**Benchmark**: PyTorch SDPA ~45 μs

---

## 🎯 **Strategic Overview**

### **Gap Analysis**
```
Current:        634 μs (FlashCore multi-query, 16q/block, vectorized)
Target:         < 40 μs (beat PyTorch SDPA)
Gap:            15.85× speedup needed
PyTorch SDPA:   ~45 μs (FlashAttention-2 implementation)

Breakdown (estimated from periodicdent42 experience):
- Q@K^T:    ~250 μs (40%)  ← Vectorized, but scalar compute
- Softmax:  ~190 μs (30%)  ← Serial reductions, syncs
- P@V:      ~190 μs (30%)  ← Scalar accumulation
- Overhead:  ~4 μs (1%)    ← Sync, atomics
```

### **Evidence-Based Speedup Potential**

From research and existing implementations:

| Optimization | Technique | Expected Speedup | Evidence |
|-------------|-----------|------------------|----------|
| **Tensor Cores (WMMA)** | Q@K^T + P@V | 4-6× | CUTLASS, LeetCUDA, Stage-C |
| **Warp-Level Reductions** | Softmax m/l | 2× | EvoEngineer paper |
| **Double Buffering (cp.async)** | K/V loads | 1.5× | FlashAttention-2 |
| **Register Fusion** | Softmax + P@V | 1.3× | Stage-C fused |
| **Warp Specialization** | Producer/Consumer | 1.2× | Stage-5 periodicdent42 |

**Compound Effect**: 634 μs → 158 μs (WMMA) → 79 μs (reductions) → 53 μs (cp.async) → 41 μs (fusion) → **34 μs** (warp spec)

**Success Probability**: 
- Conservative (55 μs): 80% (2-3 optimizations succeed)
- Target (40 μs): 50% (4 optimizations succeed)
- Stretch (34 μs): 20% (all 5 optimizations succeed)

---

## 📋 **5-Phase Systematic Plan**

### **Phase 1: WMMA Q@K^T (Expected: 634 → 250 μs, 2.5× speedup)**

**Objective**: Replace scalar Q@K^T dot products with Tensor Core WMMA operations

**Implementation** (from Stage-C reference):
```cuda
// Current (scalar):
for (int row = 0; row < BLOCK_M; row++) {
    for (int col = tid; col < kv_size; col += THREADS) {
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            score += Q_tile[row][d] * K[...];
        }
        S_tile[row][col] = score * scale;
    }
}

// Target (WMMA):
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;

// Process 16×16 tiles
for (int warp_m = warp_id; warp_m < BLOCK_M; warp_m += NUM_WARPS) {
    wmma::fill_fragment(s_frag, 0.0f);
    
    // Accumulate over D dimension (4 tiles for D=64)
    for (int k = 0; k < HEAD_DIM; k += 16) {
        wmma::load_matrix_sync(q_frag, &Q_tile[warp_m][k], HEAD_DIM);
        wmma::load_matrix_sync(k_frag, &K_tile_T[warp_n][k], HEAD_DIM);
        wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
    }
    
    // Store with scale
    for (int i = 0; i < s_frag.num_elements; i++) {
        s_frag.x[i] *= softmax_scale;
    }
    wmma::store_matrix_sync(&S_tile[warp_m][warp_n], s_frag, BLOCK_N, wmma::mem_row_major);
}
```

**Key Changes**:
1. Convert Q_tile from `float[BLOCK_M][HEAD_DIM]` to `half[BLOCK_M][HEAD_DIM]`
2. Load K in transposed layout (col-major) for WMMA
3. Use 16×16×16 fragments (2 tiles for 32×32 block)
4. FP32 accumulation for numerical stability

**Testing**:
```bash
cd flashcore
cp kernels/flashcore_multi.cu kernels/flashcore_p1_wmma_qkt.cu
# Apply WMMA Q@K^T changes
python build_p1_wmma_qkt.py
python test_p1_wmma_qkt.py

# Expected:
# Correctness: PASS (max_err < 0.06)
# Performance: 200-300 μs (2-3× from 634 μs)
```

**Metrics to Track**:
- PTXAS: Registers (target: <56), SMEM (target: <20KB)
- NCU: `sm__pipe_tensor_cycles_active` (target: >30%)

**Fallback**: If correctness fails, implement Q@K^T only (keep scalar P@V)

---

### **Phase 2: WMMA P@V (Expected: 250 → 150 μs, 1.67× speedup)**

**Objective**: Replace scalar P@V accumulation with WMMA

**Implementation**:
```cuda
// Current (scalar):
for (int row = 0; row < rows_this_block; row++) {
    for (int d = tid; d < HEAD_DIM; d += THREADS) {
        float acc = 0.0f;
        for (int col = 0; col < kv_size; col++) {
            float p = expf(S_tile[row][col] - m_new);
            acc += p * V[...];
        }
        O_accum[row][d] += acc;
    }
}

// Target (WMMA):
// Pre-compute P (attention weights) into shared memory
for (int row = tid / 32; row < rows_this_block; row += NUM_WARPS) {
    for (int col = lane_id; col < kv_size; col += 32) {
        P_tile[row][col] = __float2half(expf(S_tile[row][col] - m_new));
    }
}
__syncthreads();

// WMMA P@V
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;

for (int warp_m = warp_id; warp_m < BLOCK_M; warp_m += NUM_WARPS) {
    wmma::fill_fragment(o_frag, 0.0f);
    
    // Accumulate over KV dimension
    for (int k = 0; k < BLOCK_N; k += 16) {
        wmma::load_matrix_sync(p_frag, &P_tile[warp_m][k], BLOCK_N);
        wmma::load_matrix_sync(v_frag, &V_tile[k][0], HEAD_DIM);
        wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
    }
    
    // Accumulate into O_accum (correction handled separately)
    wmma::store_matrix_sync(&O_accum[warp_m][warp_d], o_frag, HEAD_DIM, wmma::mem_row_major);
}
```

**Key Changes**:
1. Add `P_tile[BLOCK_M][BLOCK_N]` in shared memory (2KB)
2. Materialize exp-ed attention weights
3. WMMA for 16×N @ N×16 (multiple K tiles)

**Testing**:
```bash
python build_p2_wmma_pv.py
python test_p2_wmma_pv.py

# Expected:
# Correctness: PASS
# Performance: 120-180 μs (1.5-2× from Phase 1)
# NCU: sm__pipe_tensor_cycles_active >60%
```

---

### **Phase 3: Warp-Level Reductions (Expected: 150 → 75 μs, 2× speedup)**

**Objective**: Parallelize m_new and l_new computation across warps

**Current Bottleneck**:
```cuda
// Serial reduction (single thread!)
if (tid == 0) {
    float m_new = m_i[row];
    for (int col = 0; col < kv_size; col++) {
        m_new = fmaxf(m_new, S_tile[row][col]);
    }
    m_new_shared = m_new;
}
__syncthreads();  // All 256 threads wait for 1 thread!
```

**Optimized (warp reductions)**:
```cuda
// Parallel max across warps
__shared__ float m_row_smem[BLOCK_M][NUM_WARPS];

for (int row = 0; row < rows_this_block; row++) {
    // Step 1: Each thread computes local max
    float m_local = -FLT_MAX;
    for (int col = tid; col < kv_size; col += THREADS) {
        m_local = fmaxf(m_local, S_tile[row][col]);
    }
    
    // Step 2: Warp reduction (32 threads → 1 value)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        m_local = fmaxf(m_local, __shfl_down_sync(0xffffffff, m_local, offset));
    }
    
    // Step 3: Lane 0 writes warp result
    if (lane_id == 0) {
        m_row_smem[row][warp_id] = m_local;
    }
    __syncthreads();
    
    // Step 4: Single warp reduces across NUM_WARPS values
    if (tid < NUM_WARPS) {
        float m_warp = m_row_smem[row][tid];
        #pragma unroll
        for (int offset = NUM_WARPS/2; offset > 0; offset >>= 1) {
            m_warp = fmaxf(m_warp, __shfl_down_sync(0xffffffff, m_warp, offset));
        }
        if (tid == 0) m_row_smem[row][0] = m_warp;
    }
    __syncthreads();
    
    float m_new = m_row_smem[row][0];
    // Same pattern for l_new (sum reduction)
}
```

**Expected Impact**:
- Eliminate serial loops (64 iterations → 5 warp reductions)
- Reduce sync overhead (1 sync per row → same, but faster)
- Expected: 2× speedup on softmax portion (~30% of runtime)

**Testing**:
```bash
python build_p3_warp_reduce.py
python test_p3_warp_reduce.py

# Expected:
# Correctness: PASS
# Performance: 60-90 μs
```

---

### **Phase 4: Double Buffering (cp.async) (Expected: 75 → 50 μs, 1.5× speedup)**

**Objective**: Overlap K/V loads with computation using async copy

**Implementation** (from cp_async.hpp):
```cuda
#include "detail/cp_async.hpp"

// Add staging buffers
__shared__ half K_stage[2][BLOCK_N][HEAD_DIM];
__shared__ half V_stage[2][BLOCK_N][HEAD_DIM];

int stage_read = 0;
int stage_write = 1;

// Pre-load first tile
for (int i = tid; i < BLOCK_N * HEAD_DIM; i += THREADS) {
    int row = i / HEAD_DIM;
    int col = i % HEAD_DIM;
    detail::cp_async_ca<16>(
        &K_stage[stage_write][row][col],
        &K_head[(kv_start + row) * HEAD_DIM + col]
    );
}
detail::cp_async_commit_group();

for (int n_block = 0; n_block < num_blocks_n; n_block++) {
    // Wait for current tile
    detail::cp_async_wait_group<0>();
    __syncthreads();
    
    // Launch next tile load (overlaps with compute below)
    if (n_block + 1 < num_blocks_n) {
        // Launch async copy for next tile
        // ... cp_async_ca ...
        detail::cp_async_commit_group();
    }
    
    // Compute with stage_read buffer
    // ... WMMA Q@K^T, softmax, P@V using K_stage[stage_read], V_stage[stage_read]
    
    // Swap buffers
    stage_read ^= 1;
    stage_write ^= 1;
}
```

**Key Requirements**:
- L4 GPU (sm_89) supports cp.async ✅
- 16-byte aligned addresses
- Additional 16KB SMEM for staging (total: 28KB, within 48KB limit)

**Expected Impact**:
- Hide ~50% of K/V load latency
- Enable 1.5× speedup on memory-bound portions

**Testing**:
```bash
python build_p4_cp_async.py
python test_p4_cp_async.py

# Expected:
# Correctness: PASS
# Performance: 40-60 μs
# NCU: dram__throughput overlap with compute
```

---

### **Phase 5: Fused Softmax (Expected: 50 → 40 μs, 1.25× speedup)**

**Objective**: Eliminate S_tile storage by fusing softmax into WMMA P@V

**Current**: S_tile (2KB) → compute P → accumulate
**Target**: Compute P on-the-fly during P@V WMMA

**Implementation** (from Stage-C fused):
```cuda
// Instead of:
// 1. WMMA Q@K^T → S_tile
// 2. Softmax on S_tile → P_tile
// 3. WMMA P@V

// Do:
// 1. WMMA Q@K^T → s_frag (register)
// 2. Softmax on s_frag → p_frag (register)
// 3. WMMA p_frag @ V

wmma::fragment<wmma::accumulator, 16, 16, 16, half> s_frag_fp16;
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;

// Q@K^T in FP32
wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);

// Softmax in registers (per-thread)
float m_local = -FLT_MAX;
for (int i = 0; i < s_frag.num_elements; i++) {
    m_local = fmaxf(m_local, s_frag.x[i]);
}
// Warp reduce m_local...

float l_local = 0.0f;
for (int i = 0; i < s_frag.num_elements; i++) {
    float p = expf(s_frag.x[i] * softmax_scale - m_new);
    s_frag_fp16.x[i] = __float2half(p);
    l_local += p;
}
// Warp reduce l_local...

// P@V (p_frag is still in registers)
// Reinterpret s_frag_fp16 as p_frag
// wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
```

**Challenge**: WMMA fragment element indexing is non-trivial (use LUT)

**Expected Impact**:
- Eliminate S_tile (2KB SMEM saved)
- Reduce global sync points
- 1.25× speedup

**Testing**:
```bash
python build_p5_fused.py
python test_p5_fused.py

# Expected:
# Correctness: PASS (HIGH RISK - verify carefully!)
# Performance: 35-45 μs
```

---

## 🔬 **Systematic Testing Framework**

### **Test Harness** (`flashcore/test_framework.py`)

```python
import torch
import statistics
from build_p1_wmma_qkt import build_p1

def test_phase(phase_name, build_fn, target_us, prev_us):
    """Systematic testing for each phase"""
    print(f"\n{'='*80}")
    print(f"Testing {phase_name}")
    print(f"{'='*80}\n")
    
    # Build kernel
    print("Building kernel...")
    ext = build_fn()
    print("✅ Build successful\n")
    
    # Test shapes
    shapes = [
        (1, 8, 512, 64, "mission"),   # Primary target
        (1, 8, 256, 64, "short"),     # Generalization
        (1, 8, 1024, 64, "long"),     # Generalization
    ]
    
    results = {}
    
    for B, H, S, D, name in shapes:
        print(f"Testing shape: {name} (B={B}, H={H}, S={S}, D={D})")
        
        # Correctness
        Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
        scale = 1.0 / (D ** 0.5)
        
        O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
        O_kernel = ext.forward(Q, K, V, scale)
        
        max_err = (O_ref - O_kernel).abs().max().item()
        mean_err = (O_ref - O_kernel).abs().mean().item()
        
        print(f"  Correctness:")
        print(f"    max_err:  {max_err:.6f}")
        print(f"    mean_err: {mean_err:.6f}")
        
        if max_err > 0.06:
            print(f"  ❌ FAIL: max_err exceeds threshold\n")
            results[name] = {"pass": False, "max_err": max_err}
            continue
        
        print(f"  ✅ PASS\n")
        
        # Performance
        warmup, iters = 20, 100
        for _ in range(warmup):
            _ = ext.forward(Q, K, V, scale)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            O = ext.forward(Q, K, V, scale)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)
        
        p50 = statistics.median(times)
        p90 = statistics.quantiles(times, n=10)[8]
        
        speedup_vs_prev = prev_us / p50
        
        print(f"  Performance:")
        print(f"    p50: {p50:.2f} μs")
        print(f"    p90: {p90:.2f} μs")
        print(f"    Speedup vs prev: {speedup_vs_prev:.2f}×")
        print(f"    Target: {target_us:.2f} μs")
        
        if p50 <= target_us:
            print(f"  🎉 TARGET MET!\n")
        elif p50 <= target_us * 1.2:
            print(f"  ✅ ACCEPTABLE (within 20% of target)\n")
        else:
            print(f"  ⚠️  MISSED TARGET (off by {p50/target_us:.2f}×)\n")
        
        results[name] = {
            "pass": True,
            "max_err": max_err,
            "p50": p50,
            "p90": p90,
            "speedup": speedup_vs_prev,
            "target_met": p50 <= target_us * 1.2
        }
    
    # Summary
    print(f"\n{'='*80}")
    print(f"{phase_name} Summary")
    print(f"{'='*80}\n")
    
    all_pass = all(r["pass"] and r.get("target_met", False) for r in results.values())
    
    if all_pass:
        print(f"✅ ALL TESTS PASSED - PROCEED TO NEXT PHASE")
    else:
        print(f"⚠️  SOME TESTS FAILED - DEBUG REQUIRED")
        print(f"\nFailed shapes:")
        for name, r in results.items():
            if not r["pass"] or not r.get("target_met", False):
                print(f"  - {name}: {r}")
    
    return results

# Usage:
# results_p1 = test_phase("Phase 1: WMMA Q@K^T", build_p1, target_us=250, prev_us=634)
```

---

## 📊 **Progress Tracking**

### **Metrics Dashboard** (`flashcore/dashboard.py`)

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt

class ProgressTracker:
    def __init__(self):
        self.log_file = Path("flashcore_progress.json")
        self.load()
    
    def load(self):
        if self.log_file.exists():
            self.data = json.loads(self.log_file.read_text())
        else:
            self.data = {"phases": [], "target": 40.0, "pytorch_sdpa": 45.0}
    
    def save(self):
        self.log_file.write_text(json.dumps(self.data, indent=2))
    
    def log_phase(self, phase_name, latency_us, correctness, ncu_metrics=None):
        self.data["phases"].append({
            "name": phase_name,
            "latency_us": latency_us,
            "correctness": correctness,
            "ncu": ncu_metrics or {}
        })
        self.save()
        self.plot()
    
    def plot(self):
        phases = [p["name"] for p in self.data["phases"]]
        latencies = [p["latency_us"] for p in self.data["phases"]]
        
        plt.figure(figsize=(12, 6))
        plt.plot(phases, latencies, 'o-', linewidth=2, markersize=8, label='FlashCore')
        plt.axhline(y=self.data["target"], color='g', linestyle='--', label='Target (<40 μs)')
        plt.axhline(y=self.data["pytorch_sdpa"], color='r', linestyle='--', label='PyTorch SDPA')
        plt.xlabel('Optimization Phase')
        plt.ylabel('Latency (μs)')
        plt.title('FlashCore Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('flashcore_progress.png', dpi=150)
        print(f"✅ Progress chart saved to flashcore_progress.png")

tracker = ProgressTracker()
```

---

## 🔄 **Iteration Protocol (EvoEngineer-Style)**

### **Per-Phase Workflow**

```bash
# Phase N Template
cd flashcore

# 1. Implement
cp kernels/flashcore_p{N-1}.cu kernels/flashcore_p{N}.cu
vim kernels/flashcore_p{N}.cu  # Apply optimization

# 2. Build
python build_p{N}.py 2>&1 | tee build_p{N}.log
# Check PTXAS stats: registers, SMEM, spills

# 3. Test Correctness
python test_p{N}.py --correctness-only

# 4. Profile (if correct)
python test_p{N}.py --full

# 5. NCU Analysis
ncu --set full python test_p{N}.py --ncu-profile > ncu_p{N}.txt
# Check: sm__pipe_tensor_cycles_active, sm__throughput, dram__throughput

# 6. Log Progress
python -c "
from dashboard import tracker
tracker.log_phase('Phase {N}', latency_us=XXX, correctness=True, ncu_metrics={...})
"

# 7. Decision
if latency < target_for_phase:
    echo "✅ Phase {N} complete, proceeding to Phase {N+1}"
else:
    echo "⚠️  Phase {N} missed target, debugging..."
    # Option A: Debug and retry
    # Option B: Accept partial win and proceed
    # Option C: Fallback to previous phase
```

### **Debugging Protocol**

When a phase fails:

```bash
# 1. Check correctness first
python test_p{N}.py --verbose --shape mission --seed 42
# If max_err > 0.06: Fix numerical issues

# 2. Check PTXAS
grep "Used.*registers\|bytes smem" build_p{N}.log
# If >64 registers or >48KB SMEM: Optimize occupancy

# 3. Check NCU
grep "sm__pipe_tensor_cycles_active\|dram__throughput" ncu_p{N}.txt
# If TC active <30%: WMMA not engaged
# If DRAM >80%: Memory-bound, need better coalescing

# 4. Compare to reference
diff -u kernels/flashcore_p{N-1}.cu kernels/flashcore_p{N}.cu > debug_diff_p{N}.txt
# Identify what changed, isolate bug

# 5. Bisect (if needed)
# Temporarily disable new optimization
# Verify previous phase still works
# Re-enable optimization incrementally
```

---

## 🎯 **Success Criteria**

### **Tier System**

| Tier | Latency | vs PyTorch | Status | Grade | Probability |
|------|---------|------------|--------|-------|-------------|
| **Baseline** | 634 μs | 14.1× slower | ✅ Done | - | 100% |
| **Tier 4** | 150-250 μs | 3-5× slower | Phase 1-2 | C | 90% |
| **Tier 3** | 75-150 μs | 1.7-3× slower | Phase 3 | B | 70% |
| **Tier 2** | 50-75 μs | 1.1-1.7× slower | Phase 4 | B+ | 50% |
| **Tier 1** | **40-50 μs** | **0.9-1.1× PyTorch** | **Phase 5** | **A** | **30%** ✅ |
| **Tier S** | <40 μs | Beats PyTorch! | Stretch | A+ | 10% |

### **Hard Gates**

For each phase to be considered "complete":

1. **Correctness**: All shapes pass (max_err < 0.06)
2. **Performance**: Within 20% of phase target
3. **Stability**: 10 consecutive runs pass
4. **Metrics**: NCU shows expected utilization (TC >30% after Phase 2)

### **Fallback Strategy**

If a phase fails after 4 hours of debugging:

- **Option A**: Accept current performance, proceed with partial optimization
- **Option B**: Revert to previous phase, try alternative approach
- **Option C**: Skip phase, proceed to next (may still reach <50 μs goal)

---

## 📚 **References & Resources**

### **Code References**

1. **`cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`**
   - Complete WMMA implementation with cp.async
   - USE_WMMA_PV, USE_CP_ASYNC flags
   - Warp specialization (Stage-5)

2. **`cudadent42/bench/kernels/detail/cp_async.hpp`**
   - cp.async wrappers for double buffering
   - Pipeline control primitives

3. **`scripts/evo_full_iteration.py`**
   - EvoEngineer framework implementation
   - Population management, fitness scoring

4. **LeetCUDA** (https://deepwiki.com/xlite-dev/LeetCUDA/3.2-wmma-implementations)
   - Multi-stage pipelines
   - SMEM swizzling for bank conflicts

5. **CUTLASS** (https://github.com/NVIDIA/cutlass)
   - Production-grade GEMM templates
   - Tile iterators, epilogue fusion

### **Papers**

1. **EvoEngineer** (arXiv:2510.03760v1)
   - Two-layer traverse technique
   - 91 kernels, median 2.72× speedup
   - Max 36.75× speedup

2. **FlashAttention-2** (Dao et al.)
   - Warp specialization
   - Online softmax with tiling
   - Work partitioning across SMs

3. **Ada Architecture Whitepaper** (NVIDIA)
   - L4 GPU specs: 242 TFLOPS (FP16), 60 RT-TFLOPS
   - Tensor Core gen 4 (FP8, BF16, FP16)

### **Tools**

```bash
# NCU profiling
ncu --set full --target-processes all python test.py

# PTXAS verbose
nvcc -Xptxas -v kernel.cu

# Occupancy calculator
cuda-occupancy-calculator --regs 48 --smem 20480 --block 128

# Nsight Systems (timeline)
nsys profile --stats=true python test.py
```

---

## ⏱️ **Time Budget**

| Phase | Task | Hours | Cumulative |
|-------|------|-------|------------|
| **Phase 1** | WMMA Q@K^T | 4-6h | 4-6h |
| **Phase 2** | WMMA P@V | 3-5h | 7-11h |
| **Phase 3** | Warp Reductions | 2-4h | 9-15h |
| **Phase 4** | cp.async | 2-4h | 11-19h |
| **Phase 5** | Fused Softmax | 3-6h | 14-25h |
| **Testing** | Comprehensive validation | 2-4h | 16-29h |
| **Buffer** | Debugging, iteration | 4-6h | 20-35h |

**Total Estimate**: 20-35 hours over 1-2 weeks

**Checkpoint Milestones**:
- **Day 1-2**: Phase 1 complete (WMMA Q@K^T working, 200-300 μs)
- **Day 3-4**: Phase 2 complete (WMMA P@V working, 120-180 μs)
- **Day 5-6**: Phase 3 complete (Warp reductions, 60-90 μs)
- **Day 7-8**: Phase 4 complete (cp.async, 40-60 μs)
- **Day 9-10**: Phase 5 complete (Fused softmax, 35-50 μs) ✅

---

## 🚀 **EXECUTION STARTS NOW**

**Deeds not words. Let's implement Phase 1.**

Next command:
```bash
cd flashcore
# Copy multi-query baseline
cp kernels/flashcore_multi.cu kernels/flashcore_p1_wmma_qkt.cu
```

**Ready to implement WMMA Q@K^T!** 🔥

