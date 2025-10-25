# **EvoEngineer Integration: Complete CI-Driven Loop**

**Status**: ✅ **Infrastructure Complete** - Ready for Phase A Execution  
**Date**: Oct 17, 2025  
**Hardware**: NVIDIA L4 (Ada, sm_89)

---

## **Definition of Success** (Hard Gate)

```
✅ Target Shape: B=1, H=8, S=512, D=64 (L4-specialized)
✅ Metric: Median latency over 100 runs (warmup 20, seed 42)
✅ Correctness: max |Δ| ≤ 2e-3 vs SDPA reference
✅ Pass Rule: median(candidate) < 0.95 × median(SDPA)  (5% faster)
✅ Stats: 10k bootstrap CI, 95% confidence (candidate - SDPA) < 0
✅ Scope: FP16 on L4 (Ada) with Tensor Cores enabled
```

---

## **Infrastructure Components**

### **1. SDPA Oracle** (`bench/sdpa_oracle.py`)

**Purpose**: Ground truth for correctness and performance

**Features**:
- Dual-backend: Flash vs Math SDPA selection
- Bootstrap CI: 10,000 samples, 95% confidence
- Correctness: max |Δ| ≤ 2e-3 tolerance
- Hard gate: median < 0.95× SDPA

**API**:
```python
from bench.sdpa_oracle import evaluate_candidate

results = evaluate_candidate(
    candidate_fn=lambda: my_kernel(q, k, v, scale),
    q, k, v, scale,
    sdpa_backend="flash",  # or "math"
    iters=100,
    warmup=20,
    speedup_threshold=0.95  # 5% faster required
)

# results["passed"] = True if all gates pass
```

**Usage**:
```bash
# Smoke test (SDPA vs SDPA, should pass ~1.0× speedup)
python bench/sdpa_oracle.py
```

---

### **2. IMPL Selector** (`csrc/impl_selector.h`)

**Purpose**: First-class implementation candidates for Evo loop

**Implementations**:
```cpp
enum class Impl {
    CUSTOM_V3,    // Custom FlashAttention (Phase 4: 839 μs)
    CUBLAS,       // cuBLAS for Q@K^T and P@V (Target: 400-500 μs)
    CUTENSOR,     // cuTENSOR for tensor operations
    WMMA,         // WMMA Tensor Cores manual (Target: 30-40 μs)
    HYBRID_QKT,   // cuBLAS Q@K^T + custom P@V
    HYBRID_PV     // Custom Q@K^T + cuBLAS P@V
};
```

**Usage**:
```bash
# Select implementation via environment
IMPL=wmma python bench/measure_candidate.py
IMPL=cublas python bench/measure_candidate.py
```

**Why**: EvoEngineer can A/B library vs custom per tile, enabling systematic exploration

---

### **3. Measurement Infrastructure**

#### **A. `bench/measure_sdpa.py`**

**Purpose**: Baseline SDPA performance

```bash
# Measure Flash backend
python bench/measure_sdpa.py --backend flash --out .ci/sdpa.json

# Measure Math backend
python bench/measure_sdpa.py --backend math --out .ci/sdpa_math.json

# Output: median_ms, min_ms, max_ms, samples
```

#### **B. `bench/measure_candidate.py`**

**Purpose**: Candidate kernel with Nsight Compute metrics

```bash
# Measure with NCU profiling
IMPL=wmma python bench/measure_candidate.py --ncu --out .ci/cand.json

# Output: median_ms + NCU metrics
#   - sm__pipe_tensor_cycles_active (Tensor Core %)
#   - sm__throughput (SM busy %)
#   - sm__warps_active (Occupancy %)
#   - dram__sectors_read/write (Memory traffic)
```

#### **C. `bench/gate.py`**

**Purpose**: Hard CI gate with bootstrap statistics

```bash
# Check if candidate beats SDPA
python bench/gate.py \
  --sdpa .ci/sdpa.json \
  --cand .ci/cand.json \
  --alpha 0.05 \
  --speedup 0.95

# Exit code 0 = PASS, 1 = FAIL
```

**Gates**:
1. **Performance**: `cand_median < 0.95 × sdpa_median`
2. **Bootstrap CI**: Upper bound of 95% CI must be < 0 (statistically faster)

---

### **4. Nsight Compute Integration**

**Metrics Fed to Fitness Function**:

| Metric | What It Measures | Target |
|--------|------------------|--------|
| `sm__pipe_tensor_cycles_active` | Tensor Core utilization | >60% (SDPA: ~60%) |
| `sm__throughput` | SM busy percentage | >70% |
| `sm__warps_active` | Occupancy | >50% |
| `dram__sectors_read/write` | Memory traffic | Minimize (compute-bound) |

**Usage**:
```bash
ncu --target-processes all \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__sectors_read.sum,dram__sectors_write.sum \
  python bench/measure_candidate.py --impl wmma
```

**Composite Score** (for Evo ranking):
```python
score = (
    1000 * (sdpa_ms / cand_ms) +        # Speedup (primary)
    2 * tensor_pct +                     # Tensor Core usage
    sm_busy -                            # SM utilization
    penalty(n_syncs, n_regs, n_spills)  # Penalties
)
```

---

### **5. CI Workflow** (`.github/workflows/evo_bench.yml`)

**Trigger**: `[bench]` in commit message or workflow_dispatch

**Jobs**:
1. Measure SDPA baseline (Flash backend)
2. Measure candidate (IMPL from env or input)
3. Run Nsight Compute profiling
4. Hard gate check (performance + CI)
5. Upload artifacts (JSON results)
6. Comment on PR with results table

**Self-Hosted Runner Tags**: `[self-hosted, gpu, cuda, l4]`

**Environment**:
```yaml
env:
  IMPL: wmma  # or custom_v3, cublas, etc
  TORCH_CUDA_ARCH_LIST: '8.9'  # L4 Ada
  CUDA_VISIBLE_DEVICES: '0'
```

**Example PR Comment**:
```
✅ PASSED EvoEngineer Benchmark Gate

IMPL: wmma
Shape: [1, 8, 512, 64]

| Metric | SDPA | Candidate | Speedup |
|--------|------|-----------|---------|
| Median | 0.0471 ms | 0.0380 ms | 1.239× |

Gate: Candidate must be < 0.95× SDPA
Target: < 0.0447 ms
Result: ✅ Gate passed!

Nsight Compute Metrics:
- sm__pipe_tensor_cycles_active: 62.4%
- sm__throughput: 74.1%
- sm__warps_active: 58.3%
```

---

## **EvoEngineer Modes**

### **Mode 1: EvoEngineer-Insight** (Recommended for Day-to-Day)

**Input**:
1. **Task context**: L4, FP16, S=512, D=64, target < 0.95× SDPA
2. **NCU insights**: TC active 12% → increase WMMA tile reuse
3. **Single best solution**: Current champion (diff + metrics)

**Prompt Template**:
```
Task: FlashAttention on L4 (Ada, sm_89), FP16, B=1 H=8 S=512 D=64
Target: median < 0.95× SDPA (currently 0.047 ms → target < 0.045 ms)
Correctness: max |Δ| ≤ 2e-3

Current Best (Phase 4, CUSTOM_V3):
  - Latency: 0.839 ms (17.8× slower than SDPA)
  - Correctness: 100% (on PyTorch 2.1.0, 19% on 2.5.0)
  - NCU: TC active 0%, warps 30.53%, DRAM 0.31%

NCU Insights:
  1. TC active = 0% → NO Tensor Core usage (PRIMARY BOTTLENECK)
  2. DRAM = 0.31% → Compute-bound, not memory-bound
  3. Warps = 30.53% → Moderate occupancy, room for improvement
  4. Effective throughput: 27 TFLOPS (vs 242 TFLOPS available)

Root Cause: Scalar FP16 operations (Q@K^T + P@V = 78% of runtime)

Fix Options:
  1. cuBLAS Q@K^T: Expected 839 → 400-500 μs (2× speedup)
  2. WMMA Q@K^T + P@V: Expected 400 → 150 μs (2.7× speedup)
  3. Full TC + warp spec: Expected 150 → 30-40 μs (3.8× speedup)

Rules:
  - Keep API & ABI stable
  - FP16 input/output, FP32 accumulation
  - WMMA 16×16×16 tiles (L4 Ada)
  - Double-buffer SMEM with cp.async
  - XOR swizzle for bank-conflict-free SMEM
  - Target ≤ 64 regs/thread, ≤ 48KB SMEM

Deliver:
  - Single diff patch (kernel changes only)
  - Build flags (if changed)
  - 1-paragraph rationale tied to NCU metrics
```

### **Mode 2: EvoEngineer-Full** (When Near Finish Line)

**Additional Input**:
- **Historical elites**: Top-3 solutions (diffs + metrics)
- **Elite set**: Maintain for anti-regression

**When to Use**: Once median < SDPA, optimizing tail latency and robustness

---

## **Tight Execution Loop** (Today)

### **Step 1: Restore Correctness** (Phase A, 4 hours)

**Goal**: Fix PyTorch 2.5.0 compatibility (19% → 100% correctness)

```bash
# On GPU instance
cd ~/periodicdent42
source ~/venv/bin/activate

# Test SDPA dual backends
python -c "
from bench.sdpa_oracle import sdpa_ref
import torch

q = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
k, v = q.clone(), q.clone()
scale = 1.0 / 64**0.5

# Flash backend
flash_out = sdpa_ref(q, k, v, scale, {'enable_flash': True, 'enable_math': False, 'enable_mem_efficient': False})

# Math backend
math_out = sdpa_ref(q, k, v, scale, {'enable_flash': False, 'enable_math': True, 'enable_mem_efficient': False})

print(f'Flash vs Math diff: {(flash_out - math_out).abs().max().item():.6f}')
"

# Expected: Small diff (~1e-4), use Flash as reference
```

**Task 1.1**: Test with PyTorch 2.1.0 (isolate version issue)
**Task 1.2**: Add numerical stability guards (clamp exponentials, NaN checks)
**Task 1.3**: Validate with SDPA oracle

```bash
# Use oracle for validation
python scripts/standalone_phase4_eval.py  # Should show 100% correctness
```

### **Step 2: Wire IMPL Switches** (1 hour)

**Goal**: Enable cuBLAS/WMMA as first-class candidates

```cpp
// cudadent42/bench/kernels/fa_phase3_wmma.cu

#include "csrc/impl_selector.h"

extern "C" __global__ void fa_kernel(/* ... */) {
    Impl impl = cudadent::get_impl_from_env();
    
    switch (impl) {
        case Impl::CUSTOM_V3:
            // Existing scalar Q@K^T + P@V
            break;
        case Impl::CUBLAS:
            // cuBLAS Q@K^T + P@V (to be implemented)
            break;
        case Impl::WMMA:
            // WMMA Q@K^T + P@V (to be implemented)
            break;
        // ...
    }
}
```

### **Step 3: Implement cuBLAS Path** (Phase B, 6 hours)

**Goal**: Light up Tensor Cores → 400-500 μs

```cuda
// cuBLAS Q@K^T (in kernel)
cublasGemmEx(
    cublas_handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    BLOCK_N, BLOCK_M, HEAD_DIM,
    &alpha,
    K_smem, CUDA_R_16F, HEAD_DIM,
    Q_smem, CUDA_R_16F, HEAD_DIM,
    &beta,
    S_tile, CUDA_R_32F, BLOCK_N,
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**Validate**:
```bash
IMPL=cublas python bench/measure_candidate.py --ncu --out .ci/cand_cublas.json
python bench/gate.py --sdpa .ci/sdpa.json --cand .ci/cand_cublas.json
```

**Expected**:
- Latency: 400-500 μs (still fails gate, but 2× improvement)
- NCU: `sm__pipe_tensor_cycles_active > 50%` (Tensor Cores active!)

### **Step 4: EvoEngineer-Insight Sweep** (2 hours)

**Goal**: Optimize cuBLAS config (tile sizes, algorithms)

```bash
# Run Evo sweep on cuBLAS variants
for ALGO in 0 1 2 3 4 5; do
    for TILE in "16,16" "32,32" "64,64"; do
        IMPL=cublas CUBLAS_ALGO=$ALGO BLOCK_M=$(echo $TILE | cut -d, -f1) \
        python bench/measure_candidate.py --out .ci/cand_${ALGO}_${TILE}.json
    done
done

# Rank by composite score
python bench/rank_variants.py .ci/cand_*.json
```

**Composite Score**:
```python
score = 1000 * (sdpa_ms / cand_ms) + 2 * tensor_pct + sm_busy
```

### **Step 5: Full TC Pipeline** (Phase C, 8 hours)

**Goal**: WMMA + warp specialization → 30-40 μs (BEAT SDPA!)

```cuda
// WMMA Q@K^T
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

wmma::load_matrix_sync(q_frag, Q_smem, HEAD_DIM);
wmma::load_matrix_sync(k_frag, K_smem, HEAD_DIM);
wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);

// Warp specialization (producer/consumer)
const int warp_id = threadIdx.x >> 5;
const bool is_prod = (warp_id < 2);

if (is_prod) {
    // cp.async K/V into SMEM
    __pipeline_memcpy_async(K_smem[stage], K_global, ...);
} else {
    // WMMA compute
    wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
}
```

**Validate**:
```bash
IMPL=wmma python bench/measure_candidate.py --ncu --out .ci/cand_wmma.json
python bench/gate.py --sdpa .ci/sdpa.json --cand .ci/cand_wmma.json

# Expected: ✅ GATE PASSED (median < 0.95× SDPA)
```

---

## **Why This Works**

### **1. EvoEngineer Framework Validated**

- **91 real CUDA kernels** tested in paper
- Strong speedups + high validity (exactly what we need vs SDPA)

### **2. FlashAttention-2 Architecture**

- FA2 uses Tensor Cores + warp specialization
- PyTorch SDPA uses same hardware path on L4
- Our WMMA/cuBLAS paths give Evo access to same primitives

### **3. Systematic Feedback Loop**

```
┌─────────────────────────────────────────────────┐
│ EvoEngineer Loop                                │
├─────────────────────────────────────────────────┤
│ 1. Generate variant (IMPL=wmma, tile config)    │
│ 2. Build + measure (bench/measure_candidate.py) │
│ 3. NCU profiling (Tensor Core %, SM busy, etc)  │
│ 4. Composite score (speedup + TC + SM - penalty)│
│ 5. Hard gate (< 0.95× SDPA, bootstrap CI)       │
│ 6. Keep top-K, mutate                           │
│ 7. Repeat until gate passes                     │
└─────────────────────────────────────────────────┘
```

### **4. CI-Driven Ratchet**

- Every commit with `[bench]` triggers full evaluation
- Prevents regressions (gate must pass to merge)
- PR comments show performance delta automatically
- Artifacts uploaded for offline analysis

---

## **Current Status**

✅ **Infrastructure Complete**:
- SDPA oracle with dual backends
- IMPL selector (6 variants)
- Measurement scripts (SDPA, candidate, gate)
- Nsight Compute integration
- CI workflow (GPU runner)

⚠️ **Blockers**:
- Phase 4 correctness: 19% on PyTorch 2.5.0 (was 100% on 2.1.0)
- No Tensor Core usage yet (0% TC active)

🎯 **Next Actions**:
1. **Phase A** (4h): Fix correctness → 100%
2. **Phase B** (6h): cuBLAS Q@K^T → 400-500 μs, 50%+ TC
3. **Phase C** (8h): WMMA + warp spec → 30-40 μs, **BEAT SDPA** ✅

**Total**: 18 hours to SDPA-superior performance with CI-driven loop

---

## **References**

- **EvoEngineer Paper**: Two-layer traverse (solution-guiding + prompt engineering)
- **FlashAttention-2**: Warp specialization, WMMA, double-buffering
- **KernelBench**: Fast+correct gate matters (many LLMs correct but slow)
- **Nsight Compute Docs**: Metric names, usage, interpretation
- **PyTorch SDPA Backends**: Flash vs Math selection for stable reference

---

**Ready to Execute**: Phase A (correctness) → Phase B (cuBLAS) → Phase C (WMMA) → **BEAT SDPA** ✅

