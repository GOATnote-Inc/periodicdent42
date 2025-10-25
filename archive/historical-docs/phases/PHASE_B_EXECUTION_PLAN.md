# **Phase B Execution Plan: Tensor Core Q@K^T via cuBLAS**

**Goal**: Replace scalar Q@K^T with cuBLAS Tensor Cores → 400-500 μs (2× speedup)  
**Time**: 6 hours  
**Approach**: Systematic TDD with NCU validation at each step  
**Confidence**: 90% (cuBLAS proven at 5.49 μs/tile baseline)

---

## **Current Baseline** (Phase A Complete)

```
✅ Phase 4: 870.49 μs (100% correct on PyTorch 2.1.0)
✅ PyTorch SDPA: 49.73 μs (target)
✅ Gap: 17.5× slower than SDPA
```

**Runtime Breakdown** (from NCU analysis):
```
Q@K^T (Scalar):  350 μs (40%)  ← TARGET FOR PHASE B
P@V (Scalar):    300 μs (35%)  ← Phase C
Softmax:         120 μs (14%)  ← Already optimized
Sync:            100 μs (11%)  ← Phase C (warp-sync)
```

---

## **Phase B Target**

```
After cuBLAS Q@K^T Integration:
  Expected: 400-500 μs
  Speedup: 1.7-2.1× vs Phase 4
  Gap to SDPA: 8.0-10.1× (reduced from 17.5×)
  NCU: 50-60% Tensor Core utilization ✅
```

---

## **Phase B.1: cuBLAS Single-Tile Test** ⏱️ 2 hours

**Goal**: Validate cuBLAS Tensor Core path for Q@K^T on single tile

### **TDD Steps**

#### **Test 1: Minimal cuBLAS GEMM** (30 min)
```cuda
// File: bench/test_cublas_minimal.cu
// Goal: Verify cuBLAS setup and basic operation

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>

int main() {
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    // Test 4×4×4 GEMM (minimal size)
    const int M=4, N=4, K=4;
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, M*K*sizeof(half));
    cudaMalloc(&d_B, K*N*sizeof(half));
    cudaMalloc(&d_C, M*N*sizeof(float));
    
    // Initialize to 1.0
    // ... (host data init) ...
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Launch cuBLAS GEMM
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_16F, N,
        d_A, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    // Verify result
    // Expected: All values = 4.0 (4×4×4 with 1.0 inputs)
    
    printf("✅ cuBLAS minimal test passed\n");
    return 0;
}
```

**Success Criteria**:
- ✅ Compiles without errors
- ✅ cuBLAS initializes successfully
- ✅ GEMM produces correct results
- ✅ No runtime errors

#### **Test 2: FlashAttention Tile Size (32×64×64)** (30 min)
```cuda
// File: bench/test_cublas_qkt_tile.cu
// Goal: Test actual Q@K^T tile size for FlashAttention

// Q: [32, 64] (BLOCK_M × HEAD_DIM)
// K^T: [64, 64] (HEAD_DIM × HEAD_DIM) 
// S = Q @ K^T: [32, 64] (BLOCK_M × HEAD_DIM)

const int BLOCK_M = 32;
const int HEAD_DIM = 64;
const int BLOCK_N = 64;  // KV tile size

// Allocate Q tile (32×64)
half *d_Q;
cudaMalloc(&d_Q, BLOCK_M * HEAD_DIM * sizeof(half));

// Allocate K tile (64×64, column-major for K^T)
half *d_K;
cudaMalloc(&d_K, BLOCK_N * HEAD_DIM * sizeof(half));

// Allocate S output (32×64)
float *d_S;
cudaMalloc(&d_S, BLOCK_M * BLOCK_N * sizeof(float));

// Launch cuBLAS for Q @ K^T
float alpha = scale;  // 1/sqrt(64) = 0.125
float beta = 0.0f;

cublasGemmEx(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,  // K is transposed
    BLOCK_N, BLOCK_M, HEAD_DIM,
    &alpha,
    d_K, CUDA_R_16F, HEAD_DIM,  // K: [64, 64] col-major
    d_Q, CUDA_R_16F, HEAD_DIM,  // Q: [32, 64] row-major
    &beta,
    d_S, CUDA_R_32F, BLOCK_N,   // S: [32, 64] row-major
    CUBLAS_COMPUTE_32F_FAST_16F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**Success Criteria**:
- ✅ Correct output shape (32×64)
- ✅ Numerically correct vs scalar implementation
- ✅ Performance: 5-10 μs/tile (vs ~30 μs scalar)
- ✅ NCU: Tensor Core utilization > 0%

#### **Test 3: Correctness Validation** (30 min)
```python
# File: scripts/test_cublas_qkt_correctness.py
import torch
import numpy as np

# Generate test data
Q = torch.randn(32, 64, dtype=torch.float16, device='cuda')
K = torch.randn(64, 64, dtype=torch.float16, device='cuda')
scale = 1.0 / 64**0.5

# Reference (PyTorch)
S_ref = (Q @ K.T) * scale

# cuBLAS test (via our C++ binding)
import cublas_qkt_test
S_cublas = cublas_qkt_test.forward(Q, K, scale)

# Compare
diff = (S_ref - S_cublas).abs().max().item()
print(f"Max diff: {diff:.6f}")

assert diff < 1e-3, f"Correctness failed: {diff}"
print("✅ Correctness test passed")
```

**Success Criteria**:
- ✅ max_diff < 1e-3 (matches tolerance)
- ✅ 100/100 random tests pass

#### **Test 4: Performance Benchmark** (30 min)
```python
# File: scripts/bench_cublas_qkt.py
import torch
import time

# Warmup
for _ in range(10):
    S = cublas_qkt_test.forward(Q, K, scale)

# Benchmark
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000):
    S = cublas_qkt_test.forward(Q, K, scale)
torch.cuda.synchronize()
t1 = time.perf_counter()

latency_us = (t1 - t0) * 1e6 / 1000
print(f"cuBLAS Q@K^T: {latency_us:.2f} μs/tile")

# Expected: 5-10 μs (vs ~30 μs scalar)
assert latency_us < 15, f"Performance regression: {latency_us} μs"
print("✅ Performance test passed")
```

**Success Criteria**:
- ✅ Latency: 5-10 μs/tile
- ✅ Speedup: 3-6× vs scalar (~30 μs)

---

## **Phase B.2: Integrate cuBLAS into Full Kernel** ⏱️ 2 hours

**Goal**: Replace scalar Q@K^T loop in Phase 4 with cuBLAS

### **TDD Steps**

#### **Test 1: Hybrid Kernel Structure** (45 min)
```cuda
// File: cudadent42/bench/kernels/fa_phase5_cublas.cu
// Hybrid: cuBLAS Q@K^T + scalar P@V + online softmax

__global__ void fa_phase5_kernel(
    const half* Q, const half* K, const half* V, half* O,
    cublasHandle_t cublas_handle,  // Pass handle
    int B, int H, int S, int D, float scale
) {
    // Load Q tile to SMEM (unchanged)
    __shared__ half Q_smem[BLOCK_M][HEAD_DIM];
    // ... existing Q load code ...
    
    // Initialize online softmax (unchanged)
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float O_acc[HEAD_DIM] = {0};
    
    // Iterate over KV tiles
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        // Load K tile to SMEM
        __shared__ half K_smem[BLOCK_N][HEAD_DIM];
        // ... existing K load code ...
        
        __syncthreads();
        
        // ========================================
        // NEW: cuBLAS Q@K^T (replaces scalar loop)
        // ========================================
        __shared__ float S_smem[BLOCK_M][BLOCK_N];
        
        if (threadIdx.x == 0) {  // Single thread launches cuBLAS
            cublasGemmEx(
                cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                BLOCK_N, BLOCK_M, HEAD_DIM,
                &scale,
                K_smem, CUDA_R_16F, HEAD_DIM,
                Q_smem, CUDA_R_16F, HEAD_DIM,
                &beta,
                S_smem, CUDA_R_32F, BLOCK_N,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
        }
        
        __syncthreads();
        
        // Online softmax (unchanged)
        // ... existing softmax code using S_smem ...
        
        // P@V (scalar, unchanged - optimize in Phase C)
        // ... existing P@V code ...
    }
    
    // Final normalization (unchanged)
    // ... existing code ...
}
```

**Success Criteria**:
- ✅ Compiles without errors
- ✅ cuBLAS handle passed correctly
- ✅ Single-threaded launch works

#### **Test 2: Correctness Validation** (45 min)
```python
# Test hybrid kernel vs Phase 4 reference
import fa_phase4  # Original (100% correct)
import fa_phase5_cublas  # New hybrid

Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device='cuda')
K, V = Q.clone(), Q.clone()
scale = 1.0 / 64**0.5

# Reference
O_ref = fa_phase4.forward(Q, K, V, scale)

# cuBLAS hybrid
O_cublas = fa_phase5_cublas.forward(Q, K, V, scale)

# Compare
diff = (O_ref - O_cublas).abs().max().item()
print(f"Max diff: {diff:.6f}")

assert diff < 2e-3, f"Correctness failed: {diff}"
print("✅ Hybrid kernel correctness passed")
```

**Success Criteria**:
- ✅ max_diff < 2e-3
- ✅ 100/100 random tests pass

#### **Test 3: Performance Measurement** (30 min)
```python
# Benchmark hybrid kernel
import time

# Warmup
for _ in range(10):
    O = fa_phase5_cublas.forward(Q, K, V, scale)

# Benchmark
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    O = fa_phase5_cublas.forward(Q, K, V, scale)
torch.cuda.synchronize()
t1 = time.perf_counter()

latency_us = (t1 - t0) * 1e6 / 100
print(f"Phase 5 (cuBLAS Q@K^T): {latency_us:.2f} μs")
print(f"Phase 4 (scalar): 870.49 μs")
print(f"Speedup: {870.49 / latency_us:.2f}×")

# Expected: 400-500 μs (1.7-2.1× speedup)
assert 400 < latency_us < 600, f"Unexpected performance: {latency_us} μs"
print("✅ Performance target achieved")
```

**Success Criteria**:
- ✅ Latency: 400-500 μs
- ✅ Speedup: 1.7-2.1× vs Phase 4

---

## **Phase B.3: Tune cuBLAS Config** ⏱️ 1 hour

**Goal**: Optimize cuBLAS algorithm and tile sizes

### **TDD Steps**

#### **Test 1: Algorithm Sweep** (30 min)
```python
# Test CUBLAS_GEMM_ALGO_0 through ALGO_15
algos = list(range(16))
results = []

for algo in algos:
    # Set algorithm in kernel
    os.environ['CUBLAS_ALGO'] = str(algo)
    
    # Rebuild
    build_phase5_cublas()
    import fa_phase5_cublas
    
    # Benchmark
    latency = benchmark(fa_phase5_cublas)
    
    # Correctness
    correct = check_correctness(fa_phase5_cublas)
    
    results.append({
        'algo': algo,
        'latency_us': latency,
        'correct': correct
    })
    
    if correct and latency < best_latency:
        best_algo = algo
        best_latency = latency

print(f"Best algorithm: ALGO_{best_algo} ({best_latency:.2f} μs)")
```

**Success Criteria**:
- ✅ Best algorithm identified
- ✅ 5-10% improvement possible

#### **Test 2: Tile Size Optimization** (30 min)
```python
# Test different BLOCK_M/BLOCK_N combinations
tile_configs = [
    (16, 64),  # Smaller M
    (32, 64),  # Current
    (64, 64),  # Larger M (if SMEM permits)
    (32, 32),  # Smaller N
]

for BLOCK_M, BLOCK_N in tile_configs:
    # Test SMEM usage
    smem_q = BLOCK_M * 64 * 2  # bytes
    smem_k = BLOCK_N * 64 * 2
    smem_s = BLOCK_M * BLOCK_N * 4
    smem_total = smem_q + smem_k + smem_s
    
    if smem_total > 48 * 1024:
        print(f"Skip {BLOCK_M}×{BLOCK_N}: SMEM overflow ({smem_total} bytes)")
        continue
    
    # Benchmark
    latency = benchmark_with_config(BLOCK_M, BLOCK_N)
    print(f"{BLOCK_M}×{BLOCK_N}: {latency:.2f} μs")
```

**Success Criteria**:
- ✅ Optimal tile size identified
- ✅ Within SMEM budget (48KB)

---

## **Phase B.4: NCU Validation** ⏱️ 1 hour

**Goal**: Verify Tensor Core utilization with Nsight Compute

### **TDD Steps**

#### **Test 1: Basic NCU Metrics** (30 min)
```bash
# Profile Phase 5 (cuBLAS hybrid)
ncu --target-processes all --replay-mode kernel \
  --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv -o evidence/ncu_phase5_cublas \
  python scripts/test_phase5.py
```

**Expected Metrics**:
```
sm__pipe_tensor_cycles_active: 50-60%  ✅ (vs 0% in Phase 4)
sm__throughput: 60-70%
sm__warps_active: 40-50%
dram__throughput: < 1% (still compute-bound)
```

**Success Criteria**:
- ✅ Tensor Core utilization > 50%
- ✅ SM throughput improved
- ✅ Still compute-bound (not memory-bound)

#### **Test 2: Comparison vs Phase 4** (30 min)
```python
# Load NCU results
ncu_phase4 = load_ncu_report('evidence/ncu_phase3_best.ncu-rep')
ncu_phase5 = load_ncu_report('evidence/ncu_phase5_cublas.ncu-rep')

# Compare
print("Tensor Core Utilization:")
print(f"  Phase 4: {ncu_phase4['tc_active']:.1f}%")
print(f"  Phase 5: {ncu_phase5['tc_active']:.1f}%")
print(f"  Improvement: +{ncu_phase5['tc_active']:.1f}%")

# Expected: 0% → 50-60%
assert ncu_phase5['tc_active'] > 50, "TC utilization too low"
print("✅ Tensor Core activation confirmed")
```

**Success Criteria**:
- ✅ TC utilization: 0% → 50-60%
- ✅ Performance correlates with TC usage

---

## **Phase B Success Criteria** (All Tests Must Pass)

```
✅ Correctness: 100/100 tests (max_diff < 2e-3)
✅ Performance: 400-500 μs (1.7-2.1× speedup vs Phase 4)
✅ Tensor Cores: 50-60% utilization (vs 0% in Phase 4)
✅ Gap to SDPA: 8.0-10.1× (reduced from 17.5×)
✅ Code quality: Clean, documented, tested
✅ Evidence: NCU reports, benchmark logs, correctness tests
```

---

## **Phase B Time Budget**

```
B.1: cuBLAS single-tile test: 2.0 hours
B.2: Integration: 2.0 hours
B.3: Tuning: 1.0 hour
B.4: NCU validation: 1.0 hour
───────────────────────────────────────
Total: 6.0 hours

Expected Outcome: 870 → 450 μs (1.9× speedup)
Confidence: 90% (cuBLAS proven)
```

---

## **Phase B Risk Mitigation**

| Risk | Probability | Mitigation |
|------|-------------|------------|
| cuBLAS handle passing | 20% | Test with minimal example first |
| SMEM overflow | 30% | Calculate before implementation |
| Correctness drift | 10% | TDD at every step |
| Performance < 400 μs | 20% | Tune algorithm and tiles |

**Overall Risk**: Medium (cuBLAS proven, but integration complex)

---

## **After Phase B: Phase C Preview**

**Goal**: Full WMMA pipeline + warp specialization → 50-70 μs

**Key Optimizations**:
1. WMMA for Q@K^T (replace cuBLAS for manual control)
2. WMMA for P@V (complete TC pipeline)
3. Warp specialization (producer/consumer)
4. Double-buffered SMEM (hide latency)
5. XOR swizzling (bank-conflict-free)
6. Evo sweep (find optimal config)

**Expected**: 450 → 60 μs (7.5× speedup, BEAT SDPA) ✅

---

**Ready to execute Phase B.1 with systematic TDD approach.**

