# FlashCore Architecture

**Version**: 1.0  
**Date**: October 21, 2025  
**Status**: Phase 0 (Baseline)

---

## 📐 System Design

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FlashCore System                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐       ┌──────────────┐                   │
│  │   PyTorch    │       │     CUDA     │                   │
│  │  Application │─────▶ │   Kernels    │                   │
│  └──────────────┘       └──────────────┘                   │
│                                │                             │
│                                ▼                             │
│                      ┌──────────────────┐                   │
│                      │  Tests (15 cases) │                  │
│                      └──────────────────┘                   │
│                                │                             │
│             ┌──────────────────┼──────────────────┐         │
│             ▼                  ▼                  ▼         │
│      ┌────────────┐    ┌────────────┐    ┌────────────┐   │
│      │ Benchmarks │    │  Profiling │    │   Search   │   │
│      │  (latency) │    │    (NCU)   │    │ (autotune) │   │
│      └────────────┘    └────────────┘    └────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧬 Kernel Evolution

### Phase 0: Baseline (v0.1)

**File**: `kernels/flashcore_baseline.cu`

**Algorithm**: FlashAttention minimal implementation
- Scalar accumulation (no WMMA)
- One thread block per query row
- Online softmax (running max/sum)
- FP16 storage, FP32 compute for softmax

**Performance**: ~1500 µs (mission shape)

**Hardware Utilization**:
- Registers: ~61
- Shared Memory: ~21 KB
- Tensor Cores: 0% (scalar baseline)
- DRAM Throughput: ~12% of peak

**Pseudocode**:
```python
for each query row q_i:
    m_i = -inf        # running max
    l_i = 0           # running sum
    o_i = zeros(D)    # output accumulator
    
    for each K/V tile (size BLOCK_N):
        # Load K tile, compute scores
        s_j = q_i @ k_j^T * scale
        
        # Update max
        m_new = max(m_i, max(s_j))
        
        # Load V tile
        v_tile = load_V_tile()
        
        # Compute exp, update output
        for j in tile:
            p_j = exp(s_j - m_new)
            l_new += p_j
            o_i += p_j * v_j
        
        # Rescale old output (if max changed)
        o_i *= exp(m_i - m_new)
        l_i *= exp(m_i - m_new)
        
        # Update state
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    o_i /= l_i
```

---

### Phase 1: WMMA (v0.2) - PLANNED

**File**: `kernels/flashcore_wmma.cu`

**Algorithm**: Use Tensor Cores for matrix multiplications
- WMMA for Q @ K^T (16×16×16 tiles)
- WMMA for P @ V (16×16×16 tiles)
- FP32 softmax accumulators (unchanged)
- Shared memory tiling (32×32 blocks)

**Target Performance**: ~150 µs (10× vs baseline)

**Hardware Utilization (Expected)**:
- Registers: ~100-120
- Shared Memory: ~40-60 KB
- Tensor Cores: >50% (primary goal)
- DRAM Throughput: ~30% of peak

**Key Changes**:
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

// Q @ K^T with WMMA
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> acc_frag;

load_matrix_sync(a_frag, Q_tile, 64);
load_matrix_sync(b_frag, K_tile, 64);
mma_sync(acc_frag, a_frag, b_frag, acc_frag);
store_matrix_sync(S_tile, acc_frag, 32, mem_row_major);

// P @ V with WMMA (similar)
```

---

### Phase 2: Fused (v0.3) - PLANNED

**File**: `kernels/flashcore_fused.cu`

**Algorithm**: FlashAttention-2 style fusion
- Sequence tiling (split S=512 into tiles of 64-128)
- Online softmax (maintained from baseline)
- Minimize global memory writes (only final O)
- Double buffering with cp.async (prefetch next tile)

**Target Performance**: <58 µs (26× vs baseline) ✅ **PRIMARY GOAL**

**Hardware Utilization (Expected)**:
- Registers: ~120-140
- Shared Memory: ~64-80 KB
- Tensor Cores: >60%
- DRAM Throughput: >50% of peak (memory-bound)

**Key Optimizations**:
1. **Tiling**: Split computation into tiles that fit in SMEM
2. **Fusion**: Single kernel, no intermediate global writes
3. **Prefetch**: `cp.async` to overlap load + compute
4. **Vectorization**: `float4` loads (8×FP16 at once)

---

### Phase 3: Warp-Specialized (v0.4) - PLANNED

**File**: `kernels/flashcore_optimized.cu`

**Algorithm**: Producer/consumer warp split
- Producer warps: Prefetch K/V tiles with `cp.async`
- Consumer warps: Compute WMMA while next tile loads
- Lightweight sync (warp-level instead of block-level)

**Target Performance**: ~20 µs (75× vs baseline) ⭐ STRETCH

**Hardware Utilization (Expected)**:
- Registers: ~140-160
- Shared Memory: ~80-100 KB
- Tensor Cores: >70%
- Thread-block barriers: <10 (vs 48 in baseline)

---

## 🔬 Algorithm Details

### Online Softmax (All Phases)

**Problem**: Softmax requires two passes (max, then sum)  
**Solution**: FlashAttention's online algorithm (single pass)

**State Variables** (per query row):
- `m_i`: Running maximum of scores
- `l_i`: Running sum of exp(s_j - m_i)
- `o_i`: Output accumulator (weighted sum of V)

**Update Rules** (per tile):
```python
# New max across current tile
m_new = max(m_i, max(s_tile))

# Rescale factor for old accumulator
alpha = exp(m_i - m_new)

# Update output (rescale old, add new)
o_i = o_i * alpha + sum(exp(s_tile - m_new) * v_tile)

# Update sum
l_i = l_i * alpha + sum(exp(s_tile - m_new))

# Update max
m_i = m_new
```

**Final Output**:
```python
O[i] = o_i / l_i
```

**Numerical Stability**:
- Use FP32 for `m_i`, `l_i` accumulators (prevent overflow)
- Subtract max before exp (prevent exp overflow)
- Clamp rescale factors (prevent underflow)

---

## 📊 Memory Layout

### Input/Output Tensors

**Format**: `[B, H, S, D]` (Batch, Heads, Sequence, Dimension)

```
Q: [B, H, S, D] = [1, 8, 512, 64] = 262,144 elements × 2 bytes = 524 KB
K: [B, H, S, D] = same = 524 KB
V: [B, H, S, D] = same = 524 KB
O: [B, H, S, D] = same = 524 KB

Total: 2 MB (mission shape)
```

**Memory Access Patterns**:
- Q: Row-major loads (one row per thread block)
- K: Tiled loads (BLOCK_N rows at a time)
- V: Tiled loads (BLOCK_N rows at a time)
- O: Row-major stores (one row per thread block)

### Shared Memory Tiling

**Phase 0** (Baseline):
```cuda
__shared__ float Q_row[64];           // One query row (FP32)
__shared__ float S_tile[64];          // Attention scores (FP32)
__shared__ float O_accum[64];         // Output accumulator (FP32)

Total: 64×4 + 64×4 + 64×4 = 768 bytes
```

**Phase 1-2** (WMMA):
```cuda
__shared__ half Q_smem[32][64];       // Q tile (FP16)
__shared__ half K_smem[64][64];       // K tile (FP16)
__shared__ half V_smem[64][64];       // V tile (FP16)
__shared__ float S_smem[32][64];      // Scores (FP32)

Total: 32×64×2 + 64×64×2 + 64×64×2 + 32×64×4 = 28 KB
```

**Phase 3** (Warp-Specialized):
```cuda
// Double-buffered K/V tiles
__shared__ half K_smem[2][64][64];    // Double-buffer K
__shared__ half V_smem[2][64][64];    // Double-buffer V
__shared__ volatile int kv_ready[2];  // Producer→Consumer signals
__shared__ volatile int kv_consumed[2]; // Consumer→Producer signals

Total: 2×64×64×2 + 2×64×64×2 + 16 bytes = 32 KB + 32 KB + 16 = 64 KB
```

---

## 🎯 Optimization Targets

### Performance Hierarchy

| Level | Technique | Expected Speedup | Cumulative |
|-------|-----------|------------------|------------|
| 0 | Baseline (scalar) | 1.0× | 1.0× |
| 1 | Tensor Cores (WMMA) | 10× | 10× |
| 2 | Memory Fusion (tiling) | 2.5× | 25× |
| 3 | Warp Specialization | 3× | 75× |
| 4 | Evolutionary Search | 1.3× | 100× |

**Primary Goal**: Achieve Level 2 (25× speedup → ~60 µs)

---

## 🔧 Build System

### Compilation Flags

```bash
# Optimization
-O3                         # Maximum optimization
--use_fast_math             # Fast math (exp, log)
-lineinfo                   # Line info for profiling

# Architecture
-arch=sm_89                 # L4 GPU (Ada)

# Debugging (optional)
-G                          # Enable device debug
-DDEBUG=1                   # Enable debug prints
```

### Environment Variables

```bash
CUDA_ARCH=8.9               # Target architecture
DEBUG=0                     # Debug mode (0/1)
VERBOSE=0                   # Verbose build output
```

---

## 📋 Testing Strategy

### Correctness Tests

**Multi-Shape Coverage**:
- Prevents overfitting to single input size
- Validates numerical stability across ranges
- 15 tests = 5 shapes × 3 seeds

**Accuracy Thresholds** (FP16):
```python
MAX_ERR_THRESHOLD = 0.06    # Max absolute error
MEAN_ERR_THRESHOLD = 0.02   # Mean absolute error
```

**NaN/Inf Detection**:
```python
assert not torch.isnan(O).any()
assert not torch.isinf(O).any()
```

### Performance Benchmarks

**Robust Statistics** (100 iterations):
- p50 (median): Primary metric
- p90: Tail latency
- p99: Worst-case latency
- mean, std: Consistency

**Comparison**:
- vs PyTorch SDPA: Absolute speedup
- vs Baseline: Relative improvement
- vs Target: Progress toward goal

---

## 🔬 Profiling Methodology

### NCU Metrics (Nsight Compute)

**Tensor Core Utilization**:
```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
Target: >50% (Phase 1), >70% (Phase 3)
```

**Memory Throughput**:
```
dram__throughput.avg.pct_of_peak_sustained_elapsed
Target: >50% (indicates memory-bound, expected)
```

**Warp Stalls**:
```
smsp__cycles_stalled.avg.pct_of_peak_sustained_active
Minimize: <30%
```

**Thread-Block Barriers**:
```
smsp__inst_executed_barrier.sum
Target: <10 (Phase 3, vs 48 baseline)
```

### Roofline Analysis

**Arithmetic Intensity** (FLOPs/Byte):
```
AI = (2×B×H×S×S×D) / (B×H×S×D×2 + B×H×S×D×2 + B×H×S×D×2)
   = (2×S×D) / (6×D) = S/3

For S=512: AI = 170 FLOPs/Byte
```

**Theoretical Latency**:
- Compute-bound: 536M FLOPs / 242 TFLOPS = 2.2 µs
- Memory-bound: 2 MB / 300 GB/s = 6.7 µs
- **Conclusion**: Memory-bound (6.7 µs > 2.2 µs)

---

## 📚 References

### FlashAttention Algorithm

**Paper**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"  
**Authors**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré  
**Venue**: NeurIPS 2022

**Key Contributions**:
1. Tiling to reduce memory traffic
2. Online softmax (single-pass)
3. Recomputation in backward pass

### WMMA Programming

**NVIDIA Guide**: [WMMA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)

**Key Types**:
```cuda
fragment<matrix_a, 16, 16, 16, half, row_major>
fragment<matrix_b, 16, 16, 16, half, col_major>
fragment<accumulator, 16, 16, 16, float>
```

**Operations**:
```cuda
load_matrix_sync(frag, ptr, ldm);
mma_sync(acc, a, b, acc);
store_matrix_sync(ptr, acc, ldm, mem_row_major);
```

---

## ✅ Architecture Status

**Phase 0**: ✅ Complete  
**Phase 1**: 🔄 In Progress (WMMA design)  
**Phase 2**: ⏳ Planned (Fusion design)  
**Phase 3**: ⏳ Planned (Warp specialization design)  
**Phase 4**: ⏳ Planned (Autotune design)

**Last Updated**: October 21, 2025  
**Version**: 1.0 (Baseline)

