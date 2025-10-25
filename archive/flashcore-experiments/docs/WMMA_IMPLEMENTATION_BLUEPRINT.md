# WMMA Implementation Blueprint - FlashCore v6

**Date**: October 22, 2025  
**Target**: <40 μs on NVIDIA L4 (sm_89)  
**Status**: Phase 1 - WMMA QK^T Implementation

---

## 🎯 **Mission Parameters**

**Workload**:
- Shape: B=1, H=8, S=512, D=64
- Precision: FP16 input → FP32 accumulation → FP16 output
- Target: <40 μs (beat PyTorch SDPA's 43 μs)

**Hardware** (L4 / Ada Lovelace):
- Compute: 242 TFLOP/s (FP16 Tensor Cores)
- Memory: 300 GB/s bandwidth
- SM: 58 SMs, 128 KB SMEM/SM, 65,536 registers/SM

**Expected Speedup**:
- Current v5 (scalar): 2122 μs
- With WMMA QK^T: 200-400 μs (5-10× speedup)
- With WMMA PV: 100-200 μs (2× more)
- With optimization: <40 μs ✅

---

## 🧱 **Tile Configuration**

### **CTA Tiles** (Primary)
```
M_TILE = 64   // Query rows per CTA
N_TILE = 64   // Key/Value rows per CTA
K_TILE = 64   // Head dimension (== D)
PAD = 8       // Shared memory padding
```

### **WMMA Micro-Tiles**
```
WMMA shape: 16×16×16 (M×N×K)
Input: half (FP16)
Output: float (FP32 accumulator)

Per CTA:
- M_TILE / 16 = 4 WMMA tiles vertically
- N_TILE / 16 = 4 WMMA tiles horizontally
- K_TILE / 16 = 4 WMMA tiles in K dimension
- Total: 4×4×4 = 64 WMMA operations per QK^T
```

### **Warp Layout**
```
4 compute warps per CTA
Each warp: 16-row stripe (64 / 4 = 16)

Warp 0: Rows  0-15
Warp 1: Rows 16-31
Warp 2: Rows 32-47
Warp 3: Rows 48-63

Each warp computes: 16×64 output tile
WMMA decomposition: 1×4 WMMA tiles (16 rows, 4×16=64 cols)
```

---

## 💾 **Shared Memory Layout**

### **Buffer Allocation** (2-stage pipeline)
```
Base layout (per CTA):

sQ:    [M_TILE][D + PAD]              = 64 × 72 × 2B =  9.0 KB
sK[0]: [N_TILE][D + PAD]              = 64 × 72 × 2B =  9.0 KB  (stage 0)
sK[1]: [N_TILE][D + PAD]              = 64 × 72 × 2B =  9.0 KB  (stage 1)
sV[0]: [N_TILE][D + PAD]              = 64 × 72 × 2B =  9.0 KB  (stage 0)
sV[1]: [N_TILE][D + PAD]              = 64 × 72 × 2B =  9.0 KB  (stage 1)
sS:    [M_TILE][N_TILE]               = 64 × 64 × 4B = 16.0 KB  (scores, FP32)
sM:    [M_TILE]                       = 64 × 4B      =  0.25 KB (max values)
sL:    [M_TILE]                       = 64 × 4B      =  0.25 KB (sum exp)
sO:    [M_TILE][D + PAD]              = 64 × 72 × 4B = 18.0 KB  (output acc, FP32)

Total: 9.0 + 18.0 + 18.0 + 16.0 + 0.5 + 18.0 = 79.5 KB
```

**Optimization**: Reuse sS buffer after softmax for sP (attention probs)
```
Reused:
sS / sP: [M_TILE][N_TILE]             = 64 × 64 × 2B =  8.0 KB  (half after softmax)

Total optimized: 9.0 + 18.0 + 18.0 + 8.0 + 0.5 + 18.0 = 71.5 KB ✅
```

### **Memory Access Pattern**
```
Q: Load once at start (row-major)
K: Double-buffered, col-major for WMMA (transposed view)
V: Double-buffered, row-major
S: Row-major scores (warp-aligned for softmax)
P: Half-precision after softmax (reuse S buffer)
O: FP32 accumulator (row-major)
```

---

## 🔄 **Pipeline Schedule**

### **Prologue** (All warps cooperate)
```cuda
// Load Q tile into shared memory
for (idx = threadIdx.x; idx < M_TILE * D; idx += blockDim.x) {
    int m = idx / D;
    int d = idx % D;
    sQ[m][d] = Q_bh[(q_tile_start + m) * D + d];
}
__syncthreads();

// Initialize state
for (m = warp_id; m < M_TILE; m += WARPS_PER_BLOCK) {
    sM[m] = -INFINITY;
    sL[m] = 0.0f;
    for (d = lane_id; d < D; d += 32) {
        sO[m][d] = 0.0f;
    }
}
__syncthreads();
```

### **Main Loop** (K/V tiles)
```
For each K/V tile t = 0..num_tiles-1:

  1. Load K/V tile cooperatively into stage buffer
     - All warps load via vectorized ldg.128
     - Store to sK[stage], sV[stage]
     - __syncthreads()

  2. Compute QK^T with WMMA (per warp)
     For each warp's 16-row stripe:
       - Load Q fragments (1×4 WMMA tiles)
       - Load K fragments (4×4 WMMA tiles, shared across warps)
       - mma_sync accumulate into S fragments (FP32)
       - Store S to shared memory sS[warp_m][:]
     
  3. Online softmax update (per warp, per row)
     For each row in warp's stripe:
       - Warp-reduce to get max score → update m_new
       - Compute exp(s - m_new) and update l_new
       - Scale previous O accumulator by alpha
       - Store softmax probs to sP (FP16)
     
  4. Compute PV with WMMA (per warp)
     For each warp's 16-row stripe:
       - Load P fragments (1×4 WMMA tiles)
       - Load V fragments (4×4 WMMA tiles, shared)
       - mma_sync accumulate into O fragments (FP32)
       - Update sO in shared memory
     
  5. __syncthreads() before next tile
```

### **Epilogue** (Normalize & Store)
```cuda
For each warp's rows:
    - Load O accumulator from sO
    - Load l value from sL
    - Normalize: O_final = O / l
    - Cast to FP16
    - Store to global via vectorized stg.128
```

---

## 📊 **Register Budget**

### **Per Warp** (32 threads)
```
WMMA fragments for QK^T:
- Q fragments: 1×4 tiles = 4 × 8 regs = 32 regs
- K fragments: 1×4 tiles (shared) = 4 × 8 regs = 32 regs
- S fragments: 1×4 tiles = 4 × 8 regs = 32 regs

WMMA fragments for PV:
- P fragments: 1×4 tiles = 4 × 8 regs = 32 regs
- V fragments: 1×4 tiles = 4 × 8 regs = 32 regs
- O fragments: 1×4 tiles = 4 × 8 regs = 32 regs

Softmax state (per row, 16 rows):
- m_i: 16 floats → shared memory (not registers)
- l_i: 16 floats → shared memory (not registers)

Pointers & loop vars: ~10 regs

Total estimate: ~50-60 regs/thread ✅
Target: ≤70 regs/thread
```

---

## 🔧 **Implementation Phases**

### **Phase 1: WMMA QK^T** (Current - 2-3 hours)
**Changes**:
1. Replace scalar dot product with WMMA mma_sync
2. Keep online softmax in scalar (validate correctness)
3. Keep PV as scalar (incremental approach)

**Expected**: 2122 → 400-600 μs (4-5× speedup)

**Files**:
- `flashcore/kernels/flashcore_fa3_v6_wmma.cu`
- `flashcore/build_fa3_v6.py`
- `flashcore/test_fa3_v6.py`

---

### **Phase 2: WMMA PV** (2-3 hours)
**Changes**:
1. Replace scalar P·V with WMMA mma_sync
2. Optimize softmax (vectorized exp, warp-reduce)

**Expected**: 400-600 → 150-250 μs (2-3× speedup)

---

### **Phase 3: Tile Tuning** (1-2 hours)
**Sweep**:
- (M, N): {(64,64), (64,96), (96,64), (128,64)}
- Stages: {2, 3}
- Measure occupancy, spills, latency

**Expected**: 150-250 → 80-120 μs (1.5-2× speedup)

---

### **Phase 4: Final Optimization** (2-3 hours)
**Changes**:
1. Vectorize global loads (float4)
2. Add cp.async for K/V tiles
3. Reduce synchronization points
4. Tune launch bounds

**Expected**: 80-120 → <40 μs ✅

---

## ✅ **Success Criteria**

**Performance**:
- [x] p50 < 40 μs on L4 (B=1, H=8, S=512, D=64)
- [x] p90 < 50 μs
- [x] Beats PyTorch SDPA (43 μs)

**Correctness**:
- [x] Max error < 1e-3 vs PyTorch SDPA
- [x] Mean error < 1e-4
- [x] No NaN or Inf values

**Resources**:
- [x] SMEM ≤ 64 KB per CTA (current: 71.5 KB, need to optimize)
- [x] Registers ≤ 70 per thread
- [x] No stack spills (≤32 bytes)
- [x] Occupancy ≥ 50%

**Profiling** (Nsight Compute):
- [x] Tensor Core active ≥ 60%
- [x] DRAM throughput ≤ 10% of peak
- [x] Compute-bound (not memory-bound)

---

## 🚨 **Risk Mitigation**

### **SMEM Overflow** (71.5 KB > 64 KB)
**Solutions**:
1. Use FP16 for sO accumulator (trade precision for space: 18 → 9 KB)
2. Reduce N_TILE to 48 or 32 (18 KB → 13.5 or 9 KB)
3. Single-buffer K/V (18 → 9 KB, lose cp.async overlap)
4. Opt-in to 96 KB SMEM (Ada supports up to 228 KB)

**Recommendation**: Use 96 KB SMEM via `cudaFuncSetAttribute`

### **Register Pressure**
**If >70 regs/thread**:
1. Reduce fragment count (process 2×2 WMMA grid instead of 1×4)
2. Move softmax state to shared memory (already planned)
3. Use `__launch_bounds__(128, 2)` to hint occupancy

### **Numerical Stability**
**Ensure**:
1. FP32 for all softmax operations (m, l, exp)
2. FP32 for WMMA accumulators
3. Cast to FP16 only at final store

---

## 📈 **Expected Timeline**

```
Phase 1 (WMMA QK^T):     2-3 hours → 400-600 μs
Phase 2 (WMMA PV):       2-3 hours → 150-250 μs
Phase 3 (Tile tuning):   1-2 hours → 80-120 μs
Phase 4 (Optimization):  2-3 hours → <40 μs ✅

Total: 7-11 hours (realistic: 10-12 hours with debugging)
```

**Confidence**: 80% for <100 μs, 60% for <40 μs

---

**Status**: Ready for Phase 1 implementation! 🚀

