# I6 Block-Parallel Design Document

**Status**: Design Phase  
**Target**: 15-20 μs/head (5× faster than I5)  
**Architecture**: Block-parallel with warpgroup cooperation

---

## 🎯 Design Goals

### Performance Targets
- **Latency**: 15-20 μs/head (vs 90.67 μs in I5)
- **SM Utilization**: 60-70% (vs 24.8% in I5)
- **Synchronizations**: ≤8 per kernel (vs 64 in I5)
- **Memory Bandwidth**: >500 GB/s (vs ~62 GB/s in I5)

### Security Requirements
- ✅ Maintain all constant-time primitives
- ✅ Zero data-dependent branches
- ✅ Bitwise reproducibility
- ✅ Fixed iteration counts

---

## 🏗️ Architectural Changes

### From Row-Parallel (I4/I5) to Block-Parallel (I6)

**I5 Model (WRONG)**:
```
Grid:  (B*H*S_max) threads
Block: 256 threads
Each thread: Process 1 complete row
Result: 24.8% SM utilization, 64 syncs
```

**I6 Model (CORRECT)**:
```
Grid:  (B*H) × (S_max/BM) blocks
Block: 128 threads (4 warps)
Each block: Process BM×BN tile collaboratively
Result: 60-70% SM utilization, 8 syncs
```

---

## 📐 Tile Configuration

### Optimal Tile Sizes
- **BM = 64**: Rows per block (balance between parallelism and register pressure)
- **BN = 64**: Columns per block (matches BM for square tiles)
- **BK = 64**: Head dimension (fixed by model architecture)

### Grid Dimensions
```cpp
dim3 grid(
    batch_size,              // B*H pairs
    (S_max + BM - 1) / BM    // Number of row tiles
);
dim3 block(128);  // 4 warps
```

**Example** (B=4, H=16, S=1024):
- I5: 65,536 threads total (poor utilization)
- I6: 1,024 blocks × 128 threads = 131,072 active threads (60%+ utilization) ✅

---

## 🧮 Computation Strategy

### Phase 1: Load Q Tile (One-Time)
```cpp
// Q_tile: [BM, BK] = [64, 64] = 8 KB shared memory
// Cooperative load: Each thread loads 32 elements
// Sync: 1
```

### Phase 2: Loop Over K/V Tiles
For each column tile `j` in `[0, S_max/BN)`:

**2a. Load K Tile**:
```cpp
// K_tile: [BN, BK] = [64, 64] = 8 KB
// Sync: 1 per tile
```

**2b. Compute S = Q @ K^T**:
```cpp
// S_tile: [BM, BN] = [64, 64] = 8 KB
// Each thread computes 32 dot products
// Sync: 1 per tile (after compute)
```

**2c. Load V Tile**:
```cpp
// V_tile: [BN, BK] = [64, 64] = 8 KB
// Sync: 1 per tile
```

**2d. Online Softmax + P @ V**:
```cpp
// Per-thread online softmax state
// Accumulate output: out += softmax(S) @ V
// No sync needed (thread-local)
```

**Total syncs**: 1 (Q load) + `num_tiles` × 3 (K, S, V) ≈ 1 + 16×3 = 49

⚠️ **Still too many!** Need to reduce...

### Optimized Approach (Double Buffering)
- Use 2 sets of K/V tiles
- Overlap load of next tile with compute of current tile
- **Reduces syncs to**: 1 + `num_tiles` × 2 ≈ 33 (better, but still high)

---

## 🔧 Implementation Challenges

### Challenge 1: Per-Thread Output Accumulation
**Problem**: Each thread needs to accumulate d=64 output values.
- **Registers needed**: 64 float = 64 registers
- **Total with m/l state**: ~80 registers per thread ✅ (under 255 limit)

**Solution**: Each thread processes 1-2 rows, accumulates full d=64 output.

### Challenge 2: Thread-to-Row Mapping
**Problem**: How to assign 128 threads to 64 rows?

**Option A**: 2 threads per row (warp-level reduction)
```cpp
const int row = threadIdx.x / 2;
const int sub_row = threadIdx.x % 2;
// Thread 0,1 → row 0
// Thread 2,3 → row 1
// ...
```

**Option B**: 1 thread per row, some idle
```cpp
const int row = threadIdx.x;
if (row < BM) {  // First 64 threads active
    // Process row
}
// Threads 64-127 idle (50% waste!) ❌
```

**Best**: Option A with warp reduction for better utilization.

### Challenge 3: Constant-Time with Block-Parallel
**Problem**: Causal masking creates triangular access patterns.

**Solution**: Process full rectangular tiles, mask with `ct_select`:
```cpp
for (int col = 0; col < BN; ++col) {
    int global_col = col_start + col;
    uint32_t causal_valid = ct_le_u32(global_col, global_row);
    score = ct_select_f32(-INFINITY, score, causal_valid);
}
```

All threads execute all iterations (constant-time) ✅

---

## 📊 Expected Performance Analysis

### Memory Traffic (per tile)
- **Q tile load**: 8 KB × 1 = 8 KB (one-time)
- **K tile load**: 8 KB × 16 tiles = 128 KB
- **V tile load**: 8 KB × 16 tiles = 128 KB
- **Output write**: 8 KB × 16 tiles = 128 KB
- **Total**: 392 KB per (B*H) pair

**Bandwidth** (S=1024, B*H=64):
- Total traffic: 392 KB × 64 = 25 MB
- Target latency: 20 μs
- Required BW: 25 MB / 20 μs = 1.25 TB/s ✅ (H100 has 2 TB/s)

### Compute (per tile)
- **Q@K^T**: BM × BN × BK = 64 × 64 × 64 = 262K FP16 ops
- **Softmax**: BM × BN = 4K exp() calls
- **P@V**: BM × BN × BK = 262K FP16 ops
- **Total per tile**: ~530K ops
- **Total (16 tiles)**: 8.5M ops per (B*H)

**Compute time** (scalar FP16):
- H100 FP16: 33 TFLOPS = 33M ops/μs
- Time: 8.5M / 33M = 0.26 μs per (B*H) ✅ (compute-light)

**Conclusion**: Memory-bound (as expected), BW requirements achievable.

---

## 🚧 Implementation Complexity

### Issues with Current Draft
1. **Output accumulation**: Thread-to-row mapping needs careful design
2. **Warp-level reduction**: Not implemented for 2 threads/row
3. **Memory aliasing**: Shared memory bank conflicts possible
4. **Register pressure**: Need to verify actual usage
5. **Correctness**: Complex data flow, easy to introduce bugs

### Recommended Approach: Incremental Development

**Step 1**: Simplified I6 (this session if time)
- Single-threaded per row (50% thread idle, but correct)
- Target: 30-40 μs/head (2-3× faster than I5)
- Validates architecture change

**Step 2**: Optimized I6 (next session)
- 2 threads per row with warp reduction
- Target: 15-20 μs/head (5× faster than I5)
- Production-quality

**Step 3**: I7 with WMMA (future)
- Add Tensor Core matrix multiply
- Target: 5-8 μs/head
- Competitive with PyTorch

---

## 🎓 Learning from FlashAttention-3

### What FA3 Does Right
1. **Persistent kernels**: Stay resident, no launch overhead
2. **TMA**: Async global→shared DMA
3. **WGMMA**: Native Hopper Tensor Core ops
4. **Warpgroup scheduling**: 128 threads cooperate efficiently

### What We Can Adopt Now (I6)
- ✅ Block-parallel tiling
- ✅ Shared memory staging
- ✅ Cooperative loads
- ❌ TMA (too complex for I6)
- ❌ WGMMA (needs WMMA first)
- ❌ Persistent (needs more infrastructure)

---

## 📈 Realistic Expectations

### If We Implement Simplified I6
- **Best case**: 25-30 μs/head (3× faster than I5)
- **Worst case**: 60-70 μs/head (1.3× faster than I5, due to overhead)
- **Likely**: 30-40 μs/head (2-3× faster than I5) ✅

### Why Not 5× Speedup?
- Still using scalar operations (no Tensor Cores)
- Still have 33+ syncs (no double buffering)
- No async loads (no TMA)

**Bottom line**: I6 proves the architecture, I7/I8 get the performance.

---

## 🏁 Next Steps

### For This Session (if time permits)
1. Implement simplified I6 (1 thread/row)
2. Test on H100
3. Validate 2-3× speedup
4. Document findings

### For Next Session
1. Optimize I6 (2 threads/row + warp reduction)
2. Add double buffering
3. Target 15-20 μs/head

### Future (I7+)
1. Integrate WMMA for Q@K^T and P@V
2. Add TMA for async loads
3. Implement persistent kernel
4. Target <5 μs/head

---

## 💡 Key Insight

**The fundamental architectural change (row→block parallel) is more important than any single optimization.**

Even a "naive" block-parallel implementation should be 2-3× faster than optimized row-parallel (I5).

This validates the approach before investing in complex optimizations.

