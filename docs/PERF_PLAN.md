# FP8 SDPA Stage-C WMMA — Performance Optimization Plan

**Status**: Correctness LOCKED ✅ → Ready for Performance Work  
**Baseline**: TBD (run `runner.py --shapes mission --seeds 0 --iters 500`)  
**Target**: < 50 μs for mission shape (B=1, H=8, S=512, D=64)

---

## 🎯 **Optimization Philosophy**

Following **EvoEngineer methodology**:

1. **GREEN before FAST** — Correctness gates must pass after every change
2. **Staged improvements** — Each stage on its own branch with measured delta
3. **Elite population** — Keep top-3 variants at each stage
4. **NCU-driven** — Profile → Hypothesis → Change → Validate

**Hard Rule**: If any correctness gate fails, revert immediately.

---

## 📊 **Staged Optimization Plan**

### **Stage 1: Double-Buffered K/V with cp.async**

**Goal**: Overlap global→shared memory transfers with WMMA compute

**Expected Speedup**: +10-15% (mission shape)  
**Risk**: Low (well-established pattern)  
**Effort**: 2-3 hours

#### **Changes**

1. **Ping-Pong Buffers**
   ```cuda
   __shared__ half sKT[2][TILE_N][D_PAD];  // Double-buffered
   __shared__ half sV[2][TILE_N][D_PAD];   // Double-buffered
   ```

2. **Async Copy**
   ```cuda
   // Producer: load tile t+1 while compute consumes tile t
   int write_stage = (t + 1) % 2;
   int read_stage = t % 2;
   
   // Issue cp.async for write_stage
   __pipeline_memcpy_async(&sKT[write_stage][...], &Kbh[...], ...);
   __pipeline_commit();
   
   // Compute on read_stage
   __pipeline_wait_prior(1);  // Wait for current tile
   // ... WMMA on sKT[read_stage], sV[read_stage] ...
   ```

3. **Synchronization**
   - `__pipeline_commit()` after each cp.async batch
   - `__pipeline_wait_prior(1)` before consuming a stage
   - Final `__syncthreads()` after all tiles

#### **Exit Criteria**

- ✅ All correctness gates pass (3 seeds × 2 shapes)
- ✅ Mission shape: +10-15% speedup (p50 latency)
- ✅ Registers ≤ 128/thread (check PTXAS)
- ✅ Occupancy ≥ 50% (check NCU `sm__warps_active`)

#### **NCU Validation**

```bash
# Before
scripts/profile_ncu.sh mission > baseline.txt

# After Stage 1
scripts/profile_ncu.sh mission > stage1.txt

# Compare:
# - dram__bytes.sum should stay similar (no extra traffic)
# - sm__throughput should increase (better overlap)
```

---

### **Stage 2: Vectorized P·V Accumulation**

**Goal**: Replace scalar `for (n)(for d)` with SIMD half2/float2 loads

**Expected Speedup**: +10% (mission shape)  
**Risk**: Low (math identical, just vector intrinsics)  
**Effort**: 1-2 hours

#### **Changes**

1. **Vectorize Inner Loop**
   ```cuda
   // BEFORE (scalar)
   for (int n = 0; n < kv_len; ++n) {
       float p = S_row[n];
       for (int d = lane; d < D; d += 32) {
           U_smem[r][d] += p * __half2float(sV[n][d]);
       }
   }
   
   // AFTER (vectorized)
   for (int n = 0; n < kv_len; ++n) {
       float p = S_row[n];
       for (int d = lane * 2; d < D; d += 64) {  // Process 2 elements/lane
           half2 v_vec = *reinterpret_cast<const half2*>(&sV[n][d]);
           float2 u_vec = make_float2(
               U_smem[r][d] + p * __half2float(v_vec.x),
               U_smem[r][d+1] + p * __half2float(v_vec.y)
           );
           U_smem[r][d] = u_vec.x;
           U_smem[r][d+1] = u_vec.y;
       }
   }
   ```

2. **Alignment**
   - Ensure `sV` and `U_smem` are 16B-aligned (already done with `alignas(16)`)
   - Guard tail elements if D not multiple of 2 (not needed for D=64)

#### **Exit Criteria**

- ✅ All correctness gates pass
- ✅ Mission shape: +10% speedup vs Stage 1
- ✅ Numerics identical (bit-exact if possible, else ≤ 1e-6 diff)

---

### **Stage 3: WMMA for P·V (Second Matmul)**

**Goal**: Replace scalar P·V accumulation with Tensor Core WMMA

**Expected Speedup**: +20-30% (mission shape)  
**Risk**: Medium (numerical subtlety with online softmax rescaling)  
**Effort**: 4-6 hours

#### **Design (Detailed)**

**Challenge**: P (attention weights) must be rescaled per row as new KV tiles arrive (online softmax). WMMA expects stable input fragments.

**Solution**: Accumulate in FP32, rescale U before each WMMA P·V step.

1. **P Fragment Construction**
   ```cuda
   // After online softmax, build P fragment (16×16 half)
   __shared__ half sP[TILE_M][TILE_N];  // Attention weights for current tile
   
   // Each warp fills its 16×16 slice
   for (int r = warp_m; r < warp_m + 16; r++) {
       for (int n = warp_n + lane; n < warp_n + 16; n += 32) {
           sP[r][n] = __float2half(S_row[n]);  // exp(score - m_new)
       }
   }
   __syncthreads();
   ```

2. **WMMA P·V**
   ```cuda
   // Load P: row-major [16, 16]
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
   wmma::load_matrix_sync(p_frag, &sP[warp_m][warp_n], TILE_N);
   
   // Load V: col-major or row-major (TBD: benchmark both)
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
   wmma::load_matrix_sync(v_frag, &sV[warp_n][k], D_PAD);
   
   // Accumulate into FP32 fragment
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> u_frag;
   wmma::mma_sync(u_frag, p_frag, v_frag, u_frag);
   ```

3. **Rescale U Before Next Tile**
   ```cuda
   // After computing new m_new, rescale old U by exp(m_old - m_new)
   float rescale = __expf(m_old - m_new);
   #pragma unroll
   for (int i = 0; i < u_frag.num_elements; i++) {
       u_frag.x[i] *= rescale;
   }
   ```

4. **Final Normalization**
   ```cuda
   // After all tiles, divide U by l_final
   #pragma unroll
   for (int i = 0; i < u_frag.num_elements; i++) {
       u_frag.x[i] /= l_final;
   }
   
   // Store to global
   wmma::fragment<wmma::accumulator, 16, 16, 16, half> o_frag;
   #pragma unroll
   for (int i = 0; i < u_frag.num_elements; i++) {
       o_frag.x[i] = __float2half(u_frag.x[i]);
   }
   wmma::store_matrix_sync(&Obh[...], o_frag, D_PAD, wmma::mem_row_major);
   ```

#### **Exit Criteria**

- ✅ All correctness gates pass (critical: online softmax numerics!)
- ✅ Mission shape: +20-30% speedup vs Stage 2
- ✅ NCU: `smsp__pipe_tensor_cycles_active` increases (more TC usage)
- ✅ No NaNs/infs in debug runs

#### **Risk Mitigation**

- Prototype in separate branch `feat/stage3-wmma-pv`
- Add debug print for U fragment values after each tile
- Cross-check against scalar P·V for first 3 rows
- If numerics diverge, revert and keep Stage 2

---

### **Stage 4: Warp Specialization (Producer/Consumer)**

**Goal**: Dedicate warps to cp.async (producer) vs WMMA (consumer)

**Expected Speedup**: +10-15% (mission shape)  
**Risk**: High (occupancy trade-offs, synchronization complexity)  
**Effort**: 6-8 hours

#### **Design**

Split 4 warps into:
- **1 producer warp**: Issues cp.async for K/V tiles
- **3 consumer warps**: Execute WMMA Q@K^T and P·V

**Implementation**:
```cuda
int role = (warp_id < 1) ? PRODUCER : CONSUMER;

if (role == PRODUCER) {
    // Issue cp.async for all tiles
    for (int t = 0; t < nTiles; ++t) {
        // ... cp.async for sKT[write_stage], sV[write_stage] ...
        __pipeline_commit();
    }
} else {
    // Consume tiles
    for (int t = 0; t < nTiles; ++t) {
        __pipeline_wait_prior(1);
        // ... WMMA on sKT[read_stage], sV[read_stage] ...
    }
}
```

#### **Exit Criteria**

- ✅ All correctness gates pass
- ✅ Mission shape: +10-15% speedup vs Stage 3
- ✅ Occupancy ≥ 40% (may drop due to fewer compute warps)
- ✅ NCU: `dram__bytes.sum` decreases (better prefetch)

#### **Fallback**

If occupancy drops too much or synchronization bugs arise, **skip this stage** and proceed to Stage 5.

---

### **Stage 5: XOR Swizzle for Bank Conflict Avoidance**

**Goal**: Eliminate SMEM bank conflicts on sKT/sV accesses

**Expected Speedup**: +5-10% (mission shape)  
**Risk**: Low (well-known pattern)  
**Effort**: 2-3 hours

#### **Changes**

1. **Apply XOR Swizzle**
   ```cuda
   // BEFORE
   sKT[n][d] = value;
   
   // AFTER
   int n_blk = n >> 3;
   int n_in = n & 7;
   int n_swizzle = (n_blk ^ (d >> 3)) * 8 + n_in;
   sKT[n_swizzle][d] = value;
   ```

2. **Update WMMA Load**
   - Adjust pointer arithmetic to account for swizzle
   - Or: apply inverse swizzle on load path

#### **Exit Criteria**

- ✅ All correctness gates pass
- ✅ Mission shape: +5-10% speedup vs Stage 4 (or Stage 3 if 4 skipped)
- ✅ NCU: `l1tex__data_bank_conflicts` near zero

---

## 📈 **Cumulative Targets**

Assuming baseline (after correctness fixes) = **TBD μs** for mission shape:

| Stage | Speedup vs Prev | Cumulative Speedup | Target Latency (est.) |
|-------|-----------------|--------------------|-----------------------|
| **Baseline** | - | 1.0× | TBD μs |
| **Stage 1** (cp.async) | +12% | 1.12× | TBD / 1.12 |
| **Stage 2** (SIMD P·V) | +10% | 1.23× | TBD / 1.23 |
| **Stage 3** (WMMA P·V) | +25% | 1.54× | TBD / 1.54 |
| **Stage 4** (Warp spec) | +12% | 1.73× | TBD / 1.73 |
| **Stage 5** (XOR swizzle) | +7% | 1.85× | TBD / 1.85 |

**Final Goal**: < 50 μs (requires baseline < 93 μs for 1.85× to hit target)

---

## 🛡️ **Regression Prevention**

### **Per-Stage Checklist**

Before merging each stage:

1. ✅ Run `python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small,mission --seeds 0,1,2`
2. ✅ All 6 checks (2 shapes × 3 seeds) must PASS
3. ✅ Run `scripts/profile_ncu.sh mission` and save report
4. ✅ Compare p50/p90 latency vs previous stage (must improve!)
5. ✅ Check PTXAS: regs ≤ 128, SMEM ≤ 64 KB
6. ✅ Update `results/COMPARE.md` with delta table

### **CI Integration** (Future)

```yaml
# .github/workflows/fp8_wmma_regression.yml
- name: Correctness Gate
  run: python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes small --seeds 0 --iters 10
  
- name: Performance Baseline
  run: python -m tasks.fp8_sdpa_stage_c_wmma.runner --shapes mission --seeds 0 --iters 100
```

---

## 📚 **References**

- **FlashAttention-2**: Online softmax + tiling (Dao et al., 2023)
- **CUTLASS**: Pipelined GEMM patterns (NVIDIA, 2024)
- **EvoEngineer**: Automated kernel optimization (Guo et al., 2025)
- **NVIDIA WMMA Programming Guide**: Tensor Core best practices

---

**Status**: Waiting for baseline measurement  
**Next Action**: Run `runner.py --shapes mission --seeds 0 --iters 500` on L4 GPU  
**Then**: Begin Stage 1 (cp.async double-buffering) on branch `feat/stage1-cp-async`

