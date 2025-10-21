# Warp Specialization Implementation Guide

**Status**: Phase 1 COMPLETE (infrastructure), Phase 2 PENDING (kernel logic)  
**Estimated Time**: 4-6 hours on L4  
**Risk**: Medium (complex synchronization, must test carefully)

---

## 🎯 Goal

Implement warp specialization in `sdpa_fp8_stage_c_wmma.cu` to overlap:
- **Producer warps**: Async load + dequantization (K/V tiles)
- **Consumer warps**: Compute (Q@K^T, softmax, P·V)

**Expected Speedup**: +10-20% over Stage-2 (656 μs → ~550-590 μs)

---

## 📋 Pre-Requisites

1. ✅ Toggles added (`USE_WARP_SPECIALIZATION`, etc.)
2. ✅ Sync helpers added (`stage_store_release`, `stage_spin_acquire`)
3. ✅ Benchmarking infrastructure ready (`scripts/bench_sdpa.py`)
4. ⏳ **Kernel WS logic** (this guide)

---

## 🔧 Implementation Steps

### Step 1: Add Warp Role Detection (Easy, 5 min)

**Location**: After `warp_id` declaration in kernel  
**File**: `cudadent42/bench/kernels/sdpa_fp8_stage_c_wmma.cu`

```cuda
const int warp_id = tid >> 5;             // 0..(NUM_WARPS-1)
const int lane    = tid & 31;

#if USE_WARP_SPECIALIZATION
const bool is_producer = (warp_id < NUM_PRODUCER_WARPS);
#endif
```

**Validation**: Compile with `USE_WARP_SPECIALIZATION=1`, check PTXAS output

---

### Step 2: Add Handshake Flags (Easy, 5 min)

**Location**: In shared memory declarations (after `sQ`, `sKT`, `sV`)

```cuda
__shared__ alignas(16) half sQ[TILE_M][D_PAD];
__shared__ alignas(16) half sKT[TILE_N][D_PAD];
__shared__ alignas(16) half sV[TILE_N][D_PAD];

#if USE_WARP_SPECIALIZATION
// Producer/consumer handshake flags for double-buffering
__shared__ volatile int kv_ready[2];      // Producer sets to 1 when tile ready
__shared__ volatile int kv_consumed[2];   // Consumer sets to 1 when done
#endif
```

---

### Step 3: Initialize Flags (Easy, 5 min)

**Location**: Before main tile loop, after Q load

```cuda
#if USE_WARP_SPECIALIZATION
// Init handshake flags
if (tid == 0) {
    kv_ready[0] = kv_ready[1] = 0;      // No tiles ready yet
    kv_consumed[0] = kv_consumed[1] = 1; // Buffers available for writing
}
__syncthreads();
#endif
```

---

### Step 4: Split Producer/Consumer Logic (HARD, 2-3 hours)

This is the core change. Replace the existing `cp.async` tile loop with producer/consumer split.

**Location**: Inside `#if USE_CP_ASYNC` block

#### **Current Structure** (Stage-2):
```cuda
for (int t = 0; t < nTiles; ++t) {
    const int read_stage = t % NUM_STAGES;
    
    // Prefetch next tile
    if (t + 1 < nTiles) {
        cp_async_tile_u8(t + 1, write_stage);
    }
    
    __pipeline_wait_prior(NUM_STAGES - 2);
    __syncthreads();
    
    // Dequantize u8 → half (sK_u8 → sKT, sV_u8 → sV)
    // ... existing dequant code ...
    
    __syncthreads();
    
    // Compute Q@K^T, softmax, P·V
    // ... existing compute code ...
    
    __syncthreads();
}
```

#### **New Structure** (Stage-5 WS):
```cuda
for (int t = 0; t < nTiles; ++t) {
    const int buf = t & 1;  // Double-buffer index (0 or 1)
    
#if USE_WARP_SPECIALIZATION
    // ====== PRODUCER WARPS ======
    if (is_producer) {
        // Wait for consumer to finish using this buffer
        if (lane == 0) {
            stage_spin_acquire(&kv_consumed[buf], 1);
        }
        __syncwarp();
        
        // Issue async copy for K/V tile → u8 staging buffer
        if (t < nTiles) {
            cp_async_tile_u8(t, buf);  // Existing lambda
        }
        __pipeline_wait_prior(0);  // Ensure visibility within block
        
        // Dequantize u8 → half (sK_u8[buf] → sKT, sV_u8[buf] → sV)
        // ... (move existing dequant code here, use buf index) ...
        
        // Signal that K/V tiles are ready
        if (lane == 0) {
            kv_consumed[buf] = 0;  // Mark buffer as in-use
            stage_store_release(&kv_ready[buf], 1);
        }
        __syncwarp();
    }
    
    // ====== CONSUMER WARPS ======
    else {
        // Wait for producer to finish K/V tile
        if (lane == 0) {
            stage_spin_acquire(&kv_ready[buf], 1);
        }
        __syncwarp();
        
        // Compute Q@K^T, softmax, P·V (unchanged)
        // ... existing compute code ...
        
        // Signal that buffer can be reused
        if (lane == 0) {
            kv_ready[buf] = 0;  // Mark tile as consumed
            stage_store_release(&kv_consumed[buf], 1);
        }
        __syncwarp();
    }
#else
    // Stage-2 behavior (existing code)
    // ... (keep current implementation) ...
#endif
}
```

---

### Step 5: Move Dequant Code to Producer Path (Medium, 1 hour)

**Current dequant loop** (find in existing code):
```cuda
// Existing vectorized u8→half dequantization
for (int lin = lane; lin < kv_len * D; lin += 32) {
    int n = lin / D;
    int d = lin % D;
    
    uint8_t ku = sK_u8[read_stage][n][d];
    uint8_t vu = sV_u8[read_stage][n][d];
    
    float kf = dequant_sim_fp8(ku, k_s);
    float vf = dequant_sim_fp8(vu, v_s);
    
    sKT[n][d] = __float2half(kf);
    sV[n][d]  = __float2half(vf);
}
```

**Change**: Move this into the **producer warp** block, update `read_stage` → `buf`

---

### Step 6: Keep Compute Code in Consumer Path (Easy, 30 min)

**Compute sections to preserve**:
1. WMMA Q@K^T → `sS` (scores)
2. Softmax (online softmax update, uses `m_smem`, `l_smem`)
3. WMMA P·V → `U_smem` (or scalar P·V if `USE_WMMA_PV=0`)

**No changes needed** — just ensure these sections are inside the `else` (consumer) block when `USE_WARP_SPECIALIZATION=1`.

---

### Step 7: Optional — Persistent CTAs (Medium, 1-2 hours)

**After** WS validation passes, add persistent CTA logic:

```cuda
#if USE_PERSISTENT_CTA
// Before main q_block processing
__shared__ int work_q_head;

if (tid == 0) {
    work_q_head = blockIdx.x;  // Seed with initial q_block
}
__syncthreads();

// Work queue loop (replaces single q_block)
while (true) {
    int my_q_block;
    if (tid == 0) {
        my_q_block = atomicAdd(&global_work_counter, 1);  // Need device-side counter
        work_q_head = my_q_block;
    }
    __syncthreads();
    my_q_block = work_q_head;
    
    if (my_q_block >= gridDim.x) break;  // No more work
    
    // Process q_block (existing code, replace blockIdx.x with my_q_block)
    // ...
}
#endif
```

**Note**: Requires global work counter (add as kernel parameter) and careful testing.

---

## ✅ Validation Checklist

### After Each Step
- [ ] Code compiles (no syntax errors)
- [ ] PTXAS passes (≤120 regs, ≤64 KB SMEM, 0 spills)

### After Full Implementation
- [ ] **Correctness**: `python scripts/bench_sdpa.py --shapes small` → PASS
- [ ] **Performance (small)**: Compare Stage-2 vs Stage-5 (should be similar or better)
- [ ] **Correctness (mission)**: `--shapes mission` → PASS (critical gate)
- [ ] **Performance (mission)**: p50 ≤ 590 μs (≥+10% vs 656 μs)

---

## 🐛 Debugging Tips

### Issue 1: Deadlock (kernel hangs)
**Symptom**: Kernel never returns  
**Cause**: Producer/consumer mismatch (flag never set/cleared)  
**Debug**:
```cuda
// Add timeout to spin loops
int spin_count = 0;
while (*f != expect && spin_count < 10000) {
    __nanosleep(64);
    spin_count++;
}
if (spin_count >= 10000) {
    printf("[DEADLOCK] warp=%d lane=%d flag=%d expect=%d\n", warp_id, lane, *f, expect);
}
```

### Issue 2: Correctness Failure
**Symptom**: `max_err > 0.06`  
**Cause**: Race condition (consumer reads before producer writes)  
**Debug**:
- Add `printf` after each `stage_store_release` / `stage_spin_acquire`
- Verify flags are set/cleared in correct order
- Check `__syncwarp()` placement (must be before/after shared memory access)

### Issue 3: Performance Regression
**Symptom**: Stage-5 slower than Stage-2  
**Cause**: Synchronization overhead dominates  
**Debug**:
- Profile with NCU: check for `__nanosleep` hotspots
- Try `cuda::barrier` instead of volatile flags (if available)
- Reduce `NUM_PRODUCER_WARPS` to 1

---

## 📊 Expected NCU Changes

### Stage-2 (Baseline)
```
sm__pipe_tensor_cycles_active:  ~45% (compute-bound)
smsp__cycles_active:             100%
sm__warps_active:                ~30% (moderate occupancy)
```

### Stage-5 (WS)
```
sm__pipe_tensor_cycles_active:  ~55% (MORE Tensor Core usage ✅)
smsp__cycles_active:             100%
sm__warps_active:                ~35% (better overlap)
dram__throughput:                ~40% (unchanged, still not memory-bound)
```

**Key Improvement**: Higher Tensor Core utilization without increasing memory pressure.

---

## 🚨 When to Abort

**If after 4 hours**:
- Deadlocks persist (can't fix spin logic)
- Correctness fails on small shape (fundamental bug)
- Performance regresses >5% (overhead dominates)

**Action**: Document as valid negative, revert to Stage-2.

---

## 🎯 Success Criteria

1. ✅ Code compiles with WS=1
2. ✅ PTXAS: ≤120 regs, ≤64 KB SMEM, 0 spills
3. ✅ Correctness: 6/6 tests (small/mission/long, seeds 0/1/2)
4. ✅ Performance: p50 ≤ 590 μs on mission shape (+10% vs 656 μs)
5. ✅ NCU: Tensor Core utilization ≥50%

**If all pass**: Merge to `main`, tag `v3.0-stage5-warp-spec` 🎉

---

## 📖 References

- **FlashAttention-2**: Producer/consumer warp specialization (Sec. 3.2)
- **CUTLASS**: Persistent kernels with work queues
- **CUDA Best Practices**: `__threadfence_block()` semantics (Sec. 5.4.4)

---

**Status**: Implementation guide complete ✅  
**Next**: Execute Steps 1-6 on L4 GPU (`cudadent42-l4-dev`)  
**Branch**: `feat/stage5-warp-spec-persistent`

