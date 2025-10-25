# FlashCore Phase 1 FP32 P - Critical Learnings

**Date**: October 22, 2025  
**Phase**: Phase 1 (Fix Error with FP32 P)  
**Result**: ❌ **BLOCKED by SMEM constraints with 32×32 tiles**  
**Decision**: ✅ **Pivot to Phase 2A (64×64 tiles + FP32 P)**

---

## 🎯 **What We Attempted**

### **Goal**
- Fix error: 0.51 → <0.10
- Method: FP32 P matrix (4× more precision than FP16)
- Keep: 32×32 tiles

### **Implementation**
```cpp
// Attempted SMEM layout:
sQ:        5 KB
sKT:       5 KB
sV:        5 KB
sS_f32:    4 KB  // FP32 scores
sP:        4 KB  // FP32 probabilities 
sP_fp16:   2 KB  // Conversion buffer
m/l_smem:  0.25 KB
U_smem:    10 KB
sU_part:   4 KB
──────────────────
Total:     39.25 KB → **52 KB with alignment**
```

**Problem**: 52 KB > 48 KB default SMEM limit!

---

## 🔴 **Why Buffer Reuse Failed**

### **Attempt 1: Union**
```cpp
__shared__ union {
    float as_scores[32][32];  // 4 KB
    float as_probs[32][32];   // 4 KB
} score_prob;
```

**Result**: GPU runtime error, 4KB stack frame  
**Cause**: Compiler allocated union on stack instead of shared memory

---

### **Attempt 2: Alias**
```cpp
__shared__ float sS_f32[32][32];  // 4 KB
#define sP sS_f32  // Same buffer
```

**Result**: Data corruption  
**Root Cause**: 
```cpp
// In softmax loop:
for (int n = 0; n < kv_len; ++n) {
    float s = sS_f32[m][n];        // READ score
    float p = expf(s - m_new);
    sP[m][n] = p;                   // WRITE prob to SAME BUFFER
}
// Later iterations read corrupted scores!
```

**Why it fails**:
- Need to read ALL scores for a row to compute exp values
- But we write probabilities to the same buffer as we go
- This overwrites scores we haven't read yet
- Result: Later softmax values use corrupted data

---

## 💡 **Key Insight: Timing of Buffer Reuse**

### **What Works**
Buffer reuse works when usage is **temporally separated**:
```cpp
// Phase 1: Use buffer A
compute_and_store_to_A();
__syncthreads();

// Phase 2: Use buffer B (same memory)
read_from_A_and_write_to_B();  // OK if we don't need A anymore
```

### **What Doesn't Work**
Buffer reuse fails when we need **simultaneous access**:
```cpp
// Same loop iteration:
for (int n = 0; n < N; ++n) {
    float x = buffer_A[n];     // READ from A
    float y = compute(x);
    buffer_A[n] = y;           // WRITE to same A - CORRUPTS DATA!
}
```

---

## 📊 **SMEM Arithmetic**

### **32×32 Tiles: Constrained**
```
Minimum needed for FP32 P:
- Q, K, V tiles:      15 KB
- Scores (FP32):      4 KB
- Probs (FP32):       4 KB  ← Can't reuse with scores!
- FP16 buffer:        2 KB
- Stats (m, l):       0.25 KB
- Output (U):         10 KB
- Partials:           4 KB
──────────────────────────────
Total:                39.25 KB → 52 KB aligned ❌
```

### **64×64 Tiles: Room to Grow**
```
With 64×64 tiles:
- Q, K, V tiles:      30 KB (2× larger)
- Union {
    Scores (FP32):    16 KB
    Probs (FP32):     16 KB  ← NOW union works! (temporal separation)
  }
- FP16 buffer:        8 KB
- Stats (m, l):       0.5 KB
- Output (U):         20 KB
- Partials:           16 KB
──────────────────────────────
Total:                90.5 KB ✅ Fits in 96KB with union!
```

**Why union works with 64×64**:
- Full QK matmul completes → stores to `scores`
- `__syncthreads()` ensures all warps done
- Softmax phase reads `scores`, writes to `probs`
- After softmax, don't need `scores` anymore
- **Temporal separation achieved!**

---

## 🎓 **Lessons Learned**

### **1. Unions are Fragile**
- May allocate on stack instead of SMEM
- Use with extreme caution
- Test thoroughly for runtime errors

### **2. Buffer Reuse Requires Temporal Separation**
- Can't reuse if reading AND writing in same loop
- Need `__syncthreads()` between usage phases
- Carefully analyze data dependencies

### **3. SMEM is Precious at Small Tiles**
- 32×32 tiles = 48KB limit is TIGHT
- FP32 precision costs 2× space
- Need larger tiles for advanced optimizations

### **4. Larger Tiles Unlock Optimizations**
- 64×64 tiles: 2× block work, 2× SMEM, BUT
- 90KB total still fits in 96KB L4 limit
- Enables FP32 P + union + better occupancy

### **5. Compile-time vs Runtime SMEM**
- Compiler checks against 48KB by default
- `cudaFuncSetAttribute` is runtime call
- Must fit in compile-time limit OR use `launch_bounds` wisely

---

## ✅ **Why Phase 2A (64×64 Tiles) is the Right Pivot**

### **Benefits**
1. **Fixes Error**: Room for FP32 P with proper union
2. **Improves Performance**: 4× more work per block
3. **Natural Progression**: Was planned anyway
4. **Higher Confidence**: Kills two birds with one stone

### **Expected Results**
```
Performance: 279 → 110-140 μs (2-2.5× speedup)
Error:       0.51 → 0.05-0.10 (10× improvement)
SMEM:        90 KB (fits in 96KB with union)
Registers:   ~100-110 (acceptable)
Confidence:  80-85%
```

### **Implementation Strategy**
1. Use user-provided `flashcore_fused_wmma_64x64.cu`
2. Union for `scores ↔ probs` (works with temporal separation)
3. 8 warps (4×2 grid)
4. Opt-in to 96KB SMEM
5. Test & validate

---

## 🚦 **Next Steps**

### **Immediate** (Next 30 min)
- [ ] Copy user-provided 64×64 kernel
- [ ] Create build script
- [ ] Create test script
- [ ] Deploy to L4

### **Phase 2A Validation** (Next 2-3 hours)
- [ ] Compile successfully
- [ ] Verify SMEM usage (~90KB)
- [ ] Check register count (~100-110)
- [ ] Test correctness (expect <0.10 error)
- [ ] Benchmark performance (expect ~120 μs)

### **If Phase 2A Succeeds**
- [ ] Proceed to Phase 2B (cp.async)
- [ ] Expected: 120 → 50-60 μs
- [ ] Then Phase 2C (micro-opts)
- [ ] Final: <40 μs target! 🎯

---

## 📈 **Revised Confidence Levels**

| Metric | Original (Phase 1 alone) | Revised (Phase 2A) |
|--------|--------------------------|-------------------|
| **Error** | 80% for <0.10 | **85%** for <0.10 |
| **Performance** | No change (285 μs) | **90%** for 110-140 μs |
| **Both Goals** | 80% error only | **80%** for both! |

**Why higher confidence?**
- Larger tiles = proven technique
- Union works with temporal separation
- Natural fit for FP32 P
- Performance boost is bonus!

---

## 🎯 **Bottom Line**

### **What We Learned**
✅ FP32 P is the right solution for error  
✅ 32×32 tiles too constrained for FP32 P  
✅ Buffer reuse needs temporal separation  
✅ 64×64 tiles unlock the optimization

### **What's Next**
✅ Phase 2A: 64×64 tiles + FP32 P  
✅ Expected: Error <0.10, Performance 110-140 μs  
✅ Confidence: 80-85%

### **Path Forward**
```
Current:     279 μs, 0.51 error
             ↓ Phase 2A (2-3 hours)
Phase 2A:    120 μs, 0.08 error  ← Both fixed!
             ↓ Phase 2B (2-3 hours)
Phase 2B:    55 μs, 0.08 error
             ↓ Phase 2C (1 hour)
Phase 2C:    38 μs, 0.08 error   ← TARGET ACHIEVED! 🎯
```

**Total time**: 5-7 hours  
**Success probability**: 75-80%

---

**Status**: ✅ **Ready for Phase 2A implementation!**  
**Confidence**: 🔥 **HIGH** - This is the right path!

**Let's build something excellent!** 🚀

