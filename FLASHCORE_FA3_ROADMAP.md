# FlashCore FA-3: Roadmap to <40 μs

**Date**: October 22, 2025  
**Current Status**: Correctness ✅, Performance ❌ (2812 μs)  
**Target**: <40 μs (beat PyTorch SDPA's 44 μs)

---

## ✅ **Progress So Far**

### **Simplified Kernel (Current)**
- **Correctness**: ✅ PERFECT (error 0.000244)
- **Performance**: 2812 μs
- **Registers**: 37
- **Issue**: Scalar dot products (no Tensor Cores)

**This proves**: Online softmax algorithm is correct!

---

## 🎯 **Optimization Plan**

### **Step 2: WMMA for Tensor Cores** ← **NEXT** (Expected: 2812 → 200-300 μs)

**Why**: Tensor Cores are 10-20× faster than scalar ops for matrix multiply

**Changes**:
1. Replace scalar Q·K^T with WMMA 16×16×16 tiles
2. Keep online softmax in registers (already working)
3. Use WMMA for P·V as well

**Expected**: ~200-300 μs (10× faster)

---

### **Step 3: Reduce Synchronization** (Expected: 200-300 → 100-150 μs)

**Why**: Too many `__syncthreads()` hurt performance

**Changes**:
1. Load K/V cooperatively but sync less often
2. Use warp-level sync where possible
3. Overlap loads with compute

**Expected**: ~100-150 μs (2× faster)

---

### **Step 4: Tune Tile Sizes** (Expected: 100-150 → 50-80 μs)

**Why**: Larger tiles amortize overhead

**Changes**:
1. Try M_TILE=128, N_TILE=128
2. Ensure enough shared memory
3. Profile with NCU to find sweet spot

**Expected**: ~50-80 μs (2× faster)

---

### **Step 5: Micro-optimizations** (Expected: 50-80 → <40 μs)

**Why**: Last 20-30% needs fine-tuning

**Changes**:
1. Vectorized loads (float4)
2. Loop unrolling
3. Instruction-level optimizations
4. Maybe add double-buffering if needed

**Expected**: <40 μs ✅

---

## 📊 **Performance Targets**

| Step | Description | Expected Latency | vs PyTorch (44 μs) |
|------|-------------|------------------|-------------------|
| **Current** | Simple kernel | 2812 μs | 64× slower ❌ |
| **Step 2** | + WMMA | 200-300 μs | 5-7× slower ⚠️ |
| **Step 3** | + Less sync | 100-150 μs | 2-3× slower ⚠️ |
| **Step 4** | + Tune tiles | 50-80 μs | 1.1-1.8× slower ⚠️ |
| **Step 5** | + Micro-opts | **<40 μs** | **Faster!** ✅ |

---

## 💡 **Key Insights**

1. ✅ **Online softmax algorithm is correct** (proven with simple kernel)
2. 🎯 **Tensor Cores are the key** (10-20× speedup expected)
3. ⏱️ **Synchronization overhead matters** (reduce `__syncthreads()`)
4. 📏 **Tile size tuning is critical** (larger = better amortization)
5. 🔧 **Micro-optimizations for last mile** (vectorization, unrolling)

---

## 🚀 **Next Action**

Implement **WMMA version** of the kernel:
- Use proven patterns from our earlier WMMA kernels
- Keep the working online softmax logic
- Target: 200-300 μs (10× faster than simple kernel)

**Timeline**: 1-2 hours to implement and test

---

**Status**: Ready to add WMMA and achieve 10× speedup! 🚀

