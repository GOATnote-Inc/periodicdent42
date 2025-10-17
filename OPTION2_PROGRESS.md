# Option 2 Progress: TC Implementation

**Started**: Oct 17, 2025 (1:57 AM)  
**Status**: Phase A (Debug CUTLASS) - **IN PROGRESS**

---

## ✅ **Completed**

### **1) CUTLASS Root Cause Found**
**Problem**: TensorOp config with Sm80/sm_89 causing "Error Internal"

**Test**: Created minimal CUTLASS GEMM (no TensorOp, 16×16)
- ✅ Compiles
- ✅ can_implement: Success
- ✅ initialize: Success  
- ✅ launch: **SUCCESS** (C[0,0]=0.159922)

**Conclusion**: Basic CUTLASS works. TensorOp config needs fixing.

**File**: `bench/cutlass/cutlass_simple_gemm.cu`

---

### **2) M=64 "Hang" Diagnosed**
**Problem**: M=64 times out in Evo sweep

**Test**: Added timeout diagnostic (`scripts/test_m64_hang.sh`)
- ❌ NOT a hang - runs in 1140 μs
- ❌ 35% SLOWER than M=32 (839 μs)

**Root Cause**: **SMEM Overflow**
```
M=64: 57,376 bytes SMEM (exceeds 48 KB L4 limit)
M=32: ~28,000 bytes SMEM (within limit)
```

**Why Slower**:
- Spills to local memory (global memory)
- Bank conflicts worse with larger tiles
- Register pressure (46 regs)

**Prevention**:
1. Add SMEM size guard in build script
2. Use `half` instead of `float` for intermediate buffers
3. Limit Evo search to M ≤ 48

**Files**: `scripts/test_m64_hang.sh`, ptxas output

---

## 🔄 **In Progress**

### **3) Fix CUTLASS TensorOp Config**
**Goal**: Get TC-enabled GEMM working

**Approach**:
1. Try DefaultGemm with TensorOp explicitly
2. Use known-working CUTLASS example as template
3. Match tile sizes to TC requirements (multiples of 16)

**Next Step**: Create `cutlass_tensor_gemm.cu` with correct TensorOp config

---

## 📊 **Performance Baseline (Confirmed)**

| Config | Time (μs) | SMEM (bytes) | Status |
|--------|-----------|--------------|--------|
| M=32, W=8 | **839** | ~28,000 | ✅ **BEST** |
| M=32, W=4 | 1,030 | ~28,000 | ✅ Baseline |
| M=64, W=4 | 1,140 | 57,376 | ⚠️ SMEM overflow |

---

## 🎯 **Next Actions**

1. **Fix CUTLASS TensorOp** (1-2 hours)
   - Create working TC GEMM baseline
   - Benchmark Q@K^T with TC vs scalar
   
2. **Integrate TC into Phase 4** (2-3 hours)
   - Hybrid kernel (TC Q@K^T, scalar P@V)
   - Validate correctness
   - Measure speedup

3. **Full TC Pipeline** (3-4 hours)
   - Add P@V TC path
   - Evo sweep on TC configs
   - Document final results

---

## 📝 **Key Learnings**

### **CUTLASS**
- ✅ Basic CUTLASS works on sm_89
- ❌ TensorOp config needs careful matching
- 💡 Start simple, add complexity incrementally

### **M=64**
- ❌ NOT a hang, just slow (SMEM overflow)
- 💡 Always check `ptxas -v` for resource usage
- 💡 L4 has 48KB SMEM limit (not 64KB like A100)

### **General**
- ✅ Systematic debugging works (minimal test → isolate issue)
- ✅ Measure don't guess (ptxas, timeouts, NCU)
- 💡 Document findings immediately (this file!)

---

**Commit**: `813e3c8`  
**Time Invested**: 1 hour  
**Status**: On track for Phase A completion

