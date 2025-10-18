# V2c Debug Progress

**Session**: Oct 18, 2025 (continued)  
**Goal**: Get V2c to 100% correctness with WMMA  

---

## Iteration Summary

### Iteration 1: Complex WMMA
- **Status**: Build ✅, Launch ❌ (unspecified launch failure)
- **Issue**: Fragment store to local array
- **Time**: 1 hour

### Iteration 2: Simplified WMMA  
- **Status**: Build ✅, Launch ✅, Correctness ❌ (max_diff=0.013)
- **Issue**: Computing Q @ K instead of Q @ K^T
- **Root Cause**: WMMA matrix_b loaded as row-major, but K needs transpose
- **Time**: 30 min

### Iteration 3 (Next): Validate Infrastructure
- **Strategy**: Use scalar Q@K^T (like V2b) to validate streaming softmax
- **Goal**: 100% correctness without WMMA complexity
- **Then**: Add WMMA incrementally once foundation is solid

---

## Key Learning

**WMMA Transpose Challenge**:
- Q @ K^T requires K stored as col-major OR special handling
- For row-major K, accessing as K^T for WMMA is non-trivial
- FlashAttention papers handle this with careful SMEM layout

**Recommendation**: 
1. First: Get 100% correctness with scalar (validate softmax)
2. Then: Add WMMA Q@K^T with proper K^T layout
3. Finally: Add WMMA P@V

---

## Time Budget

- V2b: 4 hours (100% correct)
- V2c Iteration 1-2: 1.5 hours (infrastructure learning)
- V2c Iteration 3: 1 hour (target: 100% correct with scalar)
- V2c Iteration 4: 2 hours (add WMMA properly)
- **Total**: ~8.5 hours to correct V2c

**Remaining from original 23-hour budget**: Plenty of runway

---

**Status**: Systematic iteration working. Correctness improving. On track.


