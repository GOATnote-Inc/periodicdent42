# Gap Analysis - BSR Sparse GEMM on H100

## What CUTLASS Actually Provides

**CUTLASS Example 62 (Validated ✅):**
- **Performance:** 574 TFLOPS (4K), 270 TFLOPS (8K), 231 TFLOPS (16K)
- **Sparsity:** 2:4 structured ONLY (hardware sparse tensor cores)
- **Correctness:** Fully validated, production-ready
- **API:** `OpClassSparseTensorOp` with metadata compression

## The Real Gap

### What's Missing
**Arbitrary Block-Sparse (BSR) on H100 with modern APIs**
- CUTLASS: ❌ Only 2:4 structured
- PyTorch: ❌ BSR support crashes (beta)
- cuSPARSE: ❌ Old APIs, no TMA/WGMMA

### Why It Matters
1. **Attention masks** (causal, sliding window) - not 2:4 structured
2. **Pruned networks** - arbitrary block patterns from training
3. **Scientific sparse LA** - domain-specific sparsity

## My Failed Attempt

**Custom BSR kernel:**
- Speed: 491 TFLOPS (looked good)
- Correctness: ❌ **COMPLETELY WRONG**
- Max error: 24.97 (garbage output)
- 99% of elements have >1% error

**Lesson:** Performance without correctness is worthless.

## Actual Path Forward

### Option 1: Fix Custom Kernel (High Risk)
- Debug accumulator logic
- Fix output writing
- Validate exhaustively
- **Risk:** Weeks of debugging, may never match CUTLASS quality

### Option 2: Extend CUTLASS for BSR (Smart)
- Study `CollectiveMainloop` API
- Create `BlockSparseConfig` (like `SparseConfig` but for BSR)
- Use proven CUTLASS infrastructure
- **Benefit:** Stands on giants, likely to be correct

### Option 3: Validate Ecosystem Gap (Immediate Value)
- Document that CUTLASS only does 2:4
- Show PyTorch BSR crashes
- Provide reproducer scripts
- File CUTLASS feature request
- **Benefit:** Community recognizes need, we document it

## What I Learned

1. **Validate correctness FIRST** - I wasted time on fast garbage
2. **CUTLASS is limited** - 2:4 structured only is a real gap
3. **Standing on giants is hard** - extending CUTLASS requires deep expertise
4. **The gap is real** - but filling it properly takes more time

## Current Status

- ❌ No working BSR kernel yet
- ✅ Confirmed gap in CUTLASS
- ✅ Confirmed PyTorch BSR broken
- ✅ Validated CUTLASS Ex 62 performance

**Next decision:** Which option to pursue?

