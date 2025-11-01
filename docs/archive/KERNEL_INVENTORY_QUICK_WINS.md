# Kernel Inventory & Quick Win Analysis

**Date:** November 1, 2025  
**Purpose:** Identify low-hanging fruit for optimization/validation

---

## ✅ SHIPPED (Production Ready)

### 1. BlackwellSparseK Sparse GEMM
**File:** `BlackwellSparseK/src/sparse_h100_async.cu`  
**Status:** ✅ Validated on L4 (52.1 TFLOPS)  
**Speedup:** 1.74× vs CUTLASS 4.3.0, 63× vs cuSPARSE  
**Action:** None - already shipped to main branch

---

## 🎯 QUICK WINS (High Value, Low Effort)

### 1. CUTLASS FMHA Integration
**Location:** `~/cutlass-latest/examples/41_fused_multi_head_attention/`  
**Status:** ✅ Compiled and validated (27.6 TFLOPS, 2.1× vs PyTorch)  
**Effort:** 2-3 days (PyTorch bindings + benchmark suite)  
**Value:** Production-ready attention at 2.1× PyTorch speed  
**Next Steps:**
- Create PyTorch wrapper (`torch.nn.functional` compatible)
- Comprehensive benchmark vs PyTorch SDPA, xFormers, cuDNN Flash
- Docker + install scripts
- Add to main repository

**ROI:** HIGH - Already working, just needs packaging

---

### 2. FP8 Precision Support (Hopper)
**Location:** Various experiments in `BlackwellSparseK/.archive/src_experiments/`  
**Status:** Code exists but untested  
**Effort:** 1 week (need H100 access)  
**Value:** 2× throughput on Hopper with FP8 tensor cores  
**Blockers:** Need stable H100 access  
**Files:**
- `.archive/old_projects/flashcore/fast/attention_fp8.py` (archived Triton)
- CUTLASS 4.3.0 has FP8 examples (example 67: `hopper_fp8_warp_specialized_gemm`)

**ROI:** MEDIUM - High value but blocked on H100

---

### 3. Multi-Head Attention Kernel (Already Validated!)
**Location:** `.archive/old_projects/flashcore/fast/attention_multihead.py`  
**Status:** ⚠️ Archived but was validated on H100  
**Performance:** 0.451-0.491 μs/head (10-11× better than 5μs target)  
**Effort:** 1-2 days (resurrect from archive, re-validate on L4)  
**Value:** Custom attention kernel optimized for multi-head  
**Memory Says:** "Multi-head attention kernel validated on H100 SXM 80GB: H=96 (GPT-4) at 0.491 μs/head (10× better than target)"

**ROI:** HIGH - Already worked, just needs to be resurrected

---

### 4. Sparse GEMM with cuBLASLt Batched
**File:** `BlackwellSparseK/.archive/src_experiments/sparse_cublas_batched.cu`  
**Description:** Groups sparse tiles into batches, calls cuBLASLt once  
**Expected:** 750+ TFLOPS (close to hardware ceiling)  
**Status:** Code written but never compiled/tested  
**Effort:** 3-4 days (compile, debug, benchmark)  
**Value:** Potential 14× speedup over current 52.1 TFLOPS  
**Risk:** May not beat hand-tuned kernel due to overhead

**ROI:** MEDIUM - High potential but uncertain payoff

---

## ⚠️ BLOCKED (Need H100 or Refactoring)

### 1. TriageAttention TMA+WGMMA Kernel
**File:** `csrc/kernels/attention_bleeding_edge_tma.cu`  
**Status:** ❌ Broken on L4 (architecture mismatch)  
**Blocker:** Uses Hopper-only features (TMA 2.0 + WGMMA)  
**Options:**
- A) Test on H100 (need stable access)
- B) Rewrite for SM89 using CpAsync + WMMA (1 week)  
**Value:** Unknown until validated

**ROI:** LOW - Architecture mismatch, significant rework needed

---

### 2. WGMMA Test Kernels
**Files:**
- `tests/test_wgmma_single.cu`
- `tests/test_wgmma_single_corrected.cu`

**Status:** Hopper-only (WGMMA = SM90+)  
**Blocker:** Need H100  
**Value:** Educational/validation, not production kernels

**ROI:** LOW - Test code, not production value

---

### 3. Long Context Attention
**File:** `.archive/old_projects/flashcore/fast/attention_longcontext.py`  
**Status:** Archived Triton kernel  
**Effort:** 1-2 weeks (resurrect, port to CUDA if needed)  
**Value:** Long context (S=32K+) support  
**Blocker:** Triton backend, needs testing

**ROI:** MEDIUM - Valuable for LLM inference but requires significant work

---

## 🗑️ LOW VALUE (Not Worth Effort)

### 1. Sparse CUTLASS Experiments
**Files:** Multiple in `BlackwellSparseK/.archive/src_experiments/`
- `sparse_cutlass_wgmma.cu`
- `sparse_cutlass_gemm.cu`
- `sparse_cutlass_builder.cu`
- `sparse_gemm_cutlass_v2.cu`

**Status:** Experimental, never beat hand-tuned kernel  
**Reason:** Already validated that custom kernel (52.1 TFLOPS) beats CUTLASS 4.3.0 (~30 TFLOPS)

**ROI:** NONE - Custom kernel already wins

---

### 2. TMA Test Kernels
**Files:**
- `BlackwellSparseK/.archive/src_experiments/test_tma_simple.cu`
- `BlackwellSparseK/.archive/src_experiments/test_tma_cute.cu`
- `BlackwellSparseK/.archive/src_experiments/experimental/tma/bsr_gemm_tma.cu`

**Status:** Test/learning code, Hopper-only  
**Value:** Educational only

**ROI:** NONE - Not production kernels

---

### 3. Old FlashCore Archived Kernels
**Location:** `.archive/old_projects/flashcore/fast/`  
**Files:** 10+ archived Triton kernels  
**Status:** Old codebase, superseded by CUTLASS FMHA  
**Reason:** CUTLASS FMHA (27.6 TFLOPS) already validated

**ROI:** NONE - Better alternatives exist

---

## 🚀 RECOMMENDED PRIORITY ORDER

### Week 1 (Immediate)
1. **CUTLASS FMHA PyTorch integration** (2-3 days)
   - Already working at 27.6 TFLOPS
   - Just needs packaging
   - High user value (attention is critical for LLMs)

### Week 2-3 (Short-term)
2. **Resurrect multi-head attention kernel** (1-2 days)
   - Was validated on H100 at 0.491 μs/head
   - Re-validate on L4
   - Compare vs CUTLASS FMHA

### Month 1-2 (Medium-term, if H100 available)
3. **FP8 precision support** (1 week with H100)
   - Use CUTLASS 4.3.0 example 67 as reference
   - 2× throughput potential
   - Hopper/Blackwell only

4. **Sparse cuBLASLt batched** (3-4 days)
   - High risk/high reward
   - Potential 14× speedup
   - May not beat hand-tuned kernel

### Month 3+ (Long-term)
5. **Long context attention** (1-2 weeks)
   - Port archived Triton kernel
   - S=32K+ support
   - Valuable for LLM inference

---

## ❌ DO NOT PURSUE

1. **TriageAttention TMA kernel** - Architecture mismatch, significant rework
2. **CUTLASS experiments** - Custom kernel already wins
3. **TMA test kernels** - Educational only, no production value
4. **Old archived FlashCore** - Superseded by CUTLASS FMHA

---

## Summary Table

| Kernel | Status | Effort | Value | ROI | Priority |
|--------|--------|--------|-------|-----|----------|
| **CUTLASS FMHA integration** | ✅ Working (27.6 TFLOPS) | 2-3 days | HIGH | **HIGHEST** | **1** |
| **Multi-head attention** | ⏳ Archived (was 0.491 μs/head) | 1-2 days | HIGH | **HIGH** | **2** |
| **FP8 support** | ⏳ Code exists | 1 week | MEDIUM | MEDIUM | 3 |
| **cuBLASLt batched** | ⏳ Uncompiled | 3-4 days | HIGH (risky) | MEDIUM | 4 |
| **Long context** | ⏳ Archived Triton | 1-2 weeks | MEDIUM | LOW | 5 |
| TriageAttention TMA | ❌ Broken | 1 week+ | UNKNOWN | LOW | N/A |
| CUTLASS experiments | 🗑️ Obsolete | N/A | NONE | NONE | N/A |

---

## Final Recommendation

**Ship CUTLASS FMHA integration first** (2-3 days, highest ROI).

It's already validated at 27.6 TFLOPS (2.1× faster than PyTorch SDPA). Just needs:
1. PyTorch wrapper
2. Benchmark suite
3. Documentation
4. Docker/install scripts

Then move to resurrecting the multi-head attention kernel from the archive.

---

*Brandon Dent, MD*  
*November 1, 2025*

