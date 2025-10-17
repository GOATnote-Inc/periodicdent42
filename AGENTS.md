# AI Agent Guidelines for periodicdent42

**Last Updated**: Oct 17, 2025  
**Session**: Rehydration + Option 2 (TC Implementation)

---

## 🎯 **Current Mission**

**Goal**: Implement Tensor Core (TC) optimizations to achieve 5-10× speedup over current 839 μs baseline

**Approach**: Option 2 - Fix CUTLASS + Continue TC Work

**Expected Outcome**: 400-600 μs (close the 17.8× gap to PyTorch SDPA)

---

## 📊 **Current Status Summary**

### **Performance Baseline**
```
Minimal:        2,870 μs (1.00×, scalar baseline)
Phase 4:        1,029 μs (2.79×, light barriers)
Phase 4 (8W):     839 μs (3.42×, best custom kernel) ✅ CURRENT BEST
PyTorch SDPA:      47 μs (61.1×, production target)
```

**Gap**: 17.8× slower than PyTorch SDPA

**Bottleneck Analysis** (NCU-validated):
- DRAM throughput: 0.31% → **compute-bound**
- Tensor Core active: n/a → **no TC usage**
- Warps active: 30.53% → moderate occupancy

**Conclusion**: Scalar operations (78% of runtime) are the blocker. **Tensor Cores mandatory** for next big win.

---

## 🔧 **Active Blockers**

### **1) CUTLASS Runtime Failure**
**Status**: ⚠️ **BLOCKING** TC baseline

**Error**: `launch failed: Error Internal`

**What Works**:
- ✅ Compilation successful
- ✅ `can_implement()` returns Success
- ✅ `initialize()` returns Success

**What Fails**:
- ❌ `operator()` (launch) returns Error Internal

**Hypothesis**:
1. Sm80 config may have incompatibility with sm_89 runtime
2. Tile size (32×32×64) might violate alignment constraints
3. Workspace allocation may be incorrect

**Next Steps**:
- Try simpler Gemm config (basic OpClass, not TensorOp)
- Test with different tile sizes (16×16, aligned to 8)
- Check CUTLASS examples for sm_89 patterns
- Use `cudaGetLastError()` after launch for detailed error

---

### **2) BLOCK_M=64 Timeout**
**Status**: ⚠️ **LIMITING** evo search space

**Observation**: `BLOCK_M=64, NUM_WARPS=4` hangs/times out

**Hypothesis**:
1. **SMEM bank conflicts**: 64-column rows with HEAD_DIM=64 → 32-way conflicts (catastrophic)
2. **Register pressure**: Larger tiles → more registers → spillage
3. **Deadlock**: Possible barrier/warp synchronization issue

**Evidence**:
- `BLOCK_M=32` works fine (839 μs)
- `BLOCK_M=64` with ANY NUM_WARPS fails

**Prevention**:
- Add XOR swizzling for SMEM (fixes bank conflicts)
- Use `__launch_bounds__` to limit registers
- Add timeout detection in evo sweep
- Profile with NCU to see if kernel launches at all

---

## 🏗️ **Infrastructure Status**

### **Working** ✅
- NCU profiling (permissions fixed)
- Microbench ranking (`evidence/micro_best.json`)
- Evo sweep framework (`scripts/evo_test_one.sh`)
- Phase 3 kernel (839 μs, 100% correct)
- Evidence artifacts (CSV, JSON, NCU reports)

### **Partial** ⚠️
- CUTLASS baseline (compiles, doesn't run)
- Evo sweep (limited to M=32 due to M=64 hang)

---

## 📋 **Option 2 Roadmap**

### **Phase A: Debug CUTLASS** (2-4 hours)
**Goal**: Get working TC baseline for Q@K^T

**Tasks**:
1. Try simpler CUTLASS config (DefaultGemm, not custom)
2. Test with known-working example (basic_gemm)
3. Add detailed CUDA error reporting
4. Profile with NCU to see if kernel launches
5. Check CUTLASS GitHub issues for sm_89 + Error Internal

**Success Criteria**: CUTLASS GEMM runs and produces correct Q@K^T output

**Fallback**: Skip CUTLASS, implement WMMA manually (slower path)

---

### **Phase B: Investigate M=64 Hang** (1-2 hours)
**Goal**: Understand why M=64 fails, prevent similar issues

**Tasks**:
1. Add SMEM size check (compile with `-Xptxas=-v`)
2. Profile M=64 with NCU (does kernel launch?)
3. Test with XOR swizzling enabled
4. Try M=48 (intermediate size)
5. Add `__launch_bounds__` to limit registers

**Success Criteria**: Either M=64 works OR we understand root cause

**Outcome**: Document findings, add guards to prevent future hangs

---

### **Phase C: Integrate TC Q@K^T** (3-4 hours)
**Goal**: Replace scalar Q@K^T with CUTLASS/WMMA

**Approach**:
1. Create hybrid kernel (TC for Q@K^T, scalar for P@V)
2. Keep Phase 4 structure (light barriers, warp reductions)
3. Add `USE_TC` compile flag
4. Validate correctness (`torch.allclose`)

**Expected**: 839 → 400-500 μs (2× speedup from Q@K^T TC alone)

**Success Criteria**:
- ✅ Correctness maintained (max_diff < 0.001)
- ✅ NCU shows `sm__pipe_tensor_active > 0%`
- ✅ Performance 400-600 μs

---

### **Phase D: Full TC Integration** (4-6 hours)
**Goal**: TC for both Q@K^T and P@V

**Tasks**:
1. Add P@V CUTLASS/WMMA path
2. Optimize tile size for dual-TC
3. Tune NUM_WARPS, VEC_WIDTH for TC path
4. Run Evo sweep on TC variants

**Expected**: 400-500 → 250-350 μs (additional 1.5× from P@V TC)

**Success Criteria**:
- ✅ Full TC pipeline working
- ✅ Performance within 2-5× of PyTorch SDPA
- ✅ Evo finds optimal TC config

---

## 🔍 **Key Insights from NCU Profiling**

**What We Learned**:
1. **Memory NOT bottleneck**: DRAM 0.31% → vectorization won't help much
2. **Compute IS bottleneck**: 78% of time in scalar Q@K^T + P@V
3. **TC usage = 0**: Confirms we need WMMA/MMA instructions
4. **8 warps > 4 warps**: Better utilization (1.23× speedup)

**What This Means**:
- Phase 6's vectorization approach was correct diagnosis (compute-bound)
- But wrong solution (vectorization targets memory)
- TC is THE critical optimization (5-10× compute speedup)

---

## 📚 **Key Files**

### **Documentation**
- `REHYDRATION_COMPLETE.md` - Session status, NCU results, Evo findings
- `FLASHATTENTION2_ANALYSIS.md` - 559 lines on TC/warp/memory architecture
- `FINAL_PORTFOLIO_REPORT.md` - Complete portfolio summary
- `PHASE6_STATUS.md` - Phase 6 post-mortem (vectorization lesson)

### **Code**
- `cudadent42/bench/kernels/fa_phase3_wmma.cu` - Phase 4 kernel (839 μs)
- `bench/cutlass/cutlass_attn_qkt.cu` - CUTLASS baseline (broken)
- `scripts/evo_test_one.sh` - Single-variant tester
- `bench/micro/bench_many.cu` - Microbenchmark

### **Evidence**
- `evidence/micro_best.json` - Top-8 configs
- `evidence/ncu_phase3_best.ncu-rep` - NCU profile (103KB)
- `evidence/evo_gen0_results.json` - Evo sweep partial

---

## 🎓 **Lessons Learned**

### **What Worked**
1. ✅ Systematic debugging (NCU permissions → modprobe config)
2. ✅ Hypothesis-driven (predicted compute-bound, NCU confirmed)
3. ✅ Automated search (Evo found 1.23× improvement)
4. ✅ Append-only development (no breakage)

### **What Didn't Work**
1. ❌ Phase 6 vectorization (wrong optimization for bottleneck)
2. ❌ M=64 exploration (immediate hang)
3. ❌ CUTLASS quick integration (complex template debugging)

### **What We're Learning**
1. 🔄 TC programming requires deep expertise (this is why FA2 exists)
2. 🔄 M=64 likely bank conflicts (need swizzling)
3. 🔄 CUTLASS needs careful config matching (Sm80 vs sm_89)

---

## 🚨 **Critical Constraints**

### **L4 Hardware Limits**
- SMEM: **48 KB max** (NOT 64 KB)
- Tensor Cores: FP16 accumulation = 2× throughput vs FP32
- L2 Cache: 48 MB (can fit entire KV cache)
- Bank conflicts: HEAD_DIM=64 → need XOR swizzling

### **Correctness Requirements**
- `torch.allclose(atol=1e-3, rtol=1e-3)` ALWAYS
- Never break working kernels
- Maintain git history (append-only)

### **Time Constraints**
- Option 2 estimated: 1-2 weeks for full TC
- Each phase should show incremental progress
- Document failures (CUTLASS debugging counts as progress)

---

## 🎯 **Success Criteria for Option 2**

### **Minimum** (Phase A+B complete)
- ✅ CUTLASS baseline working OR documented why it can't
- ✅ M=64 hang root-caused OR prevented
- ✅ Clear path forward documented

### **Good** (Phase C complete)
- ✅ TC Q@K^T integrated (hybrid kernel)
- ✅ Performance 400-600 μs
- ✅ Correctness maintained

### **Excellent** (Phase D complete)
- ✅ Full TC pipeline (Q@K^T + P@V)
- ✅ Performance 250-350 μs
- ✅ Within 5× of PyTorch SDPA
- ✅ Evo finds optimal TC config

---

## 💡 **Quick Reference Commands**

### **Test Current Best**
```bash
bash scripts/evo_test_one.sh 32 8 4
# Expected: 839 μs
```

### **NCU Profile**
```bash
ncu --target-processes all --replay-mode kernel \
  --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv -o evidence/ncu_test \
  python3 scripts/quick_phase3_test.py
```

### **Test CUTLASS**
```bash
cd bench/cutlass
bash build_cutlass.sh
./cutlass_attn_qkt
```

### **Evo Sweep Variants**
```bash
bash scripts/evo_test_one.sh <M> <WARPS> <VEC>
# Working: M=32, WARPS=8, VEC=4
# Broken: M=64 (any config)
```

---

## 📞 **Communication Style**

**Current User Preference**: "Stop writing essays, just execute"

**Preferred Format**:
- Short summaries (not 900-line reports)
- Code + diffs + results
- Web searches when debugging (CUTLASS, bank conflicts)
- Update this file (AGENTS.md) when context changes

**Avoid**:
- Long markdown reports
- README/blog posts
- Multiple options (just execute best path)
- Repeating known information

---

## 🔄 **Session State**

**Current Phase**: Option 2 - Phase A (Debug CUTLASS)

**Active Tasks**:
1. Fix CUTLASS "Error Internal"
2. Root-cause M=64 hang
3. Document findings

**Next Tasks** (if Phase A succeeds):
1. Integrate TC Q@K^T into Phase 4 kernel
2. Validate correctness
3. Measure speedup with NCU

**Repository**: Clean, 16 hours invested, portfolio-ready baseline

**Grade So Far**: A- (excellent infra, hit expected TC wall)

---

**Last Action**: Rehydration complete, NCU working, Evo found 839 μs winner  
**Next Action**: Debug CUTLASS + investigate M=64 hang (executing now)

