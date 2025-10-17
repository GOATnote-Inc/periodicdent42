# Phase D.2: TDD Plan - Register Pressure Attack

**Date**: Oct 17, 2025  
**Mission**: Fix low occupancy (9.28% → 33%+) via register pressure reduction  
**Approach**: Test-Driven Development with NCU validation gates

---

## **🎯 Target Metrics (from NCU acceptance gates)**

```
Current (xFormers):        Target (Phase D.2):
─────────────────────────  ─────────────────────────
Achieved Occupancy: 9.28%  → 20%+  ✅
Eligible Warps:     0.27   → 2+    ✅
Issue Slot Util:    25.74% → 60%+  ✅
Latency:            24.22μs → <24μs ✅
```

---

## **📋 TDD Cycle 1: Baseline Custom Kernel**

### **Test 1.1: Build Phase 4 kernel as-is**
```bash
REGCAP="" python bench/build_custom_tuned.py
python bench/run_custom_tuned.py --shape S=512,D=64
```

**Expected**: Build succeeds, establishes baseline latency  
**NCU Check**: Profile to confirm register usage

---

### **Test 1.2: Add launch bounds**
```cuda
// fa_phase4_tuned.cu
#ifdef LAUNCH_BOUNDS_THREADS
__launch_bounds__(LAUNCH_BOUNDS_THREADS, LAUNCH_BOUNDS_MIN)
#endif
__global__ void attention_kernel(...) {
```

**Expected**: Compilation succeeds with launch bounds  
**NCU Check**: Theoretical occupancy should increase

---

## **📋 TDD Cycle 2: Register Cap Exploration**

### **Test 2.1: REGCAP=96 (gentle cap)**
```bash
REGCAP=96 LB_THREADS=256 LB_MIN=2 python bench/build_custom_tuned.py
```

**Expected**: 
- ✅ Build succeeds
- ✅ Occupancy ≥ 15%
- ✅ Latency within 1.5× of baseline

**NCU Gates**:
- `smsp__warps_active.avg.pct_of_peak_sustained_active ≥ 15%`
- `smsp__warps_eligible.avg.per_cycle_active ≥ 0.5`

---

### **Test 2.2: REGCAP=88**
```bash
REGCAP=88 LB_THREADS=256 LB_MIN=2 python bench/build_custom_tuned.py
```

**Expected**:
- ✅ Build succeeds
- ✅ Occupancy ≥ 18%
- ✅ Some spills acceptable if latency improves

---

### **Test 2.3: REGCAP=80**
```bash
REGCAP=80 LB_THREADS=192 LB_MIN=2 python bench/build_custom_tuned.py
```

**Expected**:
- ✅ Build succeeds
- ✅ Occupancy ≥ 20%
- ✅ Latency < 30μs

**Critical**: This is the target config from user's directive

---

## **📋 TDD Cycle 3: SMEM Migration**

### **Test 3.1: Identify large per-thread arrays**
```bash
# Check ptxas output for register usage
nvcc -Xptxas -v ... | grep "registers"
```

**Expected**: Identify arrays/vars using most registers

---

### **Test 3.2: Move to shared memory**
```cuda
__global__ void attention_kernel(...) {
    extern __shared__ float smem[];
    
    // OLD: float per_row_max[SEQ_LEN];  // High register pressure!
    // NEW: float* per_row_max = smem + threadIdx.x * SEQ_LEN;
```

**Expected**:
- ✅ Register usage drops by 10-20%
- ✅ Occupancy increases
- ✅ Correctness maintained

---

## **📋 TDD Cycle 4: Block Size Tuning**

### **Test 4.1: THREADS=192 (vs 256)**
```bash
REGCAP=80 LB_THREADS=192 LB_MIN=2 python bench/build_custom_tuned.py
```

**Expected**: More blocks per SM → higher occupancy

---

### **Test 4.2: THREADS=128**
```bash
REGCAP=80 LB_THREADS=128 LB_MIN=3 python bench/build_custom_tuned.py
```

**Expected**: Maximum occupancy, but may have less work per block

---

## **📋 TDD Cycle 5: Unroll Reduction**

### **Test 5.1: Bounded unrolls**
```cuda
// OLD: #pragma unroll
// NEW: #pragma unroll 4
for (int i = 0; i < TILE_K; i++) {
```

**Expected**: Reduce instruction cache pressure, lower regs

---

## **📋 TDD Cycle 6: Pointer Annotations**

### **Test 6.1: Add __restrict__**
```cuda
__global__ void attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
```

**Expected**: Compiler can optimize better, fewer anti-aliasing regs

---

## **🔬 NCU Validation Script**

```bash
#!/bin/bash
# tools/ncu_validate_occupancy.sh

CONFIG="$1"  # e.g., "REGCAP=80_THREADS=192"

echo "NCU Validation: $CONFIG"

# Profile
ncu \
  --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
           smsp__warps_eligible.avg.per_cycle_active,\
           sm__issue_active.avg.pct_of_peak_sustained_active,\
           launch__occupancy_limit_blocks,\
           launch__occupancy_limit_registers,\
           launch__occupancy_limit_shared_mem,\
           launch__occupancy_per_block_size \
  --target-processes all \
  --kernel-name regex:"attention" \
  --launch-count 1 \
  --csv \
  python bench/run_custom_tuned.py \
  2>&1 | tee "evidence/ncu_${CONFIG}.log"

# Parse gates
OCC=$(grep "warps_active" "evidence/ncu_${CONFIG}.log" | awk '{print $NF}')
ELIG=$(grep "warps_eligible" "evidence/ncu_${CONFIG}.log" | awk '{print $NF}')
ISSUE=$(grep "issue_active" "evidence/ncu_${CONFIG}.log" | awk '{print $NF}')

echo ""
echo "Results:"
echo "  Occupancy: $OCC% (gate: ≥20%)"
echo "  Eligible:  $ELIG (gate: ≥2)"
echo "  Issue:     $ISSUE% (gate: ≥60%)"

# Check gates
if (( $(echo "$OCC >= 20" | bc -l) )); then
    echo "✅ Occupancy gate PASSED"
else
    echo "❌ Occupancy gate FAILED"
fi
```

---

## **📊 Progress Tracking**

### **Cycle Checklist**

- [ ] Cycle 1: Baseline (Phase 4 as-is)
- [ ] Cycle 2: REGCAP exploration (96, 88, 80)
- [ ] Cycle 3: SMEM migration (large arrays)
- [ ] Cycle 4: Block size tuning (192, 128)
- [ ] Cycle 5: Unroll reduction
- [ ] Cycle 6: Pointer annotations

### **Success Criteria**

**Minimum (Pass)**:
- ✅ Occupancy ≥ 20%
- ✅ Eligible warps ≥ 2
- ✅ Latency ≤ 30μs
- ✅ Correctness: max_diff ≤ 2e-3

**Target (Good)**:
- ✅ Occupancy ≥ 25%
- ✅ Eligible warps ≥ 3
- ✅ Latency ≤ 24μs (beat xFormers!)
- ✅ Issue slots ≥ 60%

**Stretch (Excellent)**:
- ✅ Occupancy ≥ 30%
- ✅ Eligible warps ≥ 4
- ✅ Latency ≤ 20μs
- ✅ Issue slots ≥ 70%

---

## **🚨 Red Flags (When to Stop/Pivot)**

1. **Correctness failure**: max_diff > 2e-3 → rollback immediately
2. **Severe regression**: latency > 50μs → investigate, may need different approach
3. **Build failures**: 3+ consecutive → check CUDA version, nvcc flags
4. **NCU shows worse**: all metrics degrade → config is wrong

**But**: NO QUITTING on first failure! Try 3 configs before concluding approach is wrong.

---

## **📝 Documentation for Next Session**

### **Files to Update**

1. **`PHASE_D2_RESULTS.md`**: Log all test results
2. **`PHASE_D2_BEST_CONFIG.md`**: Document winning configuration
3. **`cudadent42/bench/kernels/fa_phase4_tuned.cu`**: Optimized kernel
4. **`evidence/ncu_*.log`**: All NCU profiling results

### **Key Metrics to Record**

```markdown
| Config | Regs/Thread | Occupancy | Eligible Warps | Latency | Correct |
|--------|-------------|-----------|----------------|---------|---------|
| Baseline | ?       | 9.28%     | 0.27           | ?       | ✅      |
| REGCAP=96| ?       | ?         | ?              | ?       | ?       |
| REGCAP=88| ?       | ?         | ?              | ?       | ?       |
| REGCAP=80| ?       | ?         | ?              | ?       | ?       |
```

---

## **⏱️ Time Budget**

```
Cycle 1 (Baseline):       30 min
Cycle 2 (REGCAP):         60 min
Cycle 3 (SMEM):           60 min
Cycle 4 (Block size):     30 min
Cycle 5 (Unroll):         15 min
Cycle 6 (Restrict):       15 min
Documentation:            30 min
──────────────────────────────────
Total:                    4 hours
```

---

## **🔧 Quick Reference Commands**

```bash
# Build with config
REGCAP=80 LB_THREADS=192 LB_MIN=2 python bench/build_custom_tuned.py

# Run benchmark
python bench/run_custom_tuned.py --shape S=512,D=64

# NCU profile
ncu -o evidence/ncu_test python bench/run_custom_tuned.py

# Check register usage
grep "registers" build.log

# Compare vs xFormers
python scripts/bench_fa2_vs_xformers.py
```

---

**Status**: READY TO EXECUTE  
**Next**: Start TDD Cycle 1 (Baseline)  
**Philosophy**: NO QUITTING - Systematic iteration until gates pass

