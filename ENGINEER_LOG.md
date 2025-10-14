# V3 CUDA Kernel Engineering Log

**Session Start**: 2025-10-14 (Continuation)  
**Objective**: Fix illegal memory access in V3, validate correctness, achieve ≤0.255 ms or fallback to V2  
**Engineer**: CUDA Principal (AI-assisted)

---

## Fix Plan

### Root Cause Hypothesis
V3 kernel has "CUDA illegal memory access" at runtime. Likely causes:
1. **cp.async address alignment**: Must be 16-byte aligned for cp.async.ca<16>
2. **Persistent block work distribution**: Bounds checking on (batch, head, m_block) iteration
3. **Register array indexing**: Out-of-bounds in Q_reg, O_acc, m_i, l_i
4. **SMEM indexing**: Stage buffer pointer arithmetic (stage 0/1)
5. **half2 alignment**: Assumes 2-element alignment, may not hold

### Fix Strategy
1. Add DEBUG guards with bounds checking
2. Run compute-sanitizer (memcheck, racecheck, initcheck)
3. Fix reported issues one at a time
4. Validate correctness (7 test cases)
5. Performance gate (≤0.255 ms with CI + effect size)
6. Nsight profiling if accepted
7. Decision: Promote V3 or fallback to V2

---

## Diffs Applied

### Timestamp: [PENDING]

**File**: TBD  
**Change**: TBD  
**Reason**: TBD

---

## Commands Run

### Timestamp: [PENDING]

```bash
# Commands will be logged here
```

**Output**: [PENDING]

---

## Results

### Pass/Fail Gates

- [ ] Compute-sanitizer memcheck: CLEAN
- [ ] Compute-sanitizer racecheck: CLEAN
- [ ] Compute-sanitizer initcheck: CLEAN
- [ ] Correctness: 7/7 tests pass (atol=1e-2, rtol=5e-2)
- [ ] Performance: ≤0.255 ms (mean)
- [ ] Statistics: 95% CIs non-overlapping vs V2
- [ ] Effect size: Hedges' g ≥ 0.8
- [ ] Nsight: ↑L2 hit-rate ≥+8pp, ↓DRAM% ≥−10pp

---

## Next Decision

**Status**: IN PROGRESS  
**Next Action**: Step 0 - Add debug utilities and guards

---

## Artifacts

- `artifacts/sanitizers/`: Compute-sanitizer logs
- `artifacts/correctness/`: Test results JSON/CSV
- `artifacts/bench/`: Performance benchmark CSV
- `artifacts/nsight/`: Nsight Compute profiles

