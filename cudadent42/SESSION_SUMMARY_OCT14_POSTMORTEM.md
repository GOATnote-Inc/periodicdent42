# Post-Mortem Infrastructure Complete: V3 Correctness Repair (Steps 0-1)

**Session Date:** October 14, 2025  
**Objective:** Implement post-mortem plan for V3 correctness repair with SDPA as production champion  
**Status:** Infrastructure complete (Steps 0-1). Ready for GPU execution (Steps 1b-6).

---

## ✅ What Was Accomplished (No GPU Required)

### Step 0: Guardrails (5 min) — COMPLETE ✅

**README Updated:** PyTorch SDPA declared as production champion
- **Performance:** 0.073 ms per call (B=2, H=8, S=512, D=64, FP16)
- **Correctness:** 100% (industry-standard reference)
- **Status:** Stable, validated, ready for production

**Artifacts Directory Structure Created:**
```
cudadent42/artifacts/
├── oracle/          # Tile oracle test results
├── sanitizers/      # Compute-sanitizer logs
├── correctness/     # Correctness gate JSON
├── bench/           # Performance benchmark CSVs
├── stats/           # Statistical analysis JSON
└── nsight/          # Nsight Compute profiles
```

**Commit:** a8376e2 - "docs(prod): SDPA set as production path; start v3 fix with post-mortem plan"

---

### Step 1: Tile Oracle Infrastructure (20 min) — COMPLETE ✅

**Kernel Modifications (fa_s512_v3.cu):**

Added DEBUG_DUMP conditional compilation guards:
```cuda
#if defined(DEBUG_DUMP)
__device__ float* g_S_dump = nullptr;  // Attention scores (QK)
__device__ float* g_P_dump = nullptr;  // Attention probs (softmax)
__device__ float* g_O_dump = nullptr;  // Output
#endif
```

Dump hooks inserted at critical stages:
1. **S dump** (line 294-302): After QK computation, before softmax
2. **P dump** (line 332-340): After online softmax, before SV multiplication
3. Only dumps from block(0,0) to minimize overhead

**Oracle Test Created (tests/oracles/tile_oracle_v3.py):**

Features:
- Tests V3 on S=512 (V3's specialized size) against SDPA oracle
- Detects NaN/Inf with precise location reporting (first occurrence, count)
- Analyzes error patterns:
  * Top 5 worst elements
  * Error distribution (NaN/Inf/zero counts)
  * Max/mean absolute and relative differences
- Supports all 3 V3 configs:
  * Config 0: `32_64_4` (BLOCK_M=32, BLOCK_N=64, NUM_WARPS=4)
  * Config 1: `32_32_4` (BLOCK_M=32, BLOCK_N=32, NUM_WARPS=4)
  * Config 2: `48_64_8` (BLOCK_M=48, BLOCK_N=64, NUM_WARPS=8)
- Saves numpy arrays to `artifacts/oracle/` for deep analysis
- Command-line interface:
  ```bash
  python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal
  python3 tests/oracles/tile_oracle_v3.py --config 1 --causal
  python3 tests/oracles/tile_oracle_v3.py --both  # Test causal + non-causal
  ```

**Commit:** 05609b7 - "feat(v3): Add DEBUG_DUMP hooks and tile oracle test for V3 correctness repair"

---

### Documentation (10 min) — COMPLETE ✅

**Created POSTMORTEM_READY.md:** Complete GPU execution guide
- Step-by-step commands for Steps 1b-6
- Copy-paste ready bash scripts
- Expected outputs for each step
- Decision matrix (V3 vs SDPA champion)
- Time/cost estimates (2.5h total, $1.70, $1.00 stop-loss)
- Success criteria and fallback plans

**Updated ENGINEER_LOG.md:** Session tracking
- Post-mortem plan documented
- Steps 0-1 marked complete with commit SHAs
- Steps 1b-6 outlined with time/cost estimates
- Next session quick-start command provided

**Commit:** b7113d1 - "docs: Post-mortem Steps 0-1 complete; Steps 1b-6 ready for GPU session"

---

## 🚀 Ready for Next Session: GPU Execution (Steps 1b-6)

### Step 1b: Run Tile Oracle (10 min, $0.11)

**Goal:** Identify which stage (S, P, or O) diverges first.

**Commands:**
```bash
# Start GPU
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# Navigate
cd /workspace/periodicdent42/cudadent42/bench

# Run oracle (all 3 configs)
python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal  # 32_64_4
python3 tests/oracles/tile_oracle_v3.py --config 1 --noncausal  # 32_32_4
python3 tests/oracles/tile_oracle_v3.py --config 2 --noncausal  # 48_64_8

# Check results
cat ../artifacts/oracle/noncausal/v3_oracle_config0_results.json
```

**Expected Outcomes:**
- ✅ **Pass** (max_abs_diff < 0.01): Skip to Step 4 (correctness gate)
- ❌ **NaN/Inf detected**: Proceed to Step 2 (compute-sanitizer)
- ❌ **Divergence > 0.01**: Analyze pattern, proceed to Step 2

**Diagnostic Info Provided:**
- NaN/Inf count and first occurrence location
- Top 5 worst elements with indices and values
- Max/mean absolute and relative differences
- Saved numpy arrays: `v3_config{0,1,2}_O_{ref,test}.npy`

---

### Step 2: Compute-Sanitizer (15 min, $0.17)

**Goal:** Find race conditions, uninitialized memory, alignment issues.

**Commands:**
```bash
cd /workspace/periodicdent42/cudadent42/bench

# Memcheck (illegal memory access)
compute-sanitizer --tool memcheck \
  python3 tests/oracles/tile_oracle_v3.py --config 0 \
  2>&1 | tee ../artifacts/sanitizers/v3_memcheck.log

# Racecheck (shared memory races)
compute-sanitizer --tool racecheck \
  python3 tests/oracles/tile_oracle_v3.py --config 0 \
  2>&1 | tee ../artifacts/sanitizers/v3_racecheck.log

# Initcheck (uninitialized variables)
compute-sanitizer --tool initcheck \
  python3 tests/oracles/tile_oracle_v3.py --config 0 \
  2>&1 | tee ../artifacts/sanitizers/v3_initcheck.log
```

**Look For:**
- "Invalid __shared__ write"
- "Out of bounds"
- "Uninitialized variable"
- Exact line numbers and addresses

**Commit After Analysis:**
```bash
git add cudadent42/artifacts/sanitizers/ ENGINEER_LOG.md
git commit -m "test(v3): compute-sanitizer logs for V3 correctness debugging"
```

---

### Step 3: Fix Loop (60 min, $0.68, max 2 iterations)

**Based on oracle + sanitizer findings, fix bugs one at a time.**

#### If S diverges (QK path):
- Check WMMA layouts (lines 276-290 in `fa_s512_v3.cu`)
- Verify `softmax_scale` is applied correctly
- Check 16-byte alignment for `Q_reg` and `smem->K`

#### If P diverges (online softmax):
- Check stable softmax update (lines 304-330)
- Verify correction factor: `(m_old == -inf) ? 1.0f : expf(m_old - m_new)`
- Check causal mask uses global indices: `n = n_block * BLOCK_N + n_idx`

#### If O diverges (SV path):
- Check accumulation (lines 342-349)
- Verify final normalization: `O_acc[local_row][d] * norm` where `norm = 1.0f / l_i[local_row]`

**After Each Fix:**
```bash
# Edit fa_s512_v3.cu
python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal
git add cudadent42/bench/kernels/fa_s512_v3.cu ENGINEER_LOG.md
git commit -m "fix(v3): correct <QK|softmax|SV> (Iteration N)"
```

**Stop Condition:**
- If 2 iterations don't produce green oracle → STOP V3
- Document failure → SDPA remains champion

---

### Step 4: S=512 Correctness Gate (20 min, $0.23)

**Goal:** Validate V3 correctness on 7 test cases.

**Test Cases:**
- Shape: (B=2, H=8, S=512, D=64)
- Causal: [False, True]
- Tolerances: atol=1e-2, rtol=5e-2

**Pass Criteria:**
- ✅ All 2 tests (causal + non-causal) pass tolerances
- ❌ Any test fails → Return to Step 3 (if iterations remaining) or STOP

**Commit:**
```bash
git add cudadent42/artifacts/correctness/v3_s512.json ENGINEER_LOG.md
git commit -m "test(v3): S512 correctness gate PASSED"
```

---

### Step 5: Performance Gate (30 min, $0.34)

**Goal:** Measure V3 performance and compare to V2.

**Benchmark Configuration:**
- Shape: (B=2, H=8, S=512, D=64)
- Warmup: 20 iterations
- Measurement: 100 iterations
- Metric: Latency (ms) with bootstrap 95% CI

**Pass Criteria:**
1. Mean ≤ 0.255 ms (target: 20% faster than V2's 0.318 ms)
2. CI non-overlap vs V2 (no statistical ambiguity)
3. Hedges' g ≥ 0.8 (large effect size)

**Decision:**
- ✅ Pass all 3: Promote V3 to champion (update README)
- ❌ Fail any: Keep SDPA as champion, document V3 as "correct but slow"

**Commit:**
```bash
git add cudadent42/artifacts/bench/ cudadent42/artifacts/stats/ ENGINEER_LOG.md
git commit -m "perf(v3): performance gate results with CI/effect size"
```

---

### Step 6: Evidence & README (10 min, $0.11)

**Update README:**
- If V3 promoted: Update champion section, add V3 perf metrics
- If V3 blocked: Add "V3 Development Status" section, document blocker

**Update ENGINEER_LOG:**
- Link all artifacts in `artifacts/` directory
- Summarize findings (divergence stage, root cause, fix applied)
- Document decision rationale (champion choice)

**Create Post-Mortem (if V3 blocked):**
- Root cause explanation
- Fix attempts (2 iterations max)
- Why L4 is challenging (SMEM limits, SM count, bandwidth)
- Path forward (specific changes needed, or pivot to decode kernel)

**Commit:**
```bash
git add cudadent42/README.md ENGINEER_LOG.md cudadent42/V3_POSTMORTEM.md
git commit -m "docs: finalize post-mortem; champion decision documented"
```

---

## 📊 Decision Matrix

| Outcome | Champion | README Update | Next Steps |
|---------|----------|---------------|------------|
| V3 passes correctness + perf | V3 | Add V3 metrics | Deploy to prod |
| V3 passes correctness, fails perf | SDPA | Note "V3 correct, 6.5× slower" | Defer optimization |
| V3 fails correctness (2 iter) | SDPA | Add post-mortem | Document blocker |

---

## 🎯 Success Criteria

**Minimum (Must Have):**
- ✅ SDPA declared as champion (DONE in Step 0)
- ✅ V3 bugs diagnosed and documented
- ✅ All artifacts saved to `artifacts/`
- ✅ Honest findings in ENGINEER_LOG

**Ideal (If V3 Repairs Work):**
- V3 passes correctness gate (atol=1e-2)
- V3 ≤ 0.255 ms (20% faster than V2)
- V3 promoted to champion

**Fallback (If V3 Blocked):**
- Root cause identified and documented
- Post-mortem explains L4 challenges
- Path forward documented
- SDPA remains champion with confidence

---

## 📈 Estimated Time/Cost

| Step | Duration | GPU Cost ($0.68/hr) | Cumulative |
|------|----------|---------------------|------------|
| 1b. Run oracle | 10 min | $0.11 | $0.11 |
| 2. Sanitizers | 15 min | $0.17 | $0.28 |
| 3. Fix loop (2 iter) | 60 min | $0.68 | $0.96 |
| 4. Correctness gate | 20 min | $0.23 | $1.19 |
| 5. Perf gate | 30 min | $0.34 | $1.53 |
| 6. Evidence | 10 min | $0.11 | $1.64 |
| **Total** | **~2.5 hours** | **~$1.70** | |

**Stop Loss:** If no progress after Step 3 Iteration 2 → stop and document. Max spend: $1.00.

---

## 📝 Files Modified

**Production Code:**
- `cudadent42/README.md` (SDPA as champion)
- `cudadent42/bench/kernels/fa_s512_v3.cu` (DEBUG_DUMP hooks)

**Test Infrastructure:**
- `cudadent42/bench/tests/oracles/tile_oracle_v3.py` (new oracle test)

**Documentation:**
- `ENGINEER_LOG.md` (session tracking)
- `cudadent42/POSTMORTEM_READY.md` (GPU execution guide)
- `cudadent42/SESSION_SUMMARY_OCT14_POSTMORTEM.md` (this file)

**Artifacts:**
- `cudadent42/artifacts/` (directory structure created)

---

## 🔧 Commits

1. **a8376e2** - "docs(prod): SDPA set as production path; start v3 fix with post-mortem plan"
   - README: SDPA as champion
   - artifacts/: Directory structure
   - ENGINEER_LOG: Session initialized

2. **05609b7** - "feat(v3): Add DEBUG_DUMP hooks and tile oracle test for V3 correctness repair"
   - fa_s512_v3.cu: S and P dump hooks
   - tile_oracle_v3.py: Oracle test for all 3 configs
   - ENGINEER_LOG: Step 1 complete

3. **b7113d1** - "docs: Post-mortem Steps 0-1 complete; Steps 1b-6 ready for GPU session"
   - POSTMORTEM_READY.md: Complete GPU execution guide
   - ENGINEER_LOG: Steps 1b-6 outlined
   - SESSION_SUMMARY_OCT14_POSTMORTEM.md: This summary

---

## 🚀 Next Session Quick Start

```bash
# Start GPU
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# Navigate
cd /workspace/periodicdent42/cudadent42/bench

# Run oracle (Step 1b)
python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal

# Check results
cat ../artifacts/oracle/noncausal/v3_oracle_config0_results.json

# Decision tree:
# - NaN/Inf detected → Step 2 (compute-sanitizer)
# - Divergence > 0.01 → Step 3 (fix loop)
# - Passed → Step 4 (correctness gate)
```

**Reference:** See `POSTMORTEM_READY.md` for complete step-by-step guide.

---

## 📊 Current Status

**Infrastructure:** ✅ Complete (Steps 0-1)  
**GPU Execution:** ⏳ Ready (Steps 1b-6 require GPU)  
**Production Champion:** ✅ SDPA (0.073 ms, 100% correct)  
**V3 Status:** 🔧 Under repair (correctness bugs, max 2 iterations)  
**Documentation:** ✅ Complete (POSTMORTEM_READY.md, ENGINEER_LOG.md)  
**Commits:** ✅ 3 commits pushed (a8376e2, 05609b7, b7113d1)  
**Artifacts:** ✅ Directory structure created  
**Next:** Start GPU → Run tile oracle test

---

**Session Complete:** Infrastructure ready. Awaiting GPU session to execute diagnostic and repair Steps 1b-6.

**Estimated Next Session:** 2.5 hours, $1.70 (with $1.00 stop-loss if no progress).

**Champion Decision:** Will be made at Step 5 performance gate (or earlier if V3 blocked at correctness).

---

*End of Session Summary - October 14, 2025*

