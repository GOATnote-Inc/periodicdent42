# V3 Post-Mortem Repair: Ready for GPU Session

**Status**: Steps 0-1 complete (infrastructure). Ready for GPU execution.

**Objective**: Repair V3 correctness with SDPA as production champion. Max 2 bug-fix iterations.

---

## ✅ Completed (Local, No GPU)

### Step 0: Guardrails (5 min) — DONE
- ✅ README: SDPA declared as production champion (0.073 ms, 100% correct)
- ✅ `artifacts/` structure: `oracle/`, `sanitizers/`, `correctness/`, `bench/`, `stats/`, `nsight/`
- ✅ ENGINEER_LOG.md: Post-mortem session initialized
- ✅ Commit: a8376e2

### Step 1: Tile Oracle Infrastructure (20 min) — DONE
- ✅ Added DEBUG_DUMP hooks to `fa_s512_v3.cu`:
  * S dump after QK computation (line 294-302)
  * P dump after online softmax (line 332-340)
  * Conditional compilation (`#if defined(DEBUG_DUMP)`)
  * Only dumps from block(0,0) to minimize overhead
- ✅ Created `bench/tests/oracles/tile_oracle_v3.py`:
  * Tests V3 on S=512 vs SDPA oracle
  * Detects NaN/Inf with precise location reporting
  * Analyzes error patterns (top 5 worst elements)
  * Saves numpy arrays to `artifacts/oracle/` for analysis
  * Supports all 3 configs: 32_64_4, 32_32_4, 48_64_8
- ✅ Commit: 05609b7

---

## 🚀 Next: GPU Execution (Steps 1b-6)

### Step 1b: Run Tile Oracle (10 min)

**Start GPU:**
```bash
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a
```

**Run Oracle Test:**
```bash
cd /workspace/periodicdent42/cudadent42/bench

# Test all 3 configs
python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal  # 32_64_4
python3 tests/oracles/tile_oracle_v3.py --config 1 --noncausal  # 32_32_4
python3 tests/oracles/tile_oracle_v3.py --config 2 --noncausal  # 48_64_8

# Check results
ls -lh ../artifacts/oracle/noncausal/
cat ../artifacts/oracle/noncausal/v3_oracle_config0_results.json
```

**Expected Output:**
- If NaN/Inf detected: Location, count, first occurrence
- If divergence: Max abs diff, worst 5 elements, error pattern
- Saved files: `v3_config{0,1,2}_O_ref.npy`, `v3_config{0,1,2}_O_test.npy`

**Decision Point:**
- ✅ If passed (max_abs < 0.01): Skip to Step 4 (S=512 correctness gate)
- ❌ If NaN/Inf: Proceed to Step 2 (compute-sanitizer)
- ❌ If divergence > 0.01: Analyze pattern, proceed to Step 2

---

### Step 2: Compute-Sanitizer + Determinism (10-15 min)

**Goal**: Find race conditions, uninitialized memory, alignment issues.

**Run Sanitizers (Debug Build):**
```bash
cd /workspace/periodicdent42/cudadent42/bench

# Build debug version (no -use_fast_math, add -G -lineinfo)
# Modify build_v3_release.py to add debug flags for sanitizer run

# Memcheck
compute-sanitizer --tool memcheck python3 tests/oracles/tile_oracle_v3.py --config 0 2>&1 | tee ../artifacts/sanitizers/v3_memcheck.log

# Racecheck
compute-sanitizer --tool racecheck python3 tests/oracles/tile_oracle_v3.py --config 0 2>&1 | tee ../artifacts/sanitizers/v3_racecheck.log

# Initcheck
compute-sanitizer --tool initcheck python3 tests/oracles/tile_oracle_v3.py --config 0 2>&1 | tee ../artifacts/sanitizers/v3_initcheck.log
```

**Analysis:**
- Look for: "Invalid __shared__ write", "Out of bounds", "Uninitialized variable"
- Note exact line numbers and addresses
- Document in ENGINEER_LOG.md

**Commit:**
```bash
git add cudadent42/artifacts/sanitizers/
git commit -m "test(v3): compute-sanitizer logs for V3 correctness debugging"
```

---

### Step 3: Fix Loop (Max 2 Iterations)

**Based on Oracle + Sanitizer Results:**

#### If S diverges (QK path):
1. **Check WMMA layouts**:
   - Line 276-290 in `fa_s512_v3.cu`
   - Verify `Q_reg` and `smem->K` are accessed correctly
   - Confirm `softmax_scale` is applied

2. **Check alignment**:
   - Verify 16-byte alignment for `Q_reg[local_row][d]`
   - Verify `smem->K[stage][n_idx][d]` with stride

**Fix, test, commit:**
```bash
# Edit fa_s512_v3.cu
python3 tests/oracles/tile_oracle_v3.py --config 0 --noncausal
git add cudadent42/bench/kernels/fa_s512_v3.cu ENGINEER_LOG.md
git commit -m "fix(v3): correct QK computation (Iteration 1)"
```

#### If P diverges (online softmax):
1. **Check stable softmax update** (lines 304-330):
   ```cuda
   m_new = max(m_old, max(S_row))
   corr  = (m_old == -inf) ? 1.0f : expf(m_old - m_new)
   O_acc *= corr
   l = l * corr + sum(exp(S_row - m_new))
   ```

2. **Check causal mask**:
   - Line 268: `if (is_causal && n > m)` uses **global** indices
   - Confirm `n = n_block * BLOCK_N + n_idx` is correct

**Fix, test, commit.**

#### If O diverges (SV path):
1. **Check accumulation** (lines 342-349):
   - Verify `S_row[n_idx] * smem->V[stage][n_idx][d]`
   - Check `O_acc[local_row][d] += acc`

2. **Check final normalization** (line 385):
   - Verify `O_acc[local_row][d] * norm` where `norm = 1.0f / l_i[local_row]`

**Fix, test, commit.**

**Stop Condition:**
- If 2 iterations don't produce green oracle: STOP V3, document failure, SDPA remains champion.

---

### Step 4: S=512 Correctness Gate (15-20 min)

**Run Full Correctness Suite:**
```bash
cd /workspace/periodicdent42/cudadent42/bench

python3 -c "
from build_v3_release import build_v3_release
from bench.tests.correctness_suite import test_v3_correctness
import json

results = test_v3_correctness(
    shapes=[(2, 8, 512, 64)],
    is_causal_list=[False, True],
    atol=1e-2,
    rtol=5e-2
)

with open('../artifacts/correctness/v3_s512.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Correctness Results:', results)
"
```

**Gate:**
- ✅ Pass: All 2 tests (causal + non-causal) with atol=1e-2, rtol=5e-2
- ❌ Fail: Return to Step 3 (if iterations remaining) or STOP

**Commit:**
```bash
git add cudadent42/artifacts/correctness/v3_s512.json ENGINEER_LOG.md
git commit -m "test(v3): S512 correctness gate PASSED"
```

---

### Step 5: Performance Gate (20-30 min)

**Run Benchmark (Release Build):**
```bash
cd /workspace/periodicdent42/cudadent42/bench

python3 -c "
import torch
from build_v3_release import build_v3_release
from bench.common.stats import bootstrap_ci, hedges_g
import csv
import json

# Load V3
v3_module = build_v3_release()
v3_forward = v3_module.forward_32_64_4_2_1_1  # Or best config from oracle

# Inputs
B, H, S, D = 2, 8, 512, 64
Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
scale = 1.0 / (D ** 0.5)

# Warmup
for _ in range(20):
    _ = v3_forward(Q, K, V, scale, False)
torch.cuda.synchronize()

# Benchmark
import time
latencies = []
for _ in range(100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = v3_forward(Q, K, V, scale, False)
    torch.cuda.synchronize()
    latencies.append((time.perf_counter() - start) * 1000)

# Save
with open('../artifacts/bench/v3_latencies.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['iteration', 'latency_ms'])
    for i, lat in enumerate(latencies):
        writer.writerow([i, lat])

# Stats
mean = sum(latencies) / len(latencies)
ci_low, ci_high = bootstrap_ci(latencies)
print(f'V3: {mean:.3f} ms [{ci_low:.3f}, {ci_high:.3f}]')
"
```

**Compare to V2:**
```bash
python3 cudadent42/bench/stats/make_decision_v2.py
```

**Gate Criteria:**
- Mean ≤ 0.255 ms (target)
- CI non-overlap vs V2 (0.318 ms)
- Hedges' g ≥ 0.8 (large effect)

**Decision:**
- ✅ Pass all 3: Promote V3 to champion (update README)
- ❌ Fail any: Keep SDPA as champion, document V3 as "correct but slow"

**Commit:**
```bash
git add cudadent42/artifacts/bench/ cudadent42/artifacts/stats/ ENGINEER_LOG.md
git commit -m "perf(v3): performance gate results with CI/effect size"
```

---

### Step 6: Evidence & README (10 min)

**Update README:**
```bash
# If V3 promoted:
sed -i 's/PyTorch SDPA/V3 Custom Kernel/' cudadent42/README.md
# Add V3 perf metrics

# If V3 blocked:
# Add section: "V3 Development Status"
# Document blocker and SDPA as champion
```

**Update ENGINEER_LOG:**
- Link all artifacts
- Summarize findings
- Document decision rationale

**Create Post-Mortem:**
```bash
# If V3 failed to meet gates:
cat > cudadent42/V3_POSTMORTEM.md << 'EOF'
# V3 Post-Mortem: Why SDPA Remains Champion

## Root Cause
[Describe bug found in Step 1-3]

## Fix Attempts
1. Iteration 1: [What was tried, result]
2. Iteration 2: [What was tried, result]

## Why L4 is Challenging
[Explain SMEM limits, SM count, memory bandwidth]

## Path Forward
- Keep SDPA as production kernel (0.073 ms, 100% correct)
- Revisit V3 with [specific changes needed]
- Consider decode-optimized kernel instead
EOF
```

**Commit:**
```bash
git add cudadent42/README.md ENGINEER_LOG.md cudadent42/V3_POSTMORTEM.md
git commit -m "docs: finalize post-mortem; champion decision documented"
```

---

## 📊 Decision Matrix

| Outcome | Champion | README | Next Steps |
|---------|----------|--------|------------|
| V3 passes correctness + perf gates | V3 | Update metrics | Deploy to prod |
| V3 passes correctness, fails perf | SDPA | Note V3 as "correct, 6.5× slower" | Defer optimization |
| V3 fails correctness after 2 iterations | SDPA | Add post-mortem | Document blocker |

---

## 🎯 Success Criteria

**Minimum (Must Have):**
- ✅ SDPA declared as champion (already done)
- ✅ V3 bugs diagnosed and documented
- ✅ All artifacts saved to `artifacts/`
- ✅ Honest findings in ENGINEER_LOG

**Ideal (If V3 Repairs Work):**
- ✅ V3 passes correctness gate (atol=1e-2)
- ✅ V3 ≤ 0.255 ms (20% faster than V2)
- ✅ V3 promoted to champion

**Fallback (If V3 Blocked):**
- ✅ Root cause identified and documented
- ✅ Post-mortem explains L4 challenges
- ✅ Path forward documented
- ✅ SDPA remains champion with confidence

---

## 📝 Time/Cost Estimates

| Step | Duration | GPU Cost ($0.68/hr) | Cumulative |
|------|----------|---------------------|------------|
| 1b. Run oracle | 10 min | $0.11 | $0.11 |
| 2. Sanitizers | 15 min | $0.17 | $0.28 |
| 3. Fix loop (2 iter) | 60 min | $0.68 | $0.96 |
| 4. Correctness gate | 20 min | $0.23 | $1.19 |
| 5. Perf gate | 30 min | $0.34 | $1.53 |
| 6. Evidence | 10 min | $0.11 | $1.64 |
| **Total** | **~2.5 hours** | **~$1.70** | |

**Stop Loss:** If no progress after Step 3 Iteration 2, stop and document. Max spend: $1.00.

---

## 🚀 Quick Start (Copy-Paste)

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

# If NaN/Inf detected → Step 2 (compute-sanitizer)
# If divergence > 0.01 → Step 3 (fix loop)
# If passed → Step 4 (correctness gate)
```

---

**Status**: Infrastructure ready. Awaiting GPU session to execute Steps 1b-6.

**Files Modified:**
- `cudadent42/README.md` (SDPA as champion)
- `cudadent42/bench/kernels/fa_s512_v3.cu` (DEBUG_DUMP hooks)
- `cudadent42/bench/tests/oracles/tile_oracle_v3.py` (new test)
- `ENGINEER_LOG.md` (session tracking)

**Commits:**
- a8376e2: Step 0 guardrails
- 05609b7: Step 1 tile oracle infrastructure

**Ready for GPU**: Yes ✅

