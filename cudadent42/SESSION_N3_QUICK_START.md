# Session N+3 Quick Start (25-Minute Baseline)

**Date**: October 12, 2025, 3:58 AM PDT  
**GPU**: cudadent42-l4-dev (L4, us-central1-a)  
**Status**: ✅ ACTIVE (Pattern 7: Keep running 5+ hours)  
**Objective**: Apply WORKING_BUILD_RECIPE.md → 0.10× baseline in 25 minutes

---

## 🎯 Session Goals

1. **Fast Baseline** (25 min): Use WORKING_BUILD_RECIPE.md for reproducible 0.10× result
2. **Profile** (15 min): Nsight Compute analysis to identify bottleneck
3. **Fix ONE Thing** (20 min): Address highest-impact issue
4. **Validate** (10 min): Re-benchmark and measure improvement
5. **Document** (10 min): Update pattern library with new insights

**Total Time Budget**: 80 minutes (1.3 hours)

---

## ⏱️ Checkpoint Plan

### Checkpoint 1 (0-25 min): Baseline via WORKING_BUILD_RECIPE.md

**Decision Gate**: If baseline ≠ 0.10× ± 10%, STOP and debug build.

**Commands** (from WORKING_BUILD_RECIPE.md):
```bash
# SSH to GPU
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a --project=periodicdent42

# Verify git state
cd ~/periodicdent42/cudadent42
git status
git log -1 --oneline  # Should show working build commit

# Clean build
python setup.py clean
python setup.py build_ext --inplace 2>&1 | tee build.log

# Set environment
export PYTHONPATH=$PYTHONPATH:$(pwd)/flashmoe_science
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; print(torch.__path__[0])")/lib

# Quick smoke test
python -c "import flashmoe_science._C as fa; print('✅ Module loaded:', dir(fa))"

# Run benchmark
python benches/bench_correctness_and_speed.py 2>&1 | tee benchmark.log | grep -E "(Speedup|ERROR|seconds)"
```

**Expected Output**:
```
Speedup: 0.10× (FlashMoE-Science vs PyTorch)
Total time: ~5 seconds
```

**Success Criteria**: 0.09-0.11× speedup, no errors

---

### Checkpoint 2 (25-40 min): Nsight Compute Profile

**Decision Gate**: If profile reveals shared memory bank conflicts > 30%, fix those first.

**Commands**:
```bash
# Profile attention kernel (2048 sequence length)
ncu --set full --target-processes all \
    -o flash_attention_profile \
    python -c "
import torch
import flashmoe_science._C as fa
Q = torch.randn(1, 8, 2048, 64, dtype=torch.float16, device='cuda')
K, V = torch.randn_like(Q), torch.randn_like(Q)
_ = fa.flash_attention_forward(Q, K, V, False, 0.125)
"

# Export to text for quick analysis
ncu --import flash_attention_profile.ncu-rep --page details --csv > profile_details.csv

# Key metrics to check:
grep -E "(shared_load|shared_store|bank_conflict|occupancy)" profile_details.csv
```

**What to Look For**:
- **Shared memory bank conflicts**: Should be < 5%
- **Warp occupancy**: Should be > 50%
- **Memory throughput**: Should be > 50% of peak

---

### Checkpoint 3 (40-60 min): Fix ONE Bottleneck

**Decision Gate**: Choose highest-impact fix (expected improvement > 2×).

**Common Issues & Fixes**:

1. **Bank Conflicts** (if > 30%):
   - Add `__align__(16)` to shared memory arrays in `flash_attention_science.cu`
   - Use padding to avoid conflicts

2. **Low Occupancy** (if < 30%):
   - Reduce shared memory usage (already at 40KB for L4)
   - Reduce register usage via `-maxrregcount=64` in NVCC flags

3. **Poor Memory Throughput** (if < 30% of peak):
   - Check vectorized loads are working (should use `float4` or `half8`)
   - Verify memory coalescing

**Implementation Pattern**:
```bash
# Edit kernel
vim python/flashmoe_science/csrc/flash_attention_science.cu

# Rebuild (fast, single file)
python setup.py build_ext --inplace 2>&1 | tail -20

# Quick test
python benches/bench_correctness_and_speed.py 2>&1 | grep "Speedup"
```

---

### Checkpoint 4 (60-70 min): Validate Improvement

**Decision Gate**: If improvement < 1.5×, revert and try different fix.

**Commands**:
```bash
# Run full benchmark
python benches/bench_correctness_and_speed.py 2>&1 | tee benchmark_improved.log

# Compare results
echo "Before: 0.10× speedup"
grep "Speedup" benchmark_improved.log
```

**Success Criteria**: Speedup > 0.15× (1.5× improvement over baseline)

---

### Checkpoint 5 (70-80 min): Document Results

**Decision Gate**: None (always document, even failures)

**Actions**:
1. Create `SESSION_N3_COMPLETE_OCT12_2025.md` with:
   - Starting baseline (0.10×)
   - Bottleneck identified
   - Fix applied
   - Final speedup
   - Time breakdown vs. plan
   - New patterns discovered

2. Update `CUDA_KERNEL_LEARNING_LOOP.md` with:
   - Pattern 9 (if new insight discovered)
   - Retrospective: What worked? What didn't?
   - ROI: Time saved vs. Session N+2

3. Commit and push:
   ```bash
   git add cudadent42/SESSION_N3_COMPLETE_OCT12_2025.md
   git add cudadent42/CUDA_KERNEL_LEARNING_LOOP.md
   git commit -m "docs(cudadent42): Session N+3 complete - [X]× speedup achieved"
   git push origin opt/vectorized-loads
   ```

---

## 🚦 Decision Gates Summary

| Checkpoint | Time | Gate Condition | Action if Failed |
|------------|------|----------------|------------------|
| 1 | 25 min | Baseline = 0.10× ± 10% | Debug build, check WORKING_BUILD_RECIPE.md |
| 2 | 40 min | Profile completes, key metrics visible | Skip profiling, use prior knowledge |
| 3 | 60 min | Build succeeds, no new errors | Revert changes, document failure |
| 4 | 70 min | Speedup > 0.15× (1.5× improvement) | Document failure, analyze root cause |
| 5 | 80 min | None (always document) | — |

---

## 📊 Key Metrics to Track

| Metric | Baseline (N+2) | Target (N+3) | Actual |
|--------|----------------|--------------|--------|
| Speedup | 0.10× | 0.15-0.50× | _TBD_ |
| Build Time | 45 sec | < 60 sec | _TBD_ |
| Shared Mem Usage | 40 KB | 40 KB | _TBD_ |
| Occupancy | Unknown | > 50% | _TBD_ |
| Bank Conflicts | Unknown | < 5% | _TBD_ |

---

## 🔗 Key References

1. **WORKING_BUILD_RECIPE.md** - The exact build commands that work
2. **META_LEARNING_VALIDATION_COMPLETE.md** - Patterns 1-8
3. **CUDA_KERNEL_LEARNING_LOOP.md** - Session retrospectives
4. **cudadent42/python/flashmoe_science/csrc/build_config.h** - L4 configuration
5. **cudadent42/python/flashmoe_science/csrc/flash_attention_science.cu** - Kernel source

---

## ⚠️ Critical Reminders (Pattern Library)

- **Pattern 5**: Check GPU status before long operations (`gcloud compute instances describe`)
- **Pattern 6**: Git bisect > build archaeology (use known-working commits)
- **Pattern 7**: Keep GPU running during active sessions (5+ hour minimum)
- **Pattern 8**: Single compilation unit + native types = no linking errors

---

**Status**: ✅ READY TO EXECUTE  
**Expected Outcome**: 0.15-0.50× speedup in 80 minutes  
**Started**: 2025-10-12 03:58 AM PDT

---

## 🎯 Success Definition

**Minimum Viable Success**: 0.15× speedup (1.5× improvement) documented in 80 minutes

**Stretch Goal**: 0.50× speedup (5× improvement) with clear path to 1.0×+

**Pattern Library Update**: At least ONE new insight added to CUDA_KERNEL_LEARNING_LOOP.md

---

Let's execute! 🚀

