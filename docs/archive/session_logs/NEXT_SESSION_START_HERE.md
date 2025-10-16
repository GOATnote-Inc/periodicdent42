# Next Session: Quick Start Guide

**Status**: ✅ Infrastructure 100% Complete (8/14 phases)  
**Branch**: `feature/evoengineer-rbk-l4-optim` (9 commits)  
**GPU**: `cudadent42-l4-dev` (STOPPED - ready to start)

---

## 🚀 Quick Start (Copy-Paste Ready)

### Step 1: Start GPU (1 min)
```bash
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
# Wait ~30 seconds for boot
```

### Step 2: Copy Latest Code (5 min)
```bash
cd /Users/kiteboard/periodicdent42
gcloud compute scp --recurse \
  cudadent42/ \
  rbk_config.yaml \
  scripts/ \
  tests/ \
  third_party/ \
  cudadent42-l4-dev:~/periodicdent42/ \
  --zone=us-central1-a
```

### Step 3: SSH to GPU
```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a
```

### Step 4: Run Validation Workflow (On GPU)

```bash
cd ~/periodicdent42

# 4a. Correctness Tests (15 min)
python3 tests/test_sdpa_parity.py

# 4b. Sanitizer Suite (30 min)
./scripts/run_sanitizers.sh

# 4c. CI Gate (45 min)
./scripts/ci_local_gpu_gate.sh

# 4d. robust-kbench (30 min)
python3 scripts/run_rbk_benchmark.py --config rbk_config.yaml

# 4e. Review Results (5 min)
cat benchmarks/l4/ci_gate_*/gate_report.md
cat artifacts/sanitizers/sanitizer_summary.md
```

---

## 📊 What's Already Done

### Infrastructure (100%)
- ✅ EvoEngineer: Parameter optimizer (500+ lines)
- ✅ robust-kbench: Statistical benchmarking (550+ lines)
- ✅ Correctness gates: 72 SDPA parity test configs
- ✅ Sanitizer suite: 4 tools (memcheck/race/init/sync)
- ✅ CI regression gate: 4-stage validation
- ✅ Build flags: Debug + Release for sm_89

### Critical Fixes (Done)
- ✅ cp.async Fix A: Lines 474, 487
- ✅ DEBUG invariants: Softmax guards

---

## 🎯 Expected Results

### Correctness Tests
**Expected**: ✅ 100% pass rate  
**If fails**: Check kernel compilation, review Fix A impact

### Sanitizer Suite
**Expected**: ✅ 0 errors in all 4 tools  
**If fails**: Fix A validation critical - check synccheck tool

### CI Gate
**Expected**: ✅ Passes, baseline saved  
**Output**: `benchmarks/l4/ci_gate_YYYYMMDD_HHMMSS/gate_report.md`

### robust-kbench
**Expected**: Performance analysis vs SDPA  
**Output**: `benchmarks/l4/rbk_results/rbk_*.{json,csv,md}`

---

## 📁 Key Files to Review

After validation, check these files:

```bash
# Gate report
cat benchmarks/l4/ci_gate_*/gate_report.md

# Sanitizer summary
cat artifacts/sanitizers/sanitizer_summary.md

# Benchmark results
cat benchmarks/l4/rbk_results/rbk_sdpa.md
cat benchmarks/l4/rbk_results/rbk_v3.md
cat benchmarks/l4/rbk_results/comparison.json

# Logs (if issues)
tail -100 ci_gate_correctness.log
tail -100 ci_gate_benchmark.log
```

---

## 🔧 Troubleshooting

### If Correctness Tests Fail
```bash
# Check Fix A was applied
cd ~/periodicdent42/cudadent42
grep -n "cp_async_wait_group<0>()" bench/kernels/fa_s512_v3.cu
# Should show lines 474 and 487

# Rebuild kernel
cd bench
python3 build_v3_release.py
```

### If Sanitizer Fails
```bash
# Run specific tool with verbose output
./scripts/run_sanitizers.sh --tool synccheck --shape small

# Check log
cat artifacts/sanitizers/v3_synccheck_small.log | grep "ERROR"
```

### If CI Gate Fails
```bash
# Check what failed
cat ci_gate_*.log

# Run stages individually
python3 tests/test_sdpa_parity.py  # Stage 1
python3 scripts/bench_sdpa_baseline.py --shapes canonical  # Stage 2
```

---

## 📈 Cost Tracking

**This session**: ~$0.01 (2 minutes GPU)  
**Next session**: ~$0.40 (2-3 hours validation)  
**After validation**: ~$2.20-3.40 (optimization + profiling)

**Save costs**: Always stop GPU when done!
```bash
# On local machine (NOT on GPU)
gcloud compute instances stop cudadent42-l4-dev --zone=us-central1-a
```

---

## 🎯 After Validation: Next Steps

### If Validation Succeeds ✅
1. Review speedup analysis from robust-kbench
2. Document Fix A impact
3. Proceed to Phase 5: Nsight Compute profiling
4. Begin optimization loop (Phase 4)

### If Validation Fails ❌
1. Review logs and error messages
2. Check Fix A application
3. Run DEBUG mode build
4. Consider Fix B (2-stage pipeline) alternative

---

## 📝 Quick Reference

**Branch**: `feature/evoengineer-rbk-l4-optim`  
**GPU**: `cudadent42-l4-dev` (us-central1-a)  
**Rate**: $0.20/hour

**Documentation**:
- Full session report: `EVOENGINEER_RBK_SESSION_COMPLETE.md`
- Progress update: `SESSION_PROGRESS_UPDATE.md`
- This guide: `NEXT_SESSION_START_HERE.md`

**Support**:
- repo_specific_rule: Check for CUDA kernel best practices
- Knowledge transfer: `cudadent42/KNOWLEDGE_TRANSFER_EVOENGINEER_RBK.md`

---

**Ready to start?** → Execute Step 1 above! 🚀
