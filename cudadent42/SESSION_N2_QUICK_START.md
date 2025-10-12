# Session N+2 Quick Start Guide

**Purpose**: Get to working benchmark in 15 minutes instead of 60+ minutes  
**Based on**: Pattern 6 (Build Archaeology → Git Bisect) from Session N+1  
**Target**: Reproduce 0.09× speedup baseline, then profile and improve  

---

## 🚀 **15-Minute Baseline (Critical Path)**

### Step 1: Checkout Last Working Commit (2 min)

```bash
cd ~/periodicdent42/cudadent42

# Checkout commit 5b4c0c8 (Oct 12 session with 0.09× speedup)
git checkout 5b4c0c8

# Verify files present
ls -lh setup.py python/flashmoe_science/csrc/build_config.h benches/bench_correctness_and_speed.py
```

**Expected output**: All 3 files should exist

---

### Step 2: Start GPU and Measure PyTorch Baseline (5 min)

```bash
# Start GPU (on local machine)
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a --project=periodicdent42

# Wait 30 seconds for boot
sleep 30

# SSH and measure baseline
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a --project=periodicdent42 --command="
python3 << 'EOF'
import torch
import torch.nn.functional as F

Q = K = V = torch.randn(1, 1, 128, 64, dtype=torch.float16, device='cuda')

# Warmup
for _ in range(10):
    _ = F.scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()

# Measure
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    O = F.scaled_dot_product_attention(Q, K, V)
end.record()
torch.cuda.synchronize()

baseline_ms = start.elapsed_time(end) / 100
print(f'✅ PyTorch baseline @ S=128: {baseline_ms:.3f} ms')
print(f'   Target for 0.5× speedup: < {baseline_ms * 2:.3f} ms (ours)')
print(f'   Target for 1.0× speedup: < {baseline_ms:.3f} ms (ours)')
EOF
"
```

**Expected output**: `PyTorch baseline @ S=128: 0.026 ms` (from Session N+1)

---

### Step 3: Build Extension (5 min)

```bash
# SSH to GPU
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a --project=periodicdent42

# On GPU:
cd ~/periodicdent42/cudadent42

# Clean previous build
python3 setup.py clean --all
rm -rf build/ dist/ *.egg-info flashmoe_science.*.so

# Build (uses explicit instantiations from 5b4c0c8)
python3 setup.py build_ext --inplace 2>&1 | tee build.log

# Check for .so file
ls -lh flashmoe_science/_C.*.so
```

**Expected output**: `flashmoe_science/_C.cpython-310-x86_64-linux-gnu.so` (~14 MB)

---

### Step 4: Run Benchmark (3 min)

```bash
# Still on GPU:
cd ~/periodicdent42/cudadent42

# Run benchmark at S=128
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

python3 benches/bench_correctness_and_speed.py --config small

# Expected output:
# Config: Small (B=1, H=1, S=128, D=64)
# PyTorch: 0.046 ± 0.005 ms
# Ours:    0.313 ± 0.003 ms
# Speedup: 0.15× (15% of PyTorch)
```

**Expected speedup**: **0.09-0.15×** (baseline from Session N)

---

## ✅ **Decision Gates (After 15-Min Baseline)**

### Gate 1: Did Extension Build?
```bash
python3 -c "import sys; sys.path.insert(0, 'python'); import flashmoe_science._C; print('✅ Gate 1 PASSED')"
```

**If fails**: Check `build.log` for errors. Most likely issue: undefined symbols (need explicit instantiations).

---

### Gate 2: Does Shared Memory Fit L4? (≤48KB)
```bash
# Check build_config.h
grep -E "TILE_SIZE|NUM_WARPS" python/flashmoe_science/csrc/build_config.h

# Expected for L4 (commit 5b4c0c8):
# TILE_SIZE_M = 64
# TILE_SIZE_N = 64  
# TILE_SIZE_K = 64
# NUM_WARPS_PER_BLOCK = 8  (256 threads)

# Estimated shared memory: ~40 KB (fits in 48 KB)
```

**If fails**: Error message will say `ptxas error: Entry function uses too much shared data`

---

### Gate 3: Is Kernel Correct? (max_diff < 0.01)
```bash
python3 benches/bench_correctness_and_speed.py --config small | grep "Max diff"

# Expected: Max diff: 0.000xxx (< 0.01)
```

**If fails**: Kernel has bugs. Stop and fix correctness before performance.

---

### Gate 4: Is Speedup ≥ 0.5×?
```bash
python3 benches/bench_correctness_and_speed.py --config small | grep "Speedup"

# Baseline from Session N: 0.09-0.15× (FAIL)
```

**Expected**: ❌ **FAIL** (0.09× < 0.5×) → **STOP OPTIMIZING** → **PROFILE FIRST**

---

## 🔬 **Profiling Phase (Gate 4 Failed)**

### Mandatory: Profile with Nsight Compute

Since speedup < 0.5×, **DO NOT** guess what's wrong. **PROFILE FIRST**.

```bash
# Install Nsight Compute (if not present)
which ncu || sudo apt-get update && sudo apt-get install -y nsight-compute

# Profile tiny config (S=32 for speed)
ncu --set full --target-processes all \
    python3 benches/bench_correctness_and_speed.py --config tiny \
    > profile_report.txt 2>&1

# Key metrics to check:
grep -E "Memory Throughput|Compute Throughput|Occupancy|Launch Overhead" profile_report.txt
```

**Expected findings** (based on Session N analysis):
1. **Low memory throughput** (< 50% of peak) → Memory access pattern issues
2. **Low occupancy** (< 50%) → Not enough active warps
3. **High launch overhead** → Too many small kernel launches (64×64 tiles)

---

## 🎯 **Optimization Phase (Based on Profile)**

### IF Profile Shows: Low Memory Throughput

**Problem**: Memory access pattern not coalesced

**Fix**: Vectorized loads
```cpp
// In flash_attention_science.cu
// Change: scalar loads
float q = Q[idx];

// To: vectorized loads (float4 = 128-bit loads)
float4 q_vec = reinterpret_cast<const float4*>(Q)[idx / 4];
```

**Expected impact**: 1.5-2.0× speedup

---

### IF Profile Shows: Low Occupancy

**Problem**: Too many registers or too much shared memory per block

**Fix**: Reduce register pressure
```cpp
// In build_config.h
// Change: 64×64 tiles (40 KB shared mem)
constexpr int TILE_SIZE_M = 64;

// To: 32×32 tiles (10 KB shared mem) - allows more concurrent blocks
constexpr int TILE_SIZE_M = 32;
```

**Expected impact**: 1.2-1.3× speedup (more blocks in flight)

---

### IF Profile Shows: High Launch Overhead

**Problem**: Too many kernel launches (small tiles)

**Fix**: Fuse loops or increase tile size
```cpp
// In build_config.h
// Change: 64×64 tiles (current)
constexpr int TILE_SIZE_M = 64;

// To: 96×96 tiles (if fits in 48 KB)
constexpr int TILE_SIZE_M = 96;
```

**Trade-off**: Larger tiles = fewer launches but may exceed shared memory

---

## 📊 **Success Criteria**

| Gate | Metric | Target | Action if Fail |
|------|--------|--------|----------------|
| 1 | Extension builds | Yes | Check build.log |
| 2 | Shared memory fits | ≤48 KB | Reduce tile size |
| 3 | Correctness | max_diff < 0.01 | Fix kernel bugs |
| 4 | Performance | Speedup ≥ 0.5× | **PROFILE FIRST** |

**After profiling**:
- Fix **ONE** bottleneck (highest impact from profile)
- Rebuild and re-measure
- If speedup improves → profile again for next bottleneck
- If speedup doesn't improve → revert change, try different fix

---

## ⏱️ **Time Budget (4 hours total)**

| Phase | Time | Description |
|-------|------|-------------|
| **Baseline** | 15 min | Steps 1-4 above (checkout → build → benchmark) |
| **Profiling** | 30 min | Nsight Compute on tiny/small configs |
| **Analysis** | 15 min | Interpret profile, identify bottleneck |
| **Optimization** | 90 min | Fix ONE thing, rebuild, measure |
| **Validation** | 30 min | Re-profile, verify improvement |
| **Documentation** | 60 min | Update learning loop, commit |

**Stop conditions**:
- ✅ Speedup ≥ 1.0× → SUCCESS, document and stop
- ⏱️ 3 hours elapsed → Document progress and stop
- ❌ Speedup getting worse → Revert and try different approach

---

## 🎓 **Key Principles (From Sessions N & N+1)**

1. **Profile before optimize** (80% of optimizations target wrong bottleneck)
2. **Test S=32 first** (if slow on tiny, won't be fast on large)
3. **One variable at a time** (can't isolate impact otherwise)
4. **Git bisect > build archaeology** (5 min vs 60 min)
5. **Speedup < 0.5× → STOP** (profile, don't guess)

---

## 📦 **Files to Check at Commit 5b4c0c8**

These files should exist and be the "last known working" state:

```
cudadent42/
├── setup.py                                  (build script)
├── benches/bench_correctness_and_speed.py    (benchmark)
└── python/flashmoe_science/csrc/
    ├── build_config.h                        (NUM_WARPS=8, TILES=64×64)
    ├── flash_attention_science.cu            (kernel with explicit instantiations)
    ├── flash_attention_wrapper.cpp           (host wrapper)
    └── bindings.cpp                          (PyTorch bindings)
```

**Verification command**:
```bash
git show 5b4c0c8:cudadent42/python/flashmoe_science/csrc/build_config.h | grep -E "TILE_SIZE|NUM_WARPS"
```

Expected output:
```
constexpr int NUM_WARPS_PER_BLOCK = 8;
constexpr int TILE_SIZE_M = 64;
constexpr int TILE_SIZE_N = 64;
constexpr int TILE_SIZE_K = 64;
```

---

## 🚨 **Common Pitfalls (From Session N+1)**

### Pitfall 1: Preemptible GPU Terminates Mid-Build

**Symptom**: SSH command freezes for 10+ minutes

**Solution**: Check GPU status first
```bash
status=$(gcloud compute instances describe cudadent42-l4-dev --zone=us-central1-a --format="value(status)")
if [ "$status" != "RUNNING" ]; then
  echo "⚠️  GPU not running, restarting..."
  gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
  sleep 30
fi
```

---

### Pitfall 2: Import Fails with Undefined Symbol

**Symptom**: `undefined symbol: _ZN8flashmoe23flash_attention_forward...`

**Cause**: Missing explicit template instantiations

**Solution**: Already fixed in commit 5b4c0c8 (has explicit instantiations)

---

### Pitfall 3: Spending 60+ Min on Build Debugging

**Symptom**: Trying random fixes without checking git history

**Solution**: Use this guide! Commit 5b4c0c8 is tested and working.

---

## 🎯 **Expected Outcomes**

### Baseline (15 min)
- ✅ Extension builds
- ✅ Benchmark runs
- ✅ Speedup: 0.09-0.15× (reproducible baseline)

### After Profiling (45 min)
- ✅ Identified bottleneck (likely: memory throughput or launch overhead)
- ✅ Prioritized fix (highest impact)

### After Optimization (3 hours)
- 🎯 **Target**: Speedup ≥ 0.5× (5× better than baseline)
- 🚀 **Stretch**: Speedup ≥ 1.0× (match PyTorch)

---

## 📝 **Session N+2 Checklist**

**Before starting**:
- [ ] Read this guide
- [ ] Read `CUDA_QUICK_REFERENCE.md`
- [ ] Check GPU is running
- [ ] Have 4 uninterrupted hours

**During session**:
- [ ] Checkout commit 5b4c0c8 (don't skip!)
- [ ] Measure PyTorch baseline first
- [ ] Build extension (15 min target)
- [ ] Run benchmark to verify 0.09× baseline
- [ ] Profile with Nsight Compute (mandatory if < 0.5×)
- [ ] Fix ONE thing based on profile
- [ ] Re-measure and validate improvement

**After session**:
- [ ] Document findings in `CUDA_KERNEL_LEARNING_LOOP.md`
- [ ] Update pattern library if new learnings
- [ ] Stop GPU
- [ ] Commit and push

---

**Last Updated**: October 12, 2025 03:30 AM  
**Next Use**: Session N+2 (apply Pattern 6: git bisect)  
**Expected Time Savings**: 55 minutes (vs Session N+1)  
**Success Criteria**: Speedup ≥ 0.5× (vs 0.09× baseline)

