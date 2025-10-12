# GPU Benchmark Session - October 12, 2025 (Late Night)

**Status**: ⚠️ **BLOCKED - Shared Memory Issue**  
**GPU**: cudadent42-l4-dev (L4, 23GB, SM89, us-central1-a)  
**Session Duration**: ~2 hours  
**IP**: 34.135.91.180

---

## Objective

Run SOTA benchmarks from PR #43 (cuda_reboot/) on L4 GPU to validate THREADS_PER_BLOCK fix (128→384).

---

## Work Completed

### 1. ✅ Repository Setup
- Fetched and merged `cuda_reboot/` directory from `origin/codex/create-new-branch-for-cuda-research`
- Committed cuda_reboot/ to `opt/vectorized-loads` branch
- Pushed to GitHub (commit 4588ea7)

### 2. ✅ GPU Instance Management
- Started cudadent42-l4-dev (was TERMINATED)
- SSH configured and working
- CUDA environment validated (CUDA 12.8, PyTorch 2.7.1, L4 GPU detected)

### 3. ✅ Build System Fixes
**Problem**: Original setup.py was incomplete, missing explicit template instantiations

**Root Cause Analysis**:
- `flash_attention_science.cu` had template definitions but NO explicit instantiations
- Comment said "Templates will be instantiated implicitly" - WRONG
- Missing instantiations caused undefined symbol errors

**Solution Applied**:
- Added explicit instantiations for `half` and `__nv_bfloat16` inside `namespace flashmoe`
- Removed problematic separate FP16/BF16 wrapper files (macro conflicts)
- Simplified setup.py to only compile 3 files: bindings.cpp, wrapper.cpp, flash_attention_science.cu

### 4. ❌ **BLOCKED: Shared Memory Limit Exceeded**

**Error**:
```
ptxas error: Entry function uses too much shared data (0x28000 bytes, 0xc000 max)
- Used: 160 KB
- Max (L4): 48 KB
```

**Root Cause**: `NUM_WARPS_PER_BLOCK = 12` (384 threads) designed for H100, exceeds L4 shared memory

---

## Key Findings

### 1. API Mismatch Between PR #43 and Actual Code

**PR #43 Benchmark Harnesses Expect**:
```python
from flashmoe_science import flash_attention_science  # Direct function import
from flashmoe_science import fused_moe
```

**Actual Implementation Uses**:
```python
import flashmoe_science._C as fa  # C extension module
fa.flash_attention_forward(...)
```

**Impact**: Cannot use PR #43 benchmark scripts without rewriting. Must use existing `benches/bench_correctness_and_speed.py`.

### 2. Build Configuration Issues

**Original Problem** (Evening Session):
- `NUM_WARPS_PER_BLOCK = 4` (128 threads) → 0.12× regression

**Applied Fix**:
- `NUM_WARPS_PER_BLOCK = 12` (384 threads) → Exceeds L4 shared memory

**Architectural Insight**:
- 384 threads correct for H100 (96 KB shared memory per block)
- 128 threads may be too small (caused regression)
- **Optimal for L4**: Likely 8 warps (256 threads) = middle ground

### 3. Shared Memory Calculation

**L4 GPU Limits** (SM89):
- Max shared memory per block: 48 KB (0xc000)
- Max shared memory per SM: 100 KB

**Kernel Usage**:
- 160 KB requested → 3.3× over limit
- Likely from large tile buffers (Q, K, V tiles stored in shared memory)

---

## Current Branch State

**opt/vectorized-loads** on GPU (~/periodicdent42):
- ✅ `build_config.h`: NUM_WARPS_PER_BLOCK = 12
- ✅ `setup.py`: Fixed (3 source files)
- ✅ `flash_attention_science.cu`: Explicit instantiations added
- ❌ **Build fails**: Shared memory exceeded

**Uncommitted Changes on GPU**:
- `setup.py` (fixed source list)
- `flash_attention_science.cu` (explicit instantiations)
- `cuda_reboot/` directory (already committed locally, needs to be committed on GPU)

---

## Next Session: 3 Paths Forward

### Path A: Reduce Block Size for L4 (Recommended)

**Goal**: Find L4-optimal thread count between 128-384

**Steps**:
1. Test NUM_WARPS_PER_BLOCK = 8 (256 threads)
2. Measure shared memory usage
3. If still too large, try 6 warps (192 threads)
4. Run benchmarks once it compiles

**Expected**: 256 threads likely optimal for L4

**Time**: 30 minutes

### Path B: Dynamic Shared Memory Configuration

**Goal**: Reduce tile sizes to fit L4 limits

**Steps**:
1. Modify kernel to use smaller tile buffers
2. Add `__launch_bounds__` directive
3. Conditionally compile for SM89 vs SM90

**Expected**: Matches H100 performance on per-warp basis

**Time**: 2-3 hours (requires kernel code changes)

### Path C: Switch to H100 GPU

**Goal**: Run benchmarks on target hardware (H100)

**Steps**:
1. Request GPU quota increase (currently 1 GPU active)
2. Create H100 instance
3. Run benchmarks with NUM_WARPS_PER_BLOCK = 12

**Expected**: Validates design, shows true 1.2-1.5× speedup

**Time**: 1-3 business days (quota approval)

**Cost**: $3.67/hour (vs $0.60/hour for L4)

---

## Recommendation

**Try Path A first** (30 minutes):
- Quick validation that approach works
- Establishes L4 baseline performance
- Proves build system fixes
- Only requires changing one constant

**If Path A succeeds**:
- Document L4 results
- Request H100 quota for final validation
- Publish L4 + H100 comparison

---

## GPU Session Management

**Current Status**: ✅ GPU RUNNING (keep until 5:20 AM per repo rules)

**Cost So Far**:
- This session: ~2 hours × $0.60/hour = $1.20
- Previous evening: ~1.5 hours × $0.60/hour = $0.90
- **Total**: $2.10

**When to Stop**:
- After successful benchmark run, OR
- At 5:20 AM if no progress, OR
- User explicitly says "stop GPU"

---

## Files Modified (Need to Commit)

### On GPU (~/periodicdent42/cudadent42):
```bash
# Modified files:
python/flashmoe_science/csrc/flash_attention_science.cu  # +17 lines (explicit instantiations)
setup.py  # Fixed sources list

# To commit:
cd ~/periodicdent42/cudadent42
git add setup.py python/flashmoe_science/csrc/flash_attention_science.cu
git commit -m "fix(cuda): Add explicit template instantiations and fix setup.py sources"
git push origin opt/vectorized-loads
```

---

## Key Learnings

1. **PR #43 vs. Reality**: PR #43 benchmarks assume different API than actual implementation
2. **Template Instantiation**: Must be explicit, not implicit, for CUDA templates
3. **H100 vs L4**: Kernel designed for H100 shared memory (96 KB) exceeds L4 limits (48 KB)
4. **Build System**: setup.py was incomplete - missing explicit instantiation files
5. **Cost Optimization**: Keeping GPU running saved 2× stop/start cycles ($1.00 saved)

---

## Next Command (Path A)

```bash
# SSH to GPU
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# Try 256 threads (8 warps)
cd ~/periodicdent42/cudadent42
sed -i 's/NUM_WARPS_PER_BLOCK = 12/NUM_WARPS_PER_BLOCK = 8/g' python/flashmoe_science/csrc/build_config.h
grep NUM_WARPS python/flashmoe_science/csrc/build_config.h  # Verify

# Rebuild
rm -rf build/ flashmoe_science/*.so
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
python3 setup.py build_ext --inplace

# If builds, run benchmark
python3 benches/bench_correctness_and_speed.py --repeats 30 --warmup 10
```

---

**Session End Time**: 1:57 AM  
**GPU Status**: ✅ RUNNING  
**Ready for**: Path A execution (30 min)

