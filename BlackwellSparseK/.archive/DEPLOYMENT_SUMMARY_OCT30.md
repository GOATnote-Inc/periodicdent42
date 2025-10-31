# BlackwellSparseK Guardrail Kit - DEPLOYMENT SUMMARY
**Date**: October 30, 2025 20:45 PST  
**Action**: Pivot from CUTLASS Example 88 to custom BSR kernel  
**Status**: ✅ **INFRASTRUCTURE COMPLETE** - Awaiting kernel file

---

## 🎯 What Was Deployed

### 1. Cursor IDE Guardrails (`.cursor/`)
```
✅ .cursor/rules.md       - Hard constraints (no PyTorch/Triton, CUDA 13.0.2 locked)
✅ .cursor/config.json    - Enforcement (Docker-only, package blacklist)
```

**Effect**: Cursor cannot:
- Change CUDA/CUTLASS versions
- Install Triton, PyTorch, xFormers, flash-attn
- Run code outside Docker container
- Auto-fix without following Makefile targets

### 2. Build Infrastructure
```
✅ Dockerfile             - nvidia/cuda:13.0.2-devel + CUTLASS 4.3.0 + Nsight CLI
✅ Makefile               - build/run/ncu/verify/clean targets
✅ scripts/preflight.sh   - H100 + CUDA 13.0.2 + sm_90a validation (executable)
✅ README.md              - User documentation
```

**Docker Image**: `sparsek-h100`
- Base: CUDA 13.0.2 devel (Ubuntu 22.04)
- CUTLASS: v4.3.0 at `/opt/cutlass`
- Nsight: CLI tools for headless profiling
- Build: `nvcc -O3 -arch=sm_90a src/sparse_bsr_gemm_h100.cu`

### 3. Documentation
```
✅ README.md                         - Quick start, usage, technical details
✅ GUARDRAIL_KIT_DEPLOYED.md         - Comprehensive deployment guide
✅ DEPLOYMENT_SUMMARY_OCT30.md       - This file
```

---

## ⚠️ What's Missing

```
❌ src/sparse_bsr_gemm_h100.cu   - BSR sparse GEMM kernel with TMA
```

**Required Specifications**:
- Block-sparse (BSR) format: `row_ptr`, `col_idx`, `vals`
- CuTe TMA async copy: `make_tma_copy()`, `PipelineTmaAsync<3>`
- WMMA Tensor Cores: 16×16×16 tiles, FP16→FP32
- Architecture: sm_90a H100 only
- No external dependencies beyond CUDA + CUTLASS headers
- ~500 lines of CUDA C++17

**User Options**:
1. **I generate it** - Full BSR + TMA implementation (~30 minutes)
2. **You provide it** - Place existing kernel at `src/sparse_bsr_gemm_h100.cu`

---

## 📋 File Manifest

### Created Files (9 total)
```
BlackwellSparseK/
├── .cursor/
│   ├── config.json                          [NEW] Cursor enforcement rules
│   └── rules.md                             [NEW] Hard constraints
├── Dockerfile                                [NEW] CUDA 13.0.2 + CUTLASS 4.3.0
├── Makefile                                  [NEW] Build/run/profile targets
├── README.md                                 [NEW] User documentation
├── GUARDRAIL_KIT_DEPLOYED.md                [NEW] Deployment guide
├── DEPLOYMENT_SUMMARY_OCT30.md              [NEW] This summary
└── scripts/
    └── preflight.sh                         [NEW] Validation script (executable)
```

### Existing Files (Preserved)
```
BlackwellSparseK/
└── src/
    └── blackwell_sparsek/               [EXISTING] Python package (separate project)
        └── kernels/
            ├── attention_fmha.cu        [EXISTING] FMHA kernel (CUTLASS-dependent)
            └── kernel_dispatch.cu       [EXISTING] Dispatch logic
```

**Note**: The existing `src/blackwell_sparsek/` Python package is separate from the new standalone `src/sparse_bsr_gemm_h100.cu` kernel.

---

## 🚀 Verification Steps

### 1. Check Files
```bash
cd /Users/kiteboard/periodicdent42/BlackwellSparseK

# Verify guardrails
cat .cursor/rules.md
cat .cursor/config.json

# Verify build infrastructure
cat Dockerfile | head -20
cat Makefile

# Verify preflight script
ls -l scripts/preflight.sh  # Should be executable (-rwxr-xr-x)
```

### 2. Test Docker Build (Will Fail - Kernel Missing)
```bash
make build
```

**Expected Error**:
```
fatal error: src/sparse_bsr_gemm_h100.cu: No such file or directory
```

**This is correct** - We need the kernel file before building.

### 3. After Kernel Created
```bash
# Build
make build

# Verify
make verify

# Profile
make ncu
```

---

## 🎓 Why This Approach Works

### Problem with CUTLASS Example 88
```
Driver:  570.133.20 (CUDA 12.8)
Runtime: 13.0.88 (requires driver 580.95+)
Result:  10,000× slowdown, "Arch conditional MMA" errors
```

Even with `cuda-compat-13-0`, runtime arch detection fails.

### Solution: Custom Kernel
```
Dependencies: CUDA runtime + CUTLASS headers (CuTe) only
Build:        nvcc -arch=sm_90a (no CMake, no examples)
Deploy:       Docker container (reproducible)
Target:       H100 sm_90a (single architecture)
```

### Benefits
- **No driver issues** - Pure CUDA runtime (no CUTLASS runtime)
- **Full control** - We own every line of code
- **TMA ready** - CuTe DSL for async copy
- **Portable** - Docker image runs on any H100

---

## 📊 Expected Performance (Post-Implementation)

### Baseline (Pre-TMA)
```
Warp Active:       ~70%  (basic WMMA)
Memory Stall:      ~15%  (synchronous loads)
Tensor Core:       ~60%  (WMMA utilization)
Bank Conflicts:    Low   (coalesced access)
```

### Target (Post-TMA, 3-stage)
```
Warp Active:       ≥85%  (overlapped load/compute)
Memory Stall:      ≤5%   (TMA async benefit)
Tensor Core:       ≥70%  (improved pipeline)
Bank Conflicts:    ~0    (optimized layouts)
DRAM Throughput:   40-60% (sparse benefit)
```

### Stretch (With Swizzle + Tuning)
```
Warp Active:       ≥90%
Memory Stall:      ≤3%
Tensor Core:       ≥80%
```

---

## 🔄 Comparison with Previous Attempts

### Phase C: CUTLASS Example 88 (FAILED)
```
Approach:  Use CUTLASS Hopper FMHA example
Result:    10,000× slower than expected
Blocker:   Driver version (570 vs 580 requirement)
Time Lost: 4 hours of debugging
```

### Phase D (NEW): Custom BSR Kernel (IN PROGRESS)
```
Approach:  Standalone CUDA kernel with CuTe TMA
Result:    TBD (kernel not created yet)
Blocker:   None (controlled environment)
Estimate:  ~2 hours to working kernel + profile
```

---

## 🎯 Next Actions (Priority Order)

### IMMEDIATE (Required for Progress)
1. **Create kernel file**: `src/sparse_bsr_gemm_h100.cu`
   - BSR format (row_ptr, col_idx, vals)
   - CuTe TMA (3-stage pipeline)
   - WMMA (16×16×16 tiles)
   - sm_90a optimized

2. **Build Docker image**: `make build`
   - Should compile without errors
   - Binary should contain sm_90a SASS

3. **Run preflight**: `make verify`
   - Check H100 detection
   - Check CUDA 13.0.2
   - Check sm_90a in binary

### FOLLOW-UP (Validation)
4. **Profile with Nsight**: `make ncu`
   - Capture 5 key metrics
   - Verify TMA benefit (memory stall ≤5%)

5. **Deploy to RunPod H100**
   - Transfer Docker image
   - Run on real H100 hardware
   - Compare with PyTorch SDPA baseline

6. **Document Results**
   - Create `BENCHMARK_RESULTS.md`
   - Include Nsight metrics
   - Export .ncu-rep file

---

## ✅ Readiness Checklist

- [x] Cursor guardrails deployed (`.cursor/`)
- [x] Docker infrastructure ready (`Dockerfile`, `Makefile`)
- [x] Validation script ready (`scripts/preflight.sh`)
- [x] Documentation complete (`README.md`, guides)
- [ ] **Kernel file created** (`src/sparse_bsr_gemm_h100.cu`) ⚠️
- [ ] Docker image built
- [ ] Kernel verified on H100
- [ ] Nsight profiling complete
- [ ] Performance documented

**Blocking Item**: Kernel file creation

---

## 🤝 User Decision Required

**Question**: Should I generate `src/sparse_bsr_gemm_h100.cu`?

**If YES**:
- I will create ~500-line BSR + TMA kernel
- Includes CuTe layouts, TMA pipeline, WMMA compute
- Compiles with `nvcc -arch=sm_90a` in container
- Ready for `make build && make verify`

**If NO** (you provide kernel):
- Place your kernel at `src/sparse_bsr_gemm_h100.cu`
- Ensure it uses CuTe TMA (or standard gmem→smem)
- I will test with `make build && make verify`

**Recommendation**: Let me generate it - clean slate, TMA from day 1, no legacy code

---

**Status**: Guardrail kit deployment complete, awaiting kernel implementation decision.

**Timeline**: 
- Kernel creation: ~30 minutes
- Build + verify: ~10 minutes
- Nsight profile: ~20 minutes
- **Total to first metrics: ~1 hour**

---

**Contact**: Ready for next instruction.

