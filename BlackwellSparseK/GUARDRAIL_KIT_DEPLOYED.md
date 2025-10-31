# BlackwellSparseK Guardrail Kit - DEPLOYMENT COMPLETE
**Date**: October 30, 2025  
**Status**: ✅ **GUARDRAILS ACTIVE** - Cursor constrained to container-only workflow

---

## 🎯 Mission Recalibration

**ABANDONED**: CUTLASS Example 88 (Hopper FMHA)
- **Reason**: Driver compatibility hell (570.133 vs 580.95+ requirement)
- **Symptoms**: 10,000× slowdown, "Arch conditional MMA" errors
- **Lesson**: CUTLASS examples are NOT production deployment targets

**NEW PATH**: Custom BSR (Block Sparse Row) kernel with TMA
- **Foundation**: Direct CUDA + CuTe + WMMA (no CUTLASS dependencies)
- **Target**: H100 sm_90a only, CUDA 13.0.2, CUTLASS 4.3.0 (CuTe headers only)
- **Deployment**: Docker container as single source of truth

---

## 📦 Files Created

### 1. Cursor Guardrails

**`.cursor/rules.md`**
- Hard constraints on toolchain (CUDA 13.0.2, CUTLASS 4.3.0, sm_90a)
- Banned packages (Triton, PyTorch, FA3)
- Allowed commands (make build/run/ncu/verify/clean)
- Failure modes to avoid

**`.cursor/config.json`**
- Enforces Docker-only execution
- Denies problematic packages
- Sets compile command to container build
- Disables lint-only mode (forces real execution)

### 2. Build Infrastructure

**`Dockerfile`**
```dockerfile
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04
- CUTLASS 4.3.0 (git clone v4.3.0 tag)
- Nsight Compute CLI (headless profiling)
- Builds: nvcc -O3 -std=c++17 -arch=sm_90a src/sparse_bsr_gemm_h100.cu
```

**`Makefile`**
```makefile
build   - Build Docker image + kernel
run     - Execute kernel in container
ncu     - Profile with Nsight Compute (5 key metrics)
verify  - Run preflight checks + kernel
clean   - Remove Docker image
```

**`scripts/preflight.sh`** (executable)
```bash
- Verify CUDA 13.0.2
- Verify H100 GPU
- Verify CUTLASS present
- Verify binary contains sm_90a SASS
```

### 3. Source Structure

```
BlackwellSparseK/
├── .cursor/
│   ├── rules.md              ✅ Created
│   └── config.json           ✅ Created
├── Dockerfile                ✅ Created
├── Makefile                  ✅ Created
├── scripts/
│   └── preflight.sh          ✅ Created (executable)
└── src/
    └── sparse_bsr_gemm_h100.cu   ⚠️ MISSING - NEEDS CREATION
```

---

## ⚠️ CRITICAL: Missing Kernel File

**Status**: `src/sparse_bsr_gemm_h100.cu` does not exist

**Options**:
1. **Create from scratch** - I can generate the BSR + TMA kernel with:
   - Block-sparse (BSR) layout
   - CuTe TMA async copy (3-stage pipeline)
   - WMMA Tensor Cores (16x16x16 tiles)
   - FP16 input, FP32 accumulator
   - sm_90a H100 optimization
   - ~500 lines, compiles with nvcc -arch=sm_90a

2. **User provides** - You mentioned a working BSR kernel that should be preserved

**Recommendation**: Generate kernel from scratch with TMA patch already applied

---

## 🎯 Kernel Specification (if generated)

### Architecture
```
Block Size: 128×128 CTA (256 threads)
Warp Tile:  64×64 (4 warps)
WMMA Tile:  16×16×16 (FP16→FP32)
Pipeline:   3-stage TMA overlapped (STAGES=3)
Shared Mem: ~36 KB (smemA[3][BM*BK] + smemB[3][BK*BN])
```

### Key Features
- **BSR format**: `row_ptr`, `col_idx`, `vals` (standard sparse matrix format)
- **TMA loads**: CuTe `make_tma_copy()` + `PipelineTmaAsync<3>`
- **Producer/Consumer**: Explicit `producer_acquire/commit`, `consumer_wait/release`
- **WMMA compute**: Native `nvcuda::wmma` loads/stores/mma_sync
- **Vectorized epilogue**: `float4` stores to global memory

### Expected Nsight Metrics (Post-TMA)
```
sm__warps_active:                      ≥ 85%
smsp__stall_memory_dependency:         ≤ 5%
sm__pipe_tensor_cycles_active:         ≥ 70%
l1tex__data_bank_conflicts:            ~0
dram__throughput:                      40-60% (sparse benefit)
```

---

## 🚀 Usage Workflow

### 1. Create Kernel (REQUIRED)
```bash
# Option A: Generate kernel
# I create src/sparse_bsr_gemm_h100.cu with TMA patch

# Option B: You provide kernel
# Place your BSR kernel at src/sparse_bsr_gemm_h100.cu
```

### 2. Build and Verify
```bash
cd /Users/kiteboard/periodicdent42/BlackwellSparseK
make verify
```

**Expected Output**:
```
[Preflight] Verifying H100 + CUDA 13.0.2 + CUTLASS 4.3.0 + sm_90a
[Preflight] OK
[Kernel] Running BSR sparse GEMM...
[Kernel] Performance: X.XX TFLOPS, Y.YY ms
```

### 3. Profile with Nsight
```bash
make ncu
```

**Expected Output**:
```
ncu: Profiling "./sparse_h100"
  sm__warps_active.avg.pct_of_peak_sustained_active       (%)  XX.X
  smsp__stall_memory_dependency.avg.pct                   (%)  X.X
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_...      (%)  XX.X
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared...           X
  dram__throughput.avg.pct_of_peak_sustained_elapsed     (%)  XX.X
```

### 4. Deploy to RunPod H100
```bash
# Build image locally
make build

# Save image
docker save sparsek-h100 | gzip > sparsek-h100.tar.gz

# Copy to RunPod
scp -P 17322 sparsek-h100.tar.gz root@157.66.254.40:/workspace/

# Load and run on H100
ssh -p 17322 root@157.66.254.40
docker load < /workspace/sparsek-h100.tar.gz
docker run --gpus all --rm sparsek-h100
```

---

## 📊 Success Criteria

### Tier 1 (Baseline) - Working Kernel
- ✅ Compiles with nvcc -arch=sm_90a in container
- ✅ Runs on H100 without errors
- ✅ Preflight checks pass
- ✅ Produces valid output

### Tier 2 (Performance) - TMA Benefit Visible
- ✅ `sm__warps_active` ≥ 80%
- ✅ `smsp__stall_memory_dependency` ≤ 8%
- ✅ `sm__pipe_tensor_cycles_active` ≥ 65%
- ✅ Bank conflicts near zero

### Tier 3 (Production) - Portfolio Ready
- ✅ Nsight report exported (.ncu-rep)
- ✅ Performance documented vs. PyTorch SDPA
- ✅ README with build/run instructions
- ✅ Docker image reproducible on any H100

---

## 🔒 Guardrail Enforcement

### What Cursor CANNOT Do
❌ Install Triton, PyTorch, xFormers, flash-attn  
❌ Change CUDA version (13.0.2 is LOCKED)  
❌ Change CUTLASS version (4.3.0 is LOCKED)  
❌ Change architecture (sm_90a is LOCKED)  
❌ Run commands outside Docker container  
❌ "Auto-fix" kernel code without approval  

### What Cursor CAN Do
✅ Edit `src/sparse_bsr_gemm_h100.cu` if it compiles  
✅ Run `make build`, `make run`, `make ncu`, `make verify`  
✅ Add Nsight metrics or improve TMA pipeline  
✅ Improve CuTe layout/swizzling within sm_90a constraints  

### Cursor's Allowed Prompt
```
You must follow `.cursor/rules.md`. Do not install Triton, PyTorch, or FlashAttention.
Do not change CUDA or CUTLASS versions. Only operate through the Makefile and Dockerfile.

Task: Run `make verify`, then `make ncu`. If either fails, show the failing command
output and the *minimal* diff to fix it without changing the toolchain/library
choices, architecture flags, or containerization. Do not suggest alternative libraries.
```

---

## 📝 Next Actions

### IMMEDIATE (Required)
1. **Create `src/sparse_bsr_gemm_h100.cu`**
   - Option A: I generate BSR + TMA kernel (~500 lines)
   - Option B: You provide existing BSR kernel

2. **Test Build**
   ```bash
   make build
   ```

3. **Verify Locally**
   ```bash
   make verify
   ```

### NEXT (Once kernel works)
4. **Profile with Nsight**
   ```bash
   make ncu > nsight_report.txt
   ```

5. **Deploy to RunPod H100**
   - Build image
   - Transfer to H100
   - Run verification
   - Capture performance metrics

6. **Document Performance**
   - Create `BENCHMARK_RESULTS.md`
   - Include Nsight metrics
   - Compare with PyTorch SDPA baseline

---

## 🎓 Key Learnings Applied

### From CUTLASS Example 88 Failure
1. **Don't use examples as production code** - They're tutorials, not deployable kernels
2. **Driver compatibility is critical** - CUDA 13.0.2 requires driver 580.95+, pod has 570.133
3. **Example mode != Production mode** - Examples have validation overhead
4. **Container isolation is mandatory** - Host environment drift breaks everything

### From flashcore Success (October 25, 2025)
1. **H100 hardware works perfectly** - 0.451 μs/head achieved previously
2. **Custom kernels beat library code** - When tuned correctly
3. **TMA + WMMA on sm_90a is proven** - We know this architecture works

### From EvoEngineer Methodology
1. **Measure, don't guess** - Nsight metrics drive optimization
2. **Iterate incrementally** - TMA first, then swizzle, then tuning
3. **Preserve working code** - Don't rewrite, patch

---

## 🚨 If Build Fails

### Check 1: Docker Daemon
```bash
docker ps  # Should show running containers
```

### Check 2: NVIDIA Docker Runtime
```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi
```

### Check 3: Kernel File Exists
```bash
ls -l src/sparse_bsr_gemm_h100.cu
```

### Check 4: Preflight Script Permissions
```bash
chmod +x scripts/preflight.sh
```

---

## ✅ Deployment Status

- [x] `.cursor/rules.md` created (guardrails active)
- [x] `.cursor/config.json` created (Docker enforcement)
- [x] `Dockerfile` created (CUDA 13.0.2 + CUTLASS 4.3.0)
- [x] `Makefile` created (build/run/ncu/verify targets)
- [x] `scripts/preflight.sh` created (validation gate)
- [ ] `src/sparse_bsr_gemm_h100.cu` **NEEDS CREATION**

**Status**: Ready for kernel creation

---

**USER DECISION REQUIRED**: 
Should I generate `src/sparse_bsr_gemm_h100.cu` with BSR + TMA implementation (~500 lines, production-ready), or will you provide the kernel file?

