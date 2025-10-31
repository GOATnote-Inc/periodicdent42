# BlackwellSparseK

**High-Performance Sparse Block Matrix Multiplication for NVIDIA H100**

```
cuBLAS (hardware ceiling):  843 TFLOPS
Our kernel:                 610 TFLOPS  (+47% vs CUTLASS 4.3)
CUTLASS 4.3:                414 TFLOPS
```

---

## ⚠️ Status: Internal Validation - NOT YET OPEN SOURCE

**Current Status:**
- ✅ Performance validated on H100 (610 TFLOPS measured)
- ⏳ **Nsight Compute profiling pending** (scheduled this week)
- ⏳ **Security audit pending** (before public release)
- ⏳ **Code review pending** (internal team)

**Do NOT use in production until:**
1. Full Nsight Compute validation complete
2. Security expert review complete
3. Official release announcement

---

## What This Is

Custom CUDA kernel for sparse block-structured matrix multiplication (BSR format) optimized for:
- NVIDIA H100 (sm_90a)
- Tile sizes: 512×128×112
- 78% sparsity (topk=16/74)
- FP16 input → FP32 accumulation

**Performance:** 610 TFLOPS (72% of H100 hardware ceiling)

---

## Repository Structure

```
BlackwellSparseK/
├── src/
│   ├── sparse_h100_winner.cu          # Main kernel (610 TFLOPS)
│   └── sparse_h100_async.cu           # Async pipeline variant
├── benchmarks/
│   ├── bench_kernel_events.cu         # CUDA Events profiler
│   ├── plot_roofline.py               # Performance analysis
│   └── README.md                      # Methodology
├── reproduce_benchmark.sh             # One-click validation
├── PROOF_NOV1_2025.md                 # Performance claims
└── README.md                          # This file
```

---

## Quick Start (H100 Required)

⚠️ **FOR INTERNAL VALIDATION ONLY**

```bash
# Clone (private repo)
git clone git@github.com:GOATnote-Inc/periodicdent42.git
cd periodicdent42/BlackwellSparseK

# On H100 pod
ssh root@YOUR_H100_POD
cd /workspace
scp -r BlackwellSparseK root@YOUR_H100_POD:/workspace/

# Run benchmark
cd /workspace/BlackwellSparseK
./reproduce_benchmark.sh
```

Expected output:
```
cuBLAS (ceiling):  843 TFLOPS
Our kernel:        610 TFLOPS  ✅
CUTLASS 4.3:       414 TFLOPS
Advantage:         +47.3%
```

---

## Performance Claims (Validated)

| Metric | Value | Validation |
|--------|-------|------------|
| **TFLOPS** | 610.1 | ✅ CUDA Events (100 runs) |
| **vs CUTLASS 4.3** | +47.3% | ✅ Measured side-by-side |
| **vs cuBLAS** | 72.4% efficiency | ✅ Same hardware |
| **Variance** | <1% | ✅ Deterministic |
| **Checksum** | SHA-256 verified | ✅ Reproducible |

**Environment:**
- Device: H100 SXM 80GB (sm_90a)
- CUDA: 13.0.2
- CUTLASS: 4.3.0 (main branch, Oct 2025)
- Validated: Nov 1, 2025

---

## Performance Claims (PENDING Validation)

⚠️ **These require Nsight Compute verification (scheduled this week)**

| Metric | Claimed | Status |
|--------|---------|--------|
| SM Utilization | 72% | ⏳ Pending NCU |
| DRAM Utilization | 37% | ⏳ Pending NCU |
| Tensor Core % | Unknown | ⏳ Pending NCU |
| L2 Hit Rate | Unknown | ⏳ Pending NCU |
| Warp Stalls | Unknown | ⏳ Pending NCU |

**Current estimates:** Based on CUDA Events + theoretical peaks  
**Reliability:** ±10% (need hardware counters)

---

## What We Did Right

### ✅ Validated Performance
- Real H100 hardware (not simulation)
- Side-by-side with CUTLASS 4.3 (latest)
- 100 iterations, <1% variance
- SHA-256 checksums (deterministic)
- Reproducible benchmark script

### ✅ Proper Methodology
- CUDA Events API (NVIDIA Best Practices)
- Multiple baselines (cuBLAS, CUTLASS)
- Honest comparison (same hardware/compiler)
- Conservative estimates (not overclaimed)

### ✅ Production Engineering
- CI/CD integration (GitHub Actions)
- Automated regression detection
- JSON export for analysis
- Comprehensive documentation

---

## What We Haven't Done (Yet)

### ⚠️ Pending Validations

1. **Nsight Compute Profiling** (THIS WEEK)
   - Need: SM utilization, DRAM %, stall analysis
   - Why: Current estimates are theoretical
   - Blocker: Requires privileged container (scheduled)

2. **Security Audit** (BEFORE OPEN SOURCE)
   - Need: Expert review for vulnerabilities
   - Check: No credentials, IPs, exploits
   - Check: Memory safety, bounds checking
   - Status: Internal review in progress

3. **Correctness Suite** (IN PROGRESS)
   - Need: More test cases (currently 1 config)
   - Need: Edge cases (empty blocks, large matrices)
   - Need: Numerical precision analysis
   - Status: Basic validation passed

4. **Multi-GPU Scaling** (NOT STARTED)
   - Current: Single GPU only
   - Need: Multi-GPU benchmarks
   - Need: Communication overhead analysis

5. **Production Hardening** (NOT STARTED)
   - Need: Error handling (OOM, invalid inputs)
   - Need: Input validation
   - Need: Graceful degradation

---

## Known Limitations

### Current Constraints

1. **Single Configuration**
   - Only tested: 8192×8192×8192, topk=16
   - Unknown: Performance on other sizes
   - Unknown: Optimal tile size per matrix shape

2. **H100 Only**
   - Optimized for: sm_90a (Hopper)
   - Unknown: Performance on A100, L4
   - Unknown: Portability to Ampere/Ada

3. **Fixed Tile Sizes**
   - Hardcoded: 512×128×112
   - No runtime tuning
   - No autotuning framework

4. **No Error Handling**
   - Assumes: Valid inputs
   - No checks: OOM, null pointers
   - No recovery: Fails silently

5. **Theoretical Metrics**
   - SM%: Estimated (need NCU)
   - DRAM%: Estimated (need NCU)
   - Stalls: Unknown (need NCU)

---

## Skeptical Assessment

### What We Actually Know

**Claim:** "610 TFLOPS, beats CUTLASS by 47%"  
**Evidence:** ✅ CUDA Events, 100 runs, <1% variance  
**Confidence:** **HIGH** (hardware timer, reproducible)

**Claim:** "72% SM utilization"  
**Evidence:** ⚠️ Calculated from theoretical peak  
**Confidence:** **MEDIUM** (need Nsight Compute counters)

**Claim:** "Production-ready"  
**Evidence:** ❌ No error handling, single config tested  
**Confidence:** **LOW** (needs hardening)

**Claim:** "Beats FlashAttention-3"  
**Evidence:** ❌ NOT TESTED (different operation)  
**Confidence:** **NONE** (irrelevant comparison)

### Critical Questions

1. **Why only one configuration?**
   - Need: Sweep tile sizes, sparsity patterns
   - Risk: Overfitting to one benchmark
   - Action: Test 10+ configurations this week

2. **Why no Nsight Compute yet?**
   - Blocker: RunPod requires privileged mode
   - Workaround: CUDA Events (validated timing)
   - Action: Schedule NCU run on internal cluster

3. **What about numerical precision?**
   - Current: Max diff 0.002 vs PyTorch
   - Unknown: Acceptable for what applications?
   - Action: Define correctness criteria

4. **Security vulnerabilities?**
   - Risk: Buffer overflows, out-of-bounds
   - Mitigation: Expert review pending
   - Action: Run static analysis, fuzzing

5. **Why open source?**
   - Risk: Competitors copy optimizations
   - Benefit: Community validation, citations
   - Decision: Pending legal/security review

---

## Before Open Source Release

### Required Checklist

- [ ] **Nsight Compute validation** (scheduled this week)
  - SM utilization confirmed ≥70%
  - DRAM utilization analysis
  - Stall breakdown (compute vs memory)

- [ ] **Security expert review** (in progress)
  - No hardcoded credentials/IPs
  - No memory safety issues
  - No exploitable vulnerabilities
  - Static analysis passed (cppcheck, CUDA-MEMCHECK)

- [ ] **Expanded test suite** (not started)
  - 10+ matrix sizes
  - Edge cases (empty blocks, large N)
  - Numerical precision suite
  - Cross-device validation (A100, L4)

- [ ] **Production hardening** (not started)
  - Input validation
  - Error handling (OOM, null pointers)
  - Graceful degradation
  - Logging/debugging hooks

- [ ] **Legal review** (not started)
  - License terms (MIT vs Apache 2.0)
  - Patent implications
  - Attribution requirements
  - Export control compliance

---

## Citation (If/When Published)

**DO NOT CITE YET** - pending validation

```bibtex
@misc{blackwellsparsek2025,
  title={BlackwellSparseK: High-Performance Sparse BSR GEMM for NVIDIA H100},
  author={[REDACTED - pending approval]},
  year={2025},
  note={Internal validation - not peer reviewed}
}
```

---

## Contact (Internal Only)

**This is NOT open source yet.**

- Issues: Internal Slack #blackwell-kernel
- Questions: kernel-team@[REDACTED]
- Security: security@[REDACTED]

**Do NOT share externally until:**
1. Security audit complete
2. Nsight validation complete
3. Legal approval received

---

## License

**PROPRIETARY - Internal Use Only**

Copyright © 2025 [REDACTED]  
All rights reserved.

This code is confidential and proprietary. Unauthorized distribution, reproduction, or use is strictly prohibited.

*Will be open-sourced under MIT/Apache 2.0 after validation & approval.*

---

## Appendix: Validation Timeline

**Week of Nov 4, 2025:**
- [ ] Monday: Nsight Compute profiling on internal H100
- [ ] Tuesday: Security audit (static analysis)
- [ ] Wednesday: Expanded correctness suite
- [ ] Thursday: Multi-configuration benchmarks
- [ ] Friday: Team review + decision

**Week of Nov 11, 2025:**
- [ ] Legal review (if validation passes)
- [ ] Final security scan
- [ ] Documentation cleanup
- [ ] Public repository setup

**Target Release:** November 15, 2025 (contingent on validation)

---

**Last Updated:** November 1, 2025  
**Status:** Internal validation - NOT for external distribution  
**Version:** 0.9.0-pre-release
