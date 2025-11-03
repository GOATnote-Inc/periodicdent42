# DHP-Safe FlashAttention
**Constant-Time Attention with Burn Methodology**

**Status**: 🚧 Foundation Complete, Ready for I4 Implementation  
**Approach**: Expert-reviewed + NCU-driven systematic iteration  
**Target**: 80% of FlashAttention-3 with zero timing leaks

---

## 🔥 **Burn Methodology Applied**

This project applies the **exact same NCU-driven systematic iteration** that achieved **8.3× speedup** in BlackwellSparseK:

| Burn Lesson | DHP Application |
|-------------|-----------------|
| **NCU is mandatory** | Profile every iteration with hardware counters |
| **SM% reveals truth** | Track tensor core utilization, not just latency |
| **Systematic wins** | I4→I14 iterations, one change at a time |
| **Library matters** | Use expert-corrected CUTLASS 4.3 + CuTe APIs |
| **Know limits** | Target 80% of FA3 (realistic, achievable) |

---

## 📊 **Performance Targets (H100)**

| Milestone | Target TFLOPS | % of FA3 | Timeline |
|-----------|---------------|----------|----------|
| **I4: Fused Softmax+PV** | 450-520 | 60-70% | Week 3 |
| **I5: Single Kernel** | 520-590 | 70-80% | Week 5 |
| **I6-I7: Warp Spec** | 590-630 | 80-85% | Week 7 |
| **I8-I13: Optimized** | 630-665 | 85-90% | Week 10 |

**Baseline**: PyTorch SDPA ~12 μs/head (run `benchmarks/bench_baseline.py`)

---

## 🏗️ **Project Structure**

```
dhp_safe_fa/
├── include/                    # Foundation utilities
│   ├── dhp_ct_enhanced.cuh    # Constant-time primitives
│   ├── tma_utils.cuh          # TMA descriptor helpers (§1.1)
│   └── barrier_utils.cuh      # Hopper barriers (§1.3)
│
├── kernels/                    # Iteration implementations
│   └── i4_fused_softmax_pv.cu # I4: Fused Softmax+PV
│
├── benchmarks/                 # Performance measurement
│   └── bench_baseline.py      # PyTorch SDPA baseline
│
├── tests/                      # Security validation
│   └── security_validate.sh   # 3-gate security methodology
│
├── audits/                     # NCU profiles & SASS dumps
│   └── (generated artifacts)
│
└── ncu_validate.sh            # NCU profiling script

```

---

## 🚀 **Quick Start (Burn Methodology)**

### **Step 1: Measure Baseline (Like Our Iter 0)**

```bash
cd /Users/kiteboard/periodicdent42/dhp_safe_fa

# Establish PyTorch SDPA baseline
python3 benchmarks/bench_baseline.py

# Expected output:
# Per-head latency: ~12 μs/head
# Target for DHP I4: ~20 μs/head (60% of baseline)
```

### **Step 2: Compile I4 Kernel**

```bash
# Compile with register info (like burn iterations)
/usr/local/cuda-13.0/bin/nvcc -O3 -std=c++17 -arch=sm_90a \
    --ptxas-options=-v \
    -I./include \
    -c kernels/i4_fused_softmax_pv.cu \
    -o build/i4_fused_softmax_pv.o

# Verify register usage ≤ 255/thread
```

### **Step 3: Security Validation (3-Gate)**

```bash
# Run security validation before performance
chmod +x tests/security_validate.sh
./tests/security_validate.sh i4_fused_softmax_pv

# Must pass all 3 gates:
# ✅ Hardware counters identical
# ✅ Zero predicated branches
# ✅ Bitwise reproducible
```

### **Step 4: NCU Profiling (Burn Methodology)**

```bash
# Profile with NCU (requires sudo)
./ncu_validate.sh i4 quick

# Compare SM% to baseline
# Target: 50-60% (memory-bound, as expected)
```

---

## 🔒 **Security Methodology (3-Gate)**

Every iteration MUST pass these gates before proceeding:

### **Gate 1: Hardware Counter Differential**
```bash
# Run kernel with two different inputs
# Verify ALL hardware counters are identical:
# - dram__bytes_read.sum
# - dram__bytes_write.sum
# - smsp__sass_thread_inst_executed.sum
```

### **Gate 2: SASS Branch Analysis**
```bash
# Generate SASS disassembly
cuobjdump -sass build/i4.cubin > audits/i4.sass

# Verify ZERO predicated branches
grep "@p.*BRA" audits/i4.sass  # Must return nothing
```

### **Gate 3: Bitwise Reproducibility**
```bash
# Run kernel 1000 times
# Verify outputs are bitwise identical (not just numerically close)
```

**If ANY gate fails → STOP. Fix security before performance.**

---

## 📚 **Documentation Reference**

Attached expert-reviewed documents (97KB total):

1. **[EXPERT_CORRECTIONS.md](../Downloads/files11.2.25.711/EXPERT_CORRECTIONS.md)** - **READ FIRST**
   - Priority 1: CuTe TMA API fixes
   - Priority 1: WGMMA partitioning corrections
   - Priority 1: Register pressure calculations

2. **[IMPLEMENTATION_READINESS.md](../Downloads/files11.2.25.711/IMPLEMENTATION_READINESS.md)** - Timeline & expectations

3. **[SECURITY_ANNOTATED_PSEUDOCODE.md](../Downloads/files11.2.25.711/SECURITY_ANNOTATED_PSEUDOCODE.md)** - Constant-time patterns

4. **[DHP_SAFE_ITERATION_PLAN_I4_I14.md](../Downloads/files11.2.25.711/DHP_SAFE_ITERATION_PLAN_I4_I14.md)** - Iteration roadmap

---

## 🎯 **Success Criteria (Burn-Validated)**

### **I4 Complete** (Week 3)
- ✅ Compiles with ≤255 registers/thread
- ✅ Passes all 3 security gates
- ✅ Achieves 60-70% of PyTorch SDPA
- ✅ NCU-validated SM% = 50-60%

### **I5 Complete** (Week 5)
- ✅ Single kernel (no global memory writes)
- ✅ TMA + WGMMA working correctly
- ✅ Achieves 70-80% of PyTorch SDPA
- ✅ Tensor core utilization >35%

### **Final Goal** (Week 10)
- ✅ 80% of FlashAttention-3 (590+ TFLOPS)
- ✅ Zero timing leaks (all gates pass)
- ✅ Production-ready (CI/CD, Docker, docs)

---

## 💡 **Key Insights from Burn**

1. **NCU is non-negotiable**: Our burn iterations showed cuBLAS 21% SM vs CUTLASS 8% SM
2. **Small problems need small tiles**: 64×64×64 will work better than 128×256×64
3. **Systematic iteration beats guesswork**: 9 iterations found optimal solution
4. **Know when 80% is excellent**: Like our 8.3× vs 10× goal

---

## 🔗 **Related Projects**

- **BlackwellSparseK**: Burn methodology source (`NCU_FINAL_RESULTS.md`)
- **FlashAttention-3**: Performance target (740 TFLOPS baseline)
- **CUTLASS 4.3**: Expert-corrected API usage

---

## 📞 **Support**

**Stuck on**:
- CuTe APIs → `EXPERT_CORRECTIONS.md` §1.1-1.2
- Security → `SECURITY_ANNOTATED_PSEUDOCODE.md`
- Performance → Run `ncu_validate.sh` and analyze SM%

**Timeline slipping?**:
- First implementation: 60-70% is expected
- 80% is the goal, 85% is stretch
- Don't stress about 90%+

---

**Ready to start?** Run `python3 benchmarks/bench_baseline.py` to establish baseline.

**Status**: ✅ Foundation complete, ready for I4 implementation  
**Next**: Implement I4 kernel with expert corrections  
**Timeline**: 10-12 weeks to 80% of FA3 (realistic, achievable)

---

*Built with burn methodology from BlackwellSparseK NCU iterations*  
*Expert-reviewed by Senior Principal Engineer (15+ yrs @ NVIDIA)*

