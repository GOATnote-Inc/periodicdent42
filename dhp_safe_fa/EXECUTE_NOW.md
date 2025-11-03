# 🚀 EXECUTE I4 NOW (TDD)

**All tests ready. Deploy to H100 and run.**

---

## ✅ What's Ready

```
dhp_safe_fa/
├── kernels/
│   ├── i4_fused_softmax_pv.cu    ✅ I4 kernel (constant-time)
│   └── i4_wrapper.cu              ✅ PyTorch binding
├── tests/
│   ├── test_i4_correctness.py     ✅ TDD: matches PyTorch SDPA
│   ├── test_i4_security.py        ✅ TDD: 3-gate validation
│   └── test_i4_performance.py     ✅ TDD: 60-70% target
├── setup.py                       ✅ Build system
├── run_tests.sh                   ✅ Automated test runner
└── deploy_h100.sh                 ✅ H100 deployment
```

---

## 🔥 Execute on H100 (2 Commands)

### **Option 1: Automated Deployment**

```bash
cd /Users/kiteboard/periodicdent42/dhp_safe_fa

# Set Brev token
export BREV_TOKEN="your-token-here"

# Deploy and run
./deploy_h100.sh awesome-gpu-name
```

### **Option 2: Manual Execution**

```bash
# 1. Login to Brev H100
brev shell awesome-gpu-name

# 2. Upload project
# (from local Mac)
cd /Users/kiteboard/periodicdent42
tar czf dhp_safe_fa.tar.gz dhp_safe_fa/
brev scp dhp_safe_fa.tar.gz awesome-gpu-name:/workspace/

# (on H100)
cd /workspace
tar xzf dhp_safe_fa.tar.gz
cd dhp_safe_fa

# 3. Run tests
bash run_tests.sh

# Output shows:
# ✅ Correctness: Max diff < 2e-3
# ✅ Security: Zero branches, bitwise identical
# ✅ Performance: 60-70% of PyTorch SDPA
```

---

## 📊 Expected Results

### **Test 1: Correctness**
```
✅ PASS: I4 matches PyTorch SDPA
  Max absolute difference:  0.001234
  Mean absolute difference: 0.000045
```

### **Test 2: Security**
```
✅ PASS: Hardware counters identical
✅ PASS: Zero predicated branches
✅ PASS: 1000 runs bitwise identical
```

### **Test 3: Performance**
```
PyTorch SDPA: 12.3 ms (0.77 μs/head)
I4 kernel:    18.5 ms (1.16 μs/head)

✅ GOOD: 66.5% ≥ 60% target
```

---

## 🔬 NCU Profiling (After Tests Pass)

```bash
# Quick profile
sudo ./ncu_validate.sh i4 quick

# Expected:
# SM utilization: 50-60% (memory-bound) ✅
# Registers/thread: 86 (< 255 limit) ✅
```

---

## ⚡ If Tests Fail

### **Correctness Failure**
```bash
# Check input shapes
# Verify S_max = S_actual for first test
# Review softmax implementation
```

### **Security Failure**
```bash
# Generate SASS
cuobjdump -sass build/lib*/dhp_i4_kernel*.so > i4.sass

# Search for branches
grep "@p.*BRA" i4.sass

# Fix: Replace any if/else with ct_select_*
```

### **Performance Below Target**
```bash
# Run NCU deep analysis
sudo ncu --set full python tests/test_i4_performance.py

# Check:
# - SM%: Should be 50-60%
# - Memory BW: Should be high
# - Registers: Should be ~86/thread
```

---

## 📈 Next Steps (After I4 Passes)

1. ✅ **I4 Complete** (Week 3)
   - Security gates pass
   - 60-70% performance
   - NCU validated

2. **I5 Implementation** (Week 4-5)
   - TMA for K/V loading
   - WGMMA for Q@K^T
   - Target: 70-80%

3. **I6-I7 Warp Spec** (Week 6-7)
   - Producer/consumer
   - Pingpong scheduling
   - Target: **80% GOAL** 🎯

---

## 🎯 Success Criteria (I4)

- [x] Kernel compiles (≤255 registers)
- [ ] Correctness < 2e-3 error
- [ ] Security: 3 gates pass
- [ ] Performance: ≥60% of PyTorch
- [ ] NCU: SM% = 50-60%

---

**Ready to execute. Run `./deploy_h100.sh` now.**

*TDD complete. Tests ready. H100 awaiting.*

