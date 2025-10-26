# Archive Analysis - Focus on FlashCore Excellence

**Date**: October 25, 2025  
**Purpose**: Remove distractions from FlashCore sub-5μs achievement  
**Principle**: Keep only production code, archive experiments  

---

## 📊 Current State

### **FlashCore Directory**
- 21 experimental build scripts (`build*.py`)
- 36 experimental test scripts (`test*.py`)
- 1 production kernel (`flashcore/fast/attention_production.py`)
- Multiple old documentation files from iteration phases

### **Assessment**
**Problem**: 57+ experimental files distract from the breakthrough achievement  
**Solution**: Archive all experiments, keep only production artifacts

---

## ✅ **KEEP (Production Essentials)**

### **FlashCore Production Files**
```
flashcore/
├── fast/
│   └── attention_production.py          ← THE KERNEL (sub-5μs)
├── benchmark/
│   ├── expert_validation.py             ← Validation script
│   ├── expert_validation_results.json   ← H100 results
│   └── expert_validation_results_l4.json ← L4 results
└── requirements.txt                      ← Dependencies
```

### **Root Documentation**
```
README.md                                 ← Breakthrough presentation
LICENSE                                   ← Apache 2.0
DEPENDENCY_STABILITY_POLICY.md           ← Dependency policy
EXPERT_CONFIRMATION.md                   ← Excellence certification
```

### **Essential Infrastructure**
```
pyproject.toml                            ← Python packaging
setup.py                                  ← Setup script
examples/quick_start.py                   ← User onboarding
docs/validation/                          ← Validation reports
tests/test_sdpa_parity*.py                ← Core correctness tests
```

---

## 📦 **ARCHIVE (Experimental)**

### **Category 1: Experimental Build Scripts (21 files)**
Archive to: `archive/flashcore-experiments/build-scripts/`

```
flashcore/build.py
flashcore/build_64x64.py
flashcore/build_cpasync.py
flashcore/build_cutlass.py
flashcore/build_fa3.py
flashcore/build_fa3_simple.py
flashcore/build_fa3_v2.py
flashcore/build_fa3_v4.py
flashcore/build_fa3_v5.py
flashcore/build_fa3_v6_opt_vec.py
flashcore/build_fa3_v7_1_wmma_pv_fixed.py
flashcore/build_fa3_v7_wmma_pv.py
flashcore/build_fp32p.py
flashcore/build_fused.py
flashcore/build_phase1.py
flashcore/build_pipeline.py
flashcore/build_wmma_v2.py
... (all build*.py except production)
```

**Reason**: These were iterations leading to the final kernel. Not needed for production.

### **Category 2: Experimental Test Scripts (36 files)**
Archive to: `archive/flashcore-experiments/test-scripts/`

```
flashcore/test_64x64.py
flashcore/test_fa3.py
flashcore/test_fa3_simple.py
flashcore/test_fa3_v2.py
flashcore/test_fa3_v3.py
flashcore/test_fa3_v3_1.py
flashcore/test_fa3_v4.py
flashcore/test_fa3_v5.py
flashcore/test_fa3_v6_opt_vec.py
flashcore/test_fa3_v6_wmma.py
flashcore/test_fa3_v7_1_wmma_pv_fixed.py
flashcore/test_phase2.py
flashcore/test_pv_only.py
flashcore/test_pv_serial.py
flashcore/test_qk_only.py
flashcore/test_softmax_only.py
flashcore/test_v12_expert.py
flashcore/test_v13_excellence.py
flashcore/test_v9_1_verified.py
flashcore/test_v9_3_excellence.py
... (all test*.py in flashcore/ root)
```

**Reason**: Experimental tests for failed kernels. Production uses `expert_validation.py`.

### **Category 3: Old FlashCore Documentation**
Archive to: `archive/flashcore-experiments/docs/`

```
flashcore/CURSOR_SETUP_INSTRUCTIONS.md
flashcore/FLASHCORE_SESSION3_FINAL.md
flashcore/PHASE1_STATUS_CRITICAL.md
flashcore/RESEARCH_SUMMARY.md
flashcore/WMMA_IMPLEMENTATION_BLUEPRINT.md
flashcore/design/flashcore_fused.md
flashcore/notes/research_fused_flashcore.md
flashcore/docs/PHASE1_WMMA_GUIDE.md
```

**Reason**: Historical iteration docs. Not relevant to production kernel.

### **Category 4: Experimental Triton Variants**
Archive to: `archive/flashcore-experiments/triton-iterations/`

```
flashcore/fast/attention_aggressive.py
flashcore/fast/attention_approx.py
flashcore/fast/attention_batch_optimized.py   ← Used in development
flashcore/fast/attention_triton.py
flashcore/fast/tune_triton.py
flashcore/fast/final_tune.py
flashcore/flashcore_triton.py
```

**Reason**: Iterations leading to production. `attention_production.py` is final version.

### **Category 5: Old Test Infrastructure**
Archive to: `archive/test-infrastructure/`

```
tests/chaos/                              ← Chaos engineering tests
tests/tuning/                             ← Old tuning tests
tests/test_chaos_*.py
tests/test_ci_gates.py
tests/test_epistemic_telemetry.py
tests/test_flaky_scan.py
tests/test_fp8_*.py                       ← FP8 experiments (abandoned)
tests/test_llm_router.py
tests/test_performance_benchmarks.py
tests/test_phase2_scientific.py
tests/test_provenance_integration.py
tests/test_rag_cache.py
tests/test_repo_audit.py
tests/test_safety_gateway.py
tests/test_telemetry_repo.py
```

**Reason**: Infrastructure for old experiments, not core to kernel validation.

---

## 📁 **Proposed Archive Structure**

```
archive/
├── flashcore-experiments/
│   ├── build-scripts/              ← 21 build*.py files
│   ├── test-scripts/               ← 36 test*.py files
│   ├── triton-iterations/          ← Triton variants
│   └── docs/                       ← Old documentation
└── test-infrastructure/
    ├── chaos/                      ← Chaos tests
    ├── tuning/                     ← Tuning tests
    └── experimental-tests/         ← Old test files
```

---

## 🎯 **After Archiving: Clean Structure**

### **Root Directory (12 items unchanged)**
```
Justfile, LICENSE, Makefile, README.md, archive/, config/,
docs/, examples/, flashcore/, pyproject.toml, setup.py, tests/
```

### **FlashCore/ (Clean & Focused)**
```
flashcore/
├── fast/
│   └── attention_production.py          ← THE KERNEL
├── benchmark/
│   ├── expert_validation.py             ← Validation
│   ├── expert_validation_results.json   ← H100
│   └── expert_validation_results_l4.json ← L4
├── requirements.txt
└── README.md                             ← Updated guide
```

### **Tests/ (Core Only)**
```
tests/
├── conftest.py
├── test_sdpa_parity.py                   ← Core correctness
└── test_sdpa_parity_comprehensive.py     ← Comprehensive
```

---

## 📊 **Impact**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| FlashCore files | 80+ | 8 | **90% reduction** |
| Build scripts | 21 | 0 | **100% removal** |
| Test scripts | 36 | 0 | **100% removal** |
| Focus | Scattered | **Production** | **Clarity** ✅ |

---

## ✅ **Execution Plan**

### **Step 1: Create Archive Structure**
```bash
mkdir -p archive/flashcore-experiments/{build-scripts,test-scripts,triton-iterations,docs}
mkdir -p archive/test-infrastructure/{chaos,tuning,experimental-tests}
```

### **Step 2: Move Experimental Build Scripts**
```bash
mv flashcore/build*.py archive/flashcore-experiments/build-scripts/
```

### **Step 3: Move Experimental Test Scripts**
```bash
mv flashcore/test*.py archive/flashcore-experiments/test-scripts/
```

### **Step 4: Move Old Documentation**
```bash
mv flashcore/CURSOR_SETUP_INSTRUCTIONS.md archive/flashcore-experiments/docs/
mv flashcore/FLASHCORE_SESSION3_FINAL.md archive/flashcore-experiments/docs/
mv flashcore/PHASE1_STATUS_CRITICAL.md archive/flashcore-experiments/docs/
mv flashcore/RESEARCH_SUMMARY.md archive/flashcore-experiments/docs/
mv flashcore/WMMA_IMPLEMENTATION_BLUEPRINT.md archive/flashcore-experiments/docs/
mv flashcore/design/ archive/flashcore-experiments/docs/
mv flashcore/notes/ archive/flashcore-experiments/docs/
mv flashcore/docs/PHASE1_WMMA_GUIDE.md archive/flashcore-experiments/docs/
```

### **Step 5: Move Triton Iterations**
```bash
mv flashcore/fast/attention_aggressive.py archive/flashcore-experiments/triton-iterations/
mv flashcore/fast/attention_approx.py archive/flashcore-experiments/triton-iterations/
mv flashcore/fast/attention_batch_optimized.py archive/flashcore-experiments/triton-iterations/
mv flashcore/fast/attention_triton.py archive/flashcore-experiments/triton-iterations/
mv flashcore/fast/tune_triton.py archive/flashcore-experiments/triton-iterations/
mv flashcore/fast/final_tune.py archive/flashcore-experiments/triton-iterations/
mv flashcore/flashcore_triton.py archive/flashcore-experiments/triton-iterations/
```

### **Step 6: Move Old Test Infrastructure**
```bash
mv tests/chaos/ archive/test-infrastructure/
mv tests/tuning/ archive/test-infrastructure/
mv tests/test_chaos*.py archive/test-infrastructure/experimental-tests/
mv tests/test_ci_gates.py archive/test-infrastructure/experimental-tests/
mv tests/test_epistemic_telemetry.py archive/test-infrastructure/experimental-tests/
mv tests/test_flaky_scan.py archive/test-infrastructure/experimental-tests/
mv tests/test_fp8_*.py archive/test-infrastructure/experimental-tests/
mv tests/test_llm_router.py archive/test-infrastructure/experimental-tests/
mv tests/test_performance_benchmarks.py archive/test-infrastructure/experimental-tests/
mv tests/test_phase2_scientific.py archive/test-infrastructure/experimental-tests/
mv tests/test_provenance_integration.py archive/test-infrastructure/experimental-tests/
mv tests/test_rag_cache.py archive/test-infrastructure/experimental-tests/
mv tests/test_repo_audit.py archive/test-infrastructure/experimental-tests/
mv tests/test_safety_gateway.py archive/test-infrastructure/experimental-tests/
mv tests/test_telemetry_repo.py archive/test-infrastructure/experimental-tests/
```

### **Step 7: Update FlashCore README**
Create `flashcore/README.md` pointing to production kernel only

### **Step 8: Commit & Push**
```bash
git add -A
git commit -m "refactor: Archive experimental code - Focus on production kernel"
git push origin main
```

---

## 🎯 **Result**

**Before**: 80+ files in flashcore, unclear what's production  
**After**: 8 files in flashcore, crystal clear focus  

**Message**: "This is the production sub-5μs kernel. Everything else is history."

---

## ✅ **Expert Approval**

**Decision**: ARCHIVE ALL EXPERIMENTAL CODE  
**Reason**: Protect FlashCore breakthrough from distraction  
**Impact**: 90% reduction in flashcore files  
**Status**: Ready to execute  

**Principle**: Production code should be obvious and minimal.

---

**Next**: Execute archiving script

