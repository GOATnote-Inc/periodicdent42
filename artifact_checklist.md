# Artifact Evaluation Checklist
## ICSE/ISSTA/SC Style Review

**Paper Title**: GOATnote Autonomous R&D Intelligence Layer: Evidence Audit  
**Submitted**: 2025-10-06  
**Artifact Type**: Source Code + Data + CI Infrastructure  
**Repository**: https://github.com/GOATnote-Inc/periodicdent42

---

## 1. Artifact Availability

### 1.1 Public Access
- ✅ **Repository**: https://github.com/GOATnote-Inc/periodicdent42
- ✅ **License**: MIT License (see LICENSE file)
- ✅ **Persistent Archive**: GitHub repository + local snapshots
- ⏳ **DOI**: Not yet assigned (recommended: Zenodo for archival)

### 1.2 Content Completeness
- ✅ **Source Code**: All scripts in `scripts/` directory
- ✅ **Configuration**: `flake.nix`, `pyproject.toml`, `.github/workflows/*.yml`
- ✅ **Test Suites**: `tests/chaos/` (15 tests), `tests/test_phase2_scientific.py`
- ✅ **Documentation**: 13 comprehensive guides (8,500+ lines)
- ✅ **Training Data**: `training_data.json` (N=100, synthetic)
- ✅ **Trained Model**: `test_selector.pkl` (254 KB) + `test_selector.json` (metadata)
- ⏳ **Production Data**: Not included (privacy/synthetic baseline used)

---

## 2. Reproducibility Instructions

### 2.1 Hardware Requirements
- **OS**: Ubuntu 22.04+ or macOS 12+ (tested)
- **CPU**: 2+ cores (recommended 4+)
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 10 GB free space
- **Network**: Required for package downloads (Nix cache, Cloud Storage)
- **GPU**: Not required (all tests run on CPU)

### 2.2 Software Dependencies
```bash
# Core dependencies (all platforms)
- Python 3.12
- Git
- curl, jq (for scripts)

# Optional (for full reproduction)
- Nix package manager (for C1 hermetic builds)
- Docker (for container builds)
- PostgreSQL 15 (for database tests)
- Google Cloud SDK (for Cloud Storage access)
```

### 2.3 Installation Steps (Fresh Clone)
```bash
# 1. Clone repository
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42

# 2. Install Python dependencies (core)
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Install development dependencies
pip install -e ".[dev]"

# 4. Verify installation
python --version  # Should be 3.12.x
pytest --version  # Should be 8.x+
python -c "import sklearn; print(sklearn.__version__)"  # Should be 1.3.2

# 5. Optional: Install Nix (for C1 hermetic builds)
sh <(curl -L https://nixos.org/nix/install) --daemon
```

### 2.4 Deterministic Seeds
All random processes use **fixed seeds** for reproducibility:
- **ML Training**: Random seed specified in `scripts/train_test_selector.py` (line 45)
- **Chaos Testing**: `--chaos-seed 42` flag for pytest
- **Bootstrapping**: NumPy seed set in evaluation scripts

**Important**: Results may vary slightly across platforms due to floating-point differences, but should be within 1% of reported values.

---

## 3. Replication: Claim-by-Claim

### 3.1 C1: Hermetic Builds (5-30 min)

**Prerequisites**: Nix installed (skip if not available)

```bash
# Quick check (no Nix)
python scripts/recompute_build_stats.py
cat reports/build_stats.csv

# Full replication (with Nix)
nix flake check                    # Verify configuration
nix build .#default -L             # Build hermetically (1st time)
HASH1=$(nix path-info ./result --json | jq -r '.[].narHash')
rm -rf result

nix build .#default -L             # Rebuild
HASH2=$(nix path-info ./result --json | jq -r '.[].narHash')

# Compare (should be identical)
echo "Build 1: $HASH1"
echo "Build 2: $HASH2"
[ "$HASH1" == "$HASH2" ] && echo "✓ Bit-identical" || echo "✗ Mismatch"
```

**Expected Output**:
- `flake_exists: True`
- `flake_lines: 322`
- `ci_lines: 252`
- Bit-identical rebuild (if Nix available)

### 3.2 C2: ML Test Selection (5 min)

```bash
# Evaluate model on synthetic data
python scripts/eval_test_selection.py --output reports/ml_eval.json

# View results
cat reports/ml_eval.json | jq

# Expected key metrics:
# - cv_f1_mean: 0.449 ± 0.161
# - training_f1: 0.909
# - ci_time_reduction: 10.3%
# - data_type: "synthetic_baseline"
```

**Expected Output**:
```json
{
  "cv_f1_mean": 0.449,
  "cv_f1_std": 0.161,
  "ci_time_estimate": {
    "reduction_percent": 10.3
  },
  "warnings": [
    "Model trained on synthetic data (N=100)"
  ]
}
```

### 3.3 C3: Chaos Engineering (2-5 min)

```bash
# Run chaos tests (reproducible with seed)
pytest tests/chaos/test_chaos_examples.py --chaos --chaos-rate 0.10 --chaos-seed 42 -v

# Expected: 14/15 passed (93.3%)
# 95% CI: [0.75, 0.99]
```

**Expected Output**:
```
======================== test session starts ========================
tests/chaos/test_chaos_examples.py::test_fragile_api_call FAILED
tests/chaos/test_chaos_examples.py::test_resilient_api_call PASSED
...
==================== 14 passed, 1 failed in 2.3s ====================
```

### 3.4 C4: Continuous Profiling (5-10 min)

```bash
# Profile a validation script
python scripts/profile_validation.py \
  --script scripts/validate_stochastic.py \
  --output validate_stochastic

# Analyze bottlenecks
python scripts/identify_bottlenecks.py \
  artifacts/profiling/validate_stochastic_*.svg

# Expected: "No significant bottlenecks found" (script is fast)
```

**Expected Output**:
- Flamegraph SVG generated in `artifacts/profiling/`
- Runtime: ~0.2s
- No functions >1% of total time

---

## 4. Data Availability

### 4.1 Training Data
- **File**: `training_data.json` (55 KB)
- **Size**: N=100 records
- **Format**: JSON array of test execution records
- **Privacy**: Synthetic data (no real test failures)
- **License**: MIT (same as code)

### 4.2 Model Artifacts
- **File**: `test_selector.pkl` (254 KB)
- **Type**: RandomForestClassifier (scikit-learn 1.3.2)
- **Metadata**: `test_selector.json` (292 bytes)
- **Deployment**: `gs://periodicdent42-ml-models/` (public read)

### 4.3 Flamegraphs
- **Location**: `artifacts/performance_analysis/*.svg`
- **Count**: 2 (validate_rl_system, validate_stochastic)
- **Format**: SVG (interactive in browser)
- **Size**: ~100-200 KB each

### 4.4 Missing Data (Privacy/Ethics)
- ❌ **Real test failure data**: Proprietary, replaced with synthetic
- ❌ **Production incident logs**: Privacy-sensitive, not included
- ❌ **Real CI run data**: Not yet available (ML just enabled)

---

## 5. Expected Outputs & Checksums

### 5.1 Recomputation Outputs

| Output File | Size | Checksum (SHA256) | Notes |
|-------------|------|-------------------|-------|
| `reports/build_stats.csv` | ~200 bytes | TBD | Varies by platform/Nix version |
| `reports/ml_eval.json` | ~1 KB | TBD | Deterministic with seed |
| `reports/index.md` | ~5 KB | Fixed | Static documentation |

**Note**: Checksums not provided due to platform/timestamp variability. Instead, verify **metric ranges**:
- `cv_f1_mean`: 0.40 to 0.50 (±0.05 tolerance)
- `ci_time_reduction`: 9% to 12% (±1% tolerance)
- `chaos_pass_rate_10pct`: 12-15 passed out of 15 (80%-100%)

### 5.2 Artifact Sizes
```
flake.nix:                11 KB
ci-nix.yml:               10 KB
test_selector.pkl:        254 KB
training_data.json:       55 KB
reports/:                 ~20 KB (generated)
artifacts/:               ~500 KB (flamegraphs)
Total:                    ~850 KB (excluding dependencies)
```

---

## 6. Version Pinning

### 6.1 Nix Packages
- **nixpkgs**: `github:NixOS/nixpkgs/nixos-24.05` (pinned in flake.lock)
- **Python**: 3.12.x (specified in flake.nix line 17)

### 6.2 Python Packages
```
# Core (requirements.txt)
scikit-learn==1.3.2
pandas==2.1.4
joblib==1.3.2
pytest==8.4.2
hypothesis==6.140.3

# Full list: see requirements.lock (121 lines)
```

### 6.3 CI Environment
- **GitHub Actions**: ubuntu-latest (≈ Ubuntu 22.04)
- **Python**: 3.12 (via actions/setup-python@v5)
- **Nix**: via DeterminateSystems/nix-installer-action@v9

---

## 7. Known Limitations & Platform Dependencies

### 7.1 Platform-Specific Behavior
- **macOS**: py-spy requires `sudo` (not applicable in CI)
- **Windows**: Not tested (Linux/macOS only)
- **ARM vs x86**: Nix builds may differ, but should be bit-identical per architecture

### 7.2 Non-Determinism Sources
- **Timestamps**: `created_at` fields vary by run
- **Floating-point**: Minor differences across CPUs (<1e-6)
- **Pytest order**: Use `--seed 42` for deterministic test order
- **Network**: Cloud Storage downloads may fail (use local files as fallback)

### 7.3 Time-Dependent Results
- **Chaos pass rates**: Depend on `--chaos-seed` (fixed seed = reproducible)
- **ML cross-validation**: Depends on random seed (fixed in script)
- **CI time estimates**: Based on 90s baseline (may vary by machine)

---

## 8. Troubleshooting

### 8.1 Common Issues

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`  
**Fix**: `pip install scikit-learn==1.3.2`

**Issue**: `nix: command not found`  
**Fix**: Install Nix or skip C1 hermetic builds test

**Issue**: Chaos tests fail with "No module named 'tests.chaos'"`  
**Fix**: Ensure `tests/chaos/__init__.py` exists (created automatically)

**Issue**: Flamegraph generation fails with "py-spy requires root"`  
**Fix**: Run on Linux (CI) or use `sudo` on macOS

### 8.2 Contact for Support
- **Email**: b@thegoatnote.com
- **GitHub Issues**: https://github.com/GOATnote-Inc/periodicdent42/issues
- **Documentation**: See 13 comprehensive guides in repository root

---

## 9. Artifact Evaluation Badges

Based on ACM artifact evaluation criteria:

### 9.1 Artifacts Available ✅
- ✅ Repository public and accessible
- ✅ License specified (MIT)
- ✅ Documentation comprehensive

### 9.2 Artifacts Functional ✅
- ✅ All scripts execute without errors
- ✅ Dependencies clearly specified
- ✅ Installation instructions complete

### 9.3 Results Reproduced ⚠️
- ✅ C2, C3, C4: Fully reproducible with provided data
- ⏳ C1: Requires Nix (not available on all platforms)
- ⏳ Production validation needed (2 weeks data collection)

### 9.4 Results Replicated ⏳
- ⏳ Independent replication pending (new data collection required)

**Recommended Badge**: **Artifacts Available + Functional**

---

## 10. Maintenance & Long-Term Availability

### 10.1 Repository Maintenance
- **Status**: Actively maintained (last commit: 2025-10-06)
- **Updates**: Weekly dependency updates via Dependabot
- **CI**: All workflows passing

### 10.2 Long-Term Archival
- **GitHub**: Primary location (may change ownership)
- **Zenodo**: Recommended for DOI and permanent archival
- **Docker**: Container images available (reproducibility guarantee)

### 10.3 Contact Longevity
- **Email**: b@thegoatnote.com (organizational, not personal)
- **ORCID**: TBD (recommended for author identification)

---

## Reviewer Checklist

### Initial Review (30 min)
- [ ] Clone repository
- [ ] Install dependencies
- [ ] Run `python scripts/eval_test_selection.py`
- [ ] Verify CV F1 ≈ 0.45 ± 0.16

### Full Review (2-3 hours)
- [ ] Run all replication scripts (C1-C4)
- [ ] Verify metrics within ±10% of reported values
- [ ] Check documentation completeness
- [ ] Test on 2 platforms (Ubuntu + macOS)

### Production Validation (2 weeks)
- [ ] Collect 50+ real test runs
- [ ] Retrain ML model
- [ ] Measure real CI time reduction
- [ ] Validate false negative rate <10%

---

© 2025 GOATnote Autonomous Research Lab Initiative  
Artifact Evaluation Checklist  
Version: 1.0 (2025-10-06)
