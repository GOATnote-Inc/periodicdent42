# Repository Hardening - Implementation Tracker

**Start Date**: October 8, 2025  
**Audit Source**: PERIODIC_LABS_CRITICAL_AUDIT_OCT2025.md  
**Target**: Production-grade materials discovery platform  
**Quality Standard**: DeepMind/Google internal review criteria

---

## Executive Summary

This document tracks systematic corrections to address critical gaps identified in the Periodic Labs technical audit (2.5/5.0 → target 4.5/5.0). Each task represents an independent PR with measurable acceptance criteria and machine-verifiable evidence.

**Audit Score**: 2.5/5.0 (Below hire threshold)  
**Target Score**: 4.5/5.0 (Strong hire)  
**Timeline**: 4 weeks (8 tasks × 2-5 days each)

---

## Global Quality Gates (Enforced in CI)

| Gate | Threshold | Status |
|------|-----------|--------|
| **Test Coverage** | ≥80% | 🔴 24.3% (12,793 LOC / 3,109 tests) |
| **SOTA Baselines** | ≥3 GNN comparisons | 🔴 0 (no CGCNN/MEGNet/M3GNet) |
| **DFT Integration** | Materials Project API | 🔴 Heuristics only |
| **Uncertainty Calibration** | ECE ≤0.05 | 🔴 Not measured |
| **A-Lab Validation** | ≥1 end-to-end test | 🔴 0 tests |
| **Data Provenance** | SHA-256 verified | 🔴 No checksums |
| **Drift Detection** | Monthly CI job | 🔴 No monitoring |
| **Load Testing** | P95<2s, errors<0.5% | 🔴 Not tested |
| **Reproducibility** | Double-build hash match | 🟢 Nix flakes functional |

---

## Task Graph (Sequential Execution)

### T1: SOTA Baseline Comparisons (CRITICAL - 5 days)
**Branch**: `feat/baselines-sota`  
**Issue**: #001  
**Priority**: CRITICAL  
**Status**: 🔴 Not Started

**Problem**: 22.5% improvement claim uninterpretable without field context. Zero comparisons to graph neural networks (CGCNN, MEGNet, M3GNet).

**Deliverables**:
- [ ] `validation/baselines/cgcnn_runner.py` - CGCNN training script
- [ ] `validation/baselines/megnet_runner.py` - MEGNet training script
- [ ] `validation/baselines/m3gnet_runner.py` - M3GNet training script
- [ ] `validation/baselines/compare_models.py` - Unified comparison
- [ ] `validation/BASELINE_COMPARISON.md` - Results table
- [ ] `.github/workflows/baselines.yml` - CI job (manual trigger)
- [ ] `validation/artifacts/baselines/` - Trained weights + logs

**Acceptance Criteria**:
- ✅ Table with RMSE/MAE/R²/training time/cost for 4 models (RF, CGCNN, MEGNet, M3GNet)
- ✅ Same dataset (UCI 21,263 superconductors), same split (80/10/10), same seed (42)
- ✅ Model weights saved with SHA-256 checksums
- ✅ CI job uploads artifacts (weights, logs, plots)
- ✅ README badge linking to comparison table

**Evidence Required**:
```bash
# Expected output structure
validation/BASELINE_COMPARISON.md:
| Model | Architecture | RMSE (K) | MAE (K) | R² | Time | Cost |
| RF (this work) | Random Forest | 16.72 | 11.30 | 0.728 | 12m | $0 |
| CGCNN | Graph Conv | 12.3±0.4 | 8.7±0.3 | 0.89 | 2h 15m | $2.50 |
| MEGNet | MEGNet | 11.8±0.3 | 8.2±0.2 | 0.91 | 3h 42m | $4.20 |
| M3GNet | M3GNet | 10.9±0.2 | 7.9±0.2 | 0.93 | 5h 10m | $6.80 |
```

**Dependencies**: None  
**Risks**: GNN training may exceed Cloud Run memory (512Mi) → run locally or increase limit  
**Estimated Time**: 5 days (2 days setup, 3 days training/validation)

---

### T2: DFT Features Integration (CRITICAL - 5 days)
**Branch**: `feat/dft-features-mpapi`  
**Issue**: #002  
**Priority**: CRITICAL  
**Status**: 🔴 Not Started

**Problem**: Physics features based on composition heuristics (`lambda_ep = (dos_fermi * 100.0) / (omega_debye ** 2)`), not quantum mechanical calculations. Insufficient for novel materials discovery.

**Deliverables**:
- [ ] `matprov/features/dft_features.py` - Materials Project API integration
- [ ] `matprov/features/dft_stubs.py` - Quantum ESPRESSO/VASP stubs
- [ ] `.env.example` - API key configuration
- [ ] `docs/DFT_FEATURES.md` - Setup and usage guide
- [ ] `validation/DFT_VS_HEURISTIC.ipynb` - 100-material validation notebook
- [ ] `validation/artifacts/dft_validation/` - Scatter plots + error analysis

**Acceptance Criteria**:
- ✅ Materials Project API functional (with fallback to heuristics)
- ✅ Feature provenance labeled: `source="DFT"` or `source="heuristic"`
- ✅ Validation: σ(λ_heuristic - λ_DFT) < 0.3 for 100 materials
- ✅ Scatter plot: heuristic λ vs DFT λ with R² documented
- ✅ QE/VASP stubs with clear TODOs and unit parsers (no runtime dependency)
- ✅ Feature flag: `USE_DFT=true/false` (default: false for backward compatibility)

**Evidence Required**:
```python
# validation/DFT_VS_HEURISTIC.ipynb output
Mean absolute error: σ(λ_heuristic - λ_DFT) = 0.24 ± 0.05
R²: 0.73
Max error: 0.81 (material: mp-1234, formula: La2CuO4)
Scatter plot saved: validation/artifacts/dft_validation/lambda_comparison.png
```

**Dependencies**: T1 (for fair GNN comparison with DFT features)  
**Risks**: Materials Project API rate limits (50 req/sec) → implement caching  
**Estimated Time**: 5 days (2 days API integration, 2 days validation, 1 day stubs)

---

### T3: Uncertainty Calibration (IMPORTANT - 3 days)
**Branch**: `feat/uncertainty-calibration`  
**Issue**: #003  
**Priority**: IMPORTANT  
**Status**: 🔴 Not Started

**Problem**: Active learning requires calibrated uncertainty. No Expected Calibration Error (ECE) or reliability diagrams. Miscalibration → wrong experiments selected → wasted synthesis resources.

**Deliverables**:
- [ ] `matprov/metrics/calibration.py` - ECE, reliability diagram, isotonic/Platt
- [ ] `validation/CALIBRATION_REPORT.md` - Results + plots
- [ ] `.github/workflows/ci.yml` - Add ECE gate (fail if ECE > 0.05)
- [ ] `validation/artifacts/calibration/` - Reliability diagrams

**Acceptance Criteria**:
- ✅ ECE metric computed and logged
- ✅ Reliability diagram (predicted probability vs observed frequency)
- ✅ CI gate: fail if ECE > 0.05 (configurable via env: `MAX_ECE`)
- ✅ Isotonic regression applied if ECE > 0.05
- ✅ Bootstrap predictive intervals for regression (95% CI)

**Evidence Required**:
```bash
# CI output
ECE: 0.038 (threshold: 0.05) ✅ PASS
Reliability diagram: validation/artifacts/calibration/reliability.png
Bootstrap intervals: mean=16.72K, 95% CI=[15.1K, 18.3K]
```

**Dependencies**: T1 (for GNN uncertainty calibration comparison)  
**Risks**: Calibration may degrade accuracy → document tradeoff  
**Estimated Time**: 3 days (1 day implementation, 1 day validation, 1 day CI integration)

---

### T4: Test Coverage ≥80% (IMPORTANT - 5 days)
**Branch**: `chore/tests-increase-coverage`  
**Issue**: #004  
**Priority**: IMPORTANT  
**Status**: 🔴 Not Started

**Problem**: Current coverage 24.3% (3,109 / 12,793 LOC). Physics equations untested. No adversarial tests. Industry standard: >80%.

**Deliverables**:
- [ ] `tests/unit/test_physics_equations.py` - McMillan/Allen-Dynes validation
- [ ] `tests/adversarial/test_invalid_inputs.py` - Malformed CIFs, negative λ
- [ ] `tests/property/test_physics_properties.py` - Hypothesis-based tests
- [ ] `tests/integration/test_alab_schemas.py` - Schema validation
- [ ] `.github/workflows/ci.yml` - Coverage gate (fail if <80%)

**Acceptance Criteria**:
- ✅ Coverage ≥80% (measured by `pytest --cov`)
- ✅ Physics equations validated against experimental data:
  - Nb: λ=1.0, θ_D=275K, μ*=0.1 → Tc=9.2K (tolerance: ±1.0K)
  - MgB₂: λ=0.87, θ_D=900K, μ*=0.1 → Tc=39.0K (tolerance: ±5.0K)
  - Al: λ=0.43, θ_D=428K, μ*=0.1 → Tc=1.2K (tolerance: ±0.5K)
- ✅ Adversarial tests: empty CIF, negative lattice, missing unit cell
- ✅ Property-based tests: Tc monotonic in λ, dimensional analysis
- ✅ Per-module coverage report with badge

**Evidence Required**:
```bash
# pytest output
tests/unit/test_physics_equations.py::test_mcmillan_nb PASSED
tests/unit/test_physics_equations.py::test_mcmillan_mgb2 PASSED
tests/unit/test_physics_equations.py::test_mcmillan_al PASSED
tests/adversarial/test_invalid_inputs.py::test_empty_cif PASSED
tests/property/test_physics_properties.py::test_tc_monotonic PASSED

Coverage: 82.7% (target: ≥80%) ✅ PASS
```

**Dependencies**: None (can run in parallel with T1-T3)  
**Risks**: Coverage may reveal bugs in existing code → fix as discovered  
**Estimated Time**: 5 days (3 days writing tests, 2 days fixing uncovered bugs)

---

### T5: A-Lab End-to-End Validation (IMPORTANT - 4 days)
**Branch**: `feat/alab-e2e-validation`  
**Issue**: #005  
**Priority**: IMPORTANT  
**Status**: 🔴 Not Started

**Problem**: A-Lab integration schemas correct but zero real-world validation. Cannot deploy without proof of Berkeley hardware compatibility.

**Deliverables**:
- [ ] `tests/integration/test_alab_e2e.py` - Mock + live tests
- [ ] `docs/ALAB_CONSTRAINTS.md` - Robotic constraints documentation
- [ ] `matprov/integrations/alab_client.py` - Robust API client (retries, backoff)
- [ ] `validation/artifacts/alab/` - Sample request/response JSONs

**Acceptance Criteria**:
- ✅ Mock test always runs in CI (no external dependencies)
- ✅ Live test gated by `ALAB_LIVE=1` env var (manual workflow_dispatch)
- ✅ Robotic constraints documented:
  - Arm reach: furnace positions ±5cm tolerance
  - Crucible availability: 20 crucibles, 2-hour cleaning cycle
  - Precursor inventory: query `/api/inventory` before selection
  - Furnace schedule: 4 furnaces, max 8 parallel syntheses
  - Safety: no volatile/toxic precursors (HF, CS₂, etc.)
- ✅ Error handling: API failures, synthesis failures, XRD corruption
- ✅ Sample artifacts: request/response saved to `validation/artifacts/alab/`

**Evidence Required**:
```bash
# Mock test output
tests/integration/test_alab_e2e.py::test_conversion PASSED
tests/integration/test_alab_e2e.py::test_mock_submission PASSED
tests/integration/test_alab_e2e.py::test_result_ingestion PASSED

# Live test output (manual)
ALAB_LIVE=1 pytest tests/integration/test_alab_e2e.py::test_live_submission
Job ID: alab-2025-10-08-001
Status: queued → running → complete (48h 23m)
Result: success, phase_purity=0.87
Artifact: validation/artifacts/alab/alab-2025-10-08-001-response.json
```

**Dependencies**: None  
**Risks**: Berkeley A-Lab access requires approval → mock test sufficient for initial validation  
**Estimated Time**: 4 days (2 days mock tests, 1 day live test, 1 day docs)

---

### T6: Data Provenance Ledger (IMPORTANT - 3 days)
**Branch**: `feat/provenance-ledger`  
**Issue**: #006  
**Priority**: IMPORTANT  
**Status**: 🔴 Not Started

**Problem**: No cryptographic data lineage. Scientific reproducibility requires SHA-256 checksums and git SHAs for datasets, models, and runs.

**Deliverables**:
- [ ] `data/PROVENANCE.json` - Ledger (datasets, models, runs)
- [ ] `scripts/compute_data_checksums.py` - Checksum computation
- [ ] `scripts/verify_provenance.py` - Verification script
- [ ] `.github/workflows/provenance.yml` - CI job (fail on mismatch)

**Acceptance Criteria**:
- ✅ `data/PROVENANCE.json` with SHA-256 for all datasets and models
- ✅ Verification script validates checksums (returns 0 on success, 1 on mismatch)
- ✅ CI job fails if checksums don't match
- ✅ Ledger includes:
  - Dataset: name, URL, download_date, sha256, size_bytes, num_samples
  - Model: name, training_date, sha256, training_data, hyperparameters
  - Experiment: run_id, timestamp, model, dataset, results, git_commit

**Evidence Required**:
```bash
# scripts/verify_provenance.py output
✅ Dataset 'UCI_Superconductor_21263' checksum verified
✅ Model 'test_selector_v2.pkl' checksum verified
✅ Experiment '2025-10-06_validation_run_001' git commit verified
All checksums valid. Provenance verified.
```

**Dependencies**: T1 (for baseline model checksums)  
**Risks**: Large files may slow CI → use Git LFS for weights  
**Estimated Time**: 3 days (1 day ledger, 1 day scripts, 1 day CI)

---

### T7: Drift Detection (NICE TO HAVE - 3 days)
**Branch**: `feat/drift-detection`  
**Issue**: #007  
**Priority**: NICE TO HAVE  
**Status**: 🔴 Not Started

**Problem**: Production models degrade over time. No automatic detection of data/concept drift.

**Deliverables**:
- [ ] `monitoring/drift_detection.py` - KS test + PSI
- [ ] `.github/workflows/drift-detection.yml` - Monthly cron job
- [ ] `validation/artifacts/drift/` - Reports

**Acceptance Criteria**:
- ✅ KS test (Kolmogorov-Smirnov) for each feature
- ✅ PSI (Population Stability Index) for distribution shift
- ✅ Alert if ≥3 features show drift (p < 0.05)
- ✅ Monthly CI job scheduled (cron: `0 0 1 * *`)
- ✅ Report artifact uploaded

**Evidence Required**:
```bash
# monitoring/drift_detection.py output
Feature 'dos_fermi': KS statistic=0.08, p-value=0.023 ⚠️ DRIFT
Feature 'debye_temp': KS statistic=0.12, p-value=0.001 ⚠️ DRIFT
Feature 'lambda_ep': KS statistic=0.15, p-value=0.0003 ⚠️ DRIFT
ALERT: 3 features drifted (threshold: 3)
Report: validation/artifacts/drift/2025-11-01-drift-report.json
```

**Dependencies**: T6 (for reference data checksums)  
**Risks**: False positives on seasonal data → document baseline assumptions  
**Estimated Time**: 3 days (1 day implementation, 1 day validation, 1 day CI)

---

### T8: Load Testing (NICE TO HAVE - 2 days)
**Branch**: `feat/load-testing-locust`  
**Issue**: #008  
**Priority**: NICE TO HAVE  
**Status**: 🔴 Not Started

**Problem**: Cloud Run deployment (512Mi RAM) untested under load. Need to verify P95 latency and error rate.

**Deliverables**:
- [ ] `tests/load/locustfile.py` - Locust load test
- [ ] `docs/LOAD_TESTING.md` - Setup and thresholds
- [ ] `.github/workflows/load-test.yml` - CI job (manual trigger)

**Acceptance Criteria**:
- ✅ Locust scenario: 100 users, 10 req/sec, 5 minutes
- ✅ Thresholds:
  - P95 latency < 2 seconds
  - Error rate < 0.5%
  - Memory usage < 450Mi (10% safety margin)
- ✅ Report artifact with metrics
- ✅ CI job (manual workflow_dispatch, runs against staging)

**Evidence Required**:
```bash
# Locust output
Users: 100, RPS: 10.2
P50 latency: 0.8s ✅
P95 latency: 1.7s ✅ (threshold: 2s)
P99 latency: 2.3s ⚠️
Error rate: 0.3% ✅ (threshold: 0.5%)
Memory peak: 428Mi ✅ (threshold: 450Mi)
```

**Dependencies**: T1, T2 (for realistic inference load)  
**Risks**: Load test may trigger Cloud Run scaling costs → use staging environment  
**Estimated Time**: 2 days (1 day Locust setup, 1 day CI integration)

---

## Progress Tracking

### Overall Status
- **Tasks Completed**: 0/8 (0%)
- **Quality Gates Passed**: 1/9 (11%) - only reproducibility
- **Estimated Total Time**: 30 days (4 weeks)
- **Critical Path**: T1 → T2 → T3 → T4 (parallel) → T5 → T6 → T7 → T8

### Weekly Milestones
- **Week 1**: T1 (SOTA Baselines) + T2 (DFT Features)
- **Week 2**: T3 (Uncertainty Calibration) + T4 (Test Coverage)
- **Week 3**: T5 (A-Lab Validation) + T6 (Provenance Ledger)
- **Week 4**: T7 (Drift Detection) + T8 (Load Testing) + Final CI/CD integration

---

## Evidence Artifacts (Target Structure)

```
validation/
├── artifacts/
│   ├── baselines/
│   │   ├── cgcnn_weights.pt (sha256: ...)
│   │   ├── megnet_weights.h5 (sha256: ...)
│   │   ├── m3gnet_weights.pt (sha256: ...)
│   │   ├── training_logs.txt
│   │   └── comparison_plot.png
│   ├── dft_validation/
│   │   ├── lambda_comparison.png
│   │   ├── error_distribution.png
│   │   └── validation_results.json
│   ├── calibration/
│   │   ├── reliability_diagram.png
│   │   ├── ece_history.json
│   │   └── calibration_curves.png
│   ├── alab/
│   │   ├── sample_request.json
│   │   ├── sample_response.json
│   │   └── mock_test_logs.txt
│   ├── drift/
│   │   └── 2025-11-01-drift-report.json
│   └── load/
│       └── locust_report_2025-10-08.html
├── BASELINE_COMPARISON.md
├── CALIBRATION_REPORT.md
├── DFT_VS_HEURISTIC.ipynb
└── README.md
```

---

## CI/CD Architecture (Post-Hardening)

```yaml
# .github/workflows/ci.yml (updated)
name: Comprehensive CI/CD
on: [push, pull_request]

jobs:
  test-and-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Verify provenance
        run: python scripts/verify_provenance.py
      - name: Run tests with coverage
        run: pytest --cov --cov-report=xml --cov-fail-under=80
      - name: Check calibration
        run: python -m matprov.metrics.calibration --max-ece 0.05
      - name: Upload coverage
        uses: codecov/codecov-action@v4
  
  reproducibility:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build twice
        run: |
          nix build '.#default' -o result1
          nix build '.#default' -o result2
          diff -r result1 result2 || exit 1

  baselines:
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - name: Run SOTA baselines
        run: python validation/baselines/compare_models.py
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: baseline-results
          path: validation/artifacts/baselines/

  drift-detection:
    if: github.event.schedule == '0 0 1 * *'  # Monthly
    runs-on: ubuntu-latest
    steps:
      - name: Check drift
        run: python monitoring/drift_detection.py
```

---

## Definition of Done (Global)

### Required for All Tasks
- [ ] Code changes minimal and isolated
- [ ] Tests pass (≥80% coverage for affected modules)
- [ ] Style/type checks pass (`ruff`, `mypy`, `pydocstyle`)
- [ ] Documentation updated (`docs/`, `validation/README.md`)
- [ ] Artifacts uploaded to CI
- [ ] PR description includes Why/What/How/Risk/Evidence
- [ ] Conventional commit format
- [ ] CHANGELOG.md entry added

### Required for Entire Project
- [ ] Coverage ≥80% on default branch
- [ ] Baseline comparison table published
- [ ] DFT features integrated with provenance
- [ ] ECE ≤0.05 (or calibrated with plots)
- [ ] A-Lab mock test passing
- [ ] Provenance ledger verified in CI
- [ ] Monthly drift job scheduled
- [ ] Load test thresholds met
- [ ] Reproducible double-build hashes equal
- [ ] README badges updated (coverage, baselines, provenance)

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GNN training exceeds 512Mi RAM | High | High | Run locally or increase Cloud Run limit |
| Materials Project API rate limits | Medium | Medium | Implement caching + exponential backoff |
| Berkeley A-Lab access denied | Low | Medium | Mock tests sufficient for initial validation |
| ECE > 0.05 after calibration | Medium | Low | Document accuracy tradeoff, use isotonic regression |
| Load test triggers scaling costs | Low | High | Use staging environment, cap max instances |
| Provenance checksums fail on CI | High | Low | Use Git LFS for large files, test locally first |

---

## References

- **Audit Source**: `PERIODIC_LABS_CRITICAL_AUDIT_OCT2025.md`
- **Quality Standard**: DeepMind internal review + Nature peer review
- **Test Coverage Standard**: Google (80-90%), DeepMind (85%)
- **Calibration Standard**: Industry (ECE < 0.05)
- **SOTA Baselines**: CGCNN (2018), MEGNet (2019), M3GNet (2022)

---

**Last Updated**: October 8, 2025  
**Next Review**: After T1 completion (October 13, 2025)

