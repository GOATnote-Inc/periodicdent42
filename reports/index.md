# Repository Evidence Index

**Generated**: 2025-10-06  
**Audit Purpose**: Validate 4 claims (C1-C4) with verifiable evidence

---

## Evidence Locations by Area

| Area | Key Files | Last Updated | Status | Notes |
|------|-----------|--------------|--------|-------|
| **C1: Hermetic Builds** | | | | |
| Nix Configuration | `flake.nix` (323 lines) | 2025-10-06 | ✓ Present | 3 devshells (default, full, ci) |
| Nix CI | `.github/workflows/ci-nix.yml` (253 lines) | 2025-10-06 | ✓ Present | Multi-platform (Ubuntu + macOS) |
| Nix Documentation | `NIX_SETUP_GUIDE.md` (500+ lines) | 2025-10-06 | ✓ Present | Comprehensive setup guide |
| Cache Strategy | `NIX_CACHE_STRATEGY.md` (900+ lines) | 2025-10-06 | ✓ Present | Multi-layer caching |
| SLSA Setup | `SLSA_SETUP_GUIDE.md` (800+ lines) | 2025-10-06 | ✓ Present | Level 3+ attestation |
| SLSA Verification | `scripts/verify_slsa.sh` (150 lines) | 2025-10-06 | ✓ Present | Automated verification |
| **C2: ML Test Selection** | | | | |
| Model Binary | `test_selector.pkl` (254 KB) | 2025-10-06 | ✓ Present | RandomForestClassifier |
| Model Metadata | `test_selector.json` (292 bytes) | 2025-10-06 | ✓ Present | 7 features, v1.0.0 |
| Training Data | `training_data.json` (55 KB, N=100) | 2025-10-06 | ✓ Present | Synthetic baseline |
| Training Script | `scripts/train_test_selector.py` (400 lines) | 2025-10-06 | ✓ Present | CV, evaluation, export |
| Prediction Script | `scripts/predict_tests.py` (350 lines) | 2025-10-06 | ✓ Present | Git diff → test selection |
| Telemetry Schema | `app/alembic/versions/001_add_test_telemetry.py` | 2025-10-06 | ✓ Present | 7 ML features |
| Telemetry Collector | `app/src/services/test_telemetry.py` (450 lines) | 2025-10-06 | ✓ Present | Pytest plugin |
| CI Integration | `.github/workflows/ci.yml` (ml-test-selection job) | 2025-10-06 | ✓ Present | GCS download enabled |
| Documentation | `ML_TEST_SELECTION_GUIDE.md` (1,000 lines) | 2025-10-06 | ✓ Present | Complete system guide |
| **C3: Chaos Engineering** | | | | |
| Pytest Plugin | `tests/chaos/conftest.py` (225 lines) | 2025-10-06 | ✓ Present | 5 failure types |
| Resilience Patterns | `tests/chaos/resilience_patterns.py` (180 lines) | 2025-10-06 | ✓ Present | 5 patterns (retry, circuit breaker, etc) |
| Test Examples | `tests/chaos/test_chaos_examples.py` (230 lines) | 2025-10-06 | ✓ Present | 15 validated tests |
| CI Integration | `.github/workflows/ci.yml` (chaos job) | 2025-10-06 | ✓ Present | Scheduled + on-demand |
| Documentation | `CHAOS_ENGINEERING_GUIDE.md` (700 lines) | 2025-10-06 | ✓ Present | Complete framework guide |
| **C4: Continuous Profiling** | | | | |
| Profiling Script | `scripts/profile_validation.py` (150 lines) | 2025-10-06 | ✓ Present | py-spy integration |
| Bottleneck Analysis | `scripts/identify_bottlenecks.py` (400 lines) | 2025-10-06 | ✓ Present | SVG parsing, AI recommendations |
| Regression Detection | `scripts/check_regression.py` (350 lines) | 2025-10-06 | ✓ Present | Recursive JSON comparison |
| Performance Analysis | `scripts/analyze_performance.sh` (100 lines) | 2025-10-06 | ✓ Present | Automated workflow |
| CI Integration | `.github/workflows/ci.yml` (performance-profiling job) | 2025-10-06 | ✓ Present | Flamegraph generation |
| Artifacts | `artifacts/performance_analysis/*.svg` | 2025-10-06 | ✓ Present | 2 flamegraphs |
| Documentation | `CONTINUOUS_PROFILING_COMPLETE.md` (400 lines) | 2025-10-06 | ✓ Present | Setup + analysis guide |
| Documentation | `REGRESSION_DETECTION_COMPLETE.md` (700 lines) | 2025-10-06 | ✓ Present | Detection methodology |

---

## Missing or Insufficient Evidence

| Claim | Gap | Smallest Experiment to Close |
|-------|-----|------------------------------|
| C1 | No observed bit-identical rebuild data | Run `nix build` twice on same platform, compare hashes (5 min) |
| C1 | No cross-platform hash comparison | Build on Linux + macOS in CI, save hashes, compare (included in CI) |
| C1 | No SBOM generation time measurement | Time `nix path-info` in CI, report P50/P95 (5 min) |
| C2 | Model trained on synthetic data | Collect 50+ real test runs, retrain, measure F1 on real data (overnight) |
| C2 | No CI time savings measured yet | Wait for first 10 CI runs with ML enabled, measure time vs baseline (1 day) |
| C2 | No false negative tracking | Add failure tracking to CI, monitor for 2 weeks (2 weeks) |
| C3 | No production incident mapping | Collect 3 months of incident logs, map to chaos failure types (1 hour) |
| C3 | No SLO impact measurement | Define SLO (e.g., P99 latency), measure with/without chaos (1 day) |
| C4 | Only 2 flamegraphs generated | Run profiling on 10+ CI runs, generate trend data (1 week) |
| C4 | No regression detection demonstrated | Introduce synthetic regression, verify detection (1 hour) |
| C4 | No manual vs AI timing comparison | Time manual analysis (N=5 flamegraphs), compare to AI (30 min) |

---

## Data Windows

| Metric | Window Start | Window End | N | Notes |
|--------|-------------|------------|---|-------|
| Nix CI Builds | 2025-10-06 | 2025-10-06 | ~18 | Since Nix CI deployment |
| ML Training Data | 2025-10-02 | 2025-10-06 | 100 | Synthetic data baseline |
| Chaos Tests | 2025-10-06 | 2025-10-06 | 15 | Initial validation suite |
| Flamegraphs | 2025-10-06 | 2025-10-06 | 2 | Performance analysis |

---

## Reproducibility Notes

- All evidence files are timestamped
- Scripts include version information
- CI runs have unique identifiers
- Model training includes random seed
- Chaos tests use `--chaos-seed` for reproducibility

© 2025 GOATnote Autonomous Research Lab Initiative  
Evidence Index Generated: 2025-10-06
