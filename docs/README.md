# FlashCore Documentation

Comprehensive documentation for the FlashCore sub-5μs attention kernel project.

## 📂 Directory Structure

```
docs/
├── validation/           # Kernel validation reports and results
├── reports/              # Project status and progress reports
├── meta/                 # Attributions, citations, and metadata
├── getting-started/      # Installation and quick start guides
├── research/             # Research notes and papers
├── api/                  # API documentation
├── guides/               # Developer guides
└── archive/              # Historical documentation
```

## 🏆 Validation & Results

**Location**: `validation/`

### Production-Ready Kernels

| Report | Description |
|--------|-------------|
| [EXPERT_VALIDATION_REPORT.md](validation/EXPERT_VALIDATION_REPORT.md) | H100 validation (1000 trials, all configs < 5μs) |
| [CROSS_GPU_VALIDATION_REPORT.md](validation/CROSS_GPU_VALIDATION_REPORT.md) | L4 cross-validation (18K measurements) |
| [CORRECTNESS_VALIDATION_COMPLETE.txt](validation/CORRECTNESS_VALIDATION_COMPLETE.txt) | Multi-head attention validation (H=8-128) |
| [H100_VALIDATION_SUMMARY.txt](validation/H100_VALIDATION_SUMMARY.txt) | Multi-head attention H100 results |

### Blocked Kernels (Quality > Velocity)

| Report | Status | Reason |
|--------|--------|--------|
| [FP8_VALIDATION_BLOCKED.txt](validation/FP8_VALIDATION_BLOCKED.txt) | ❌ Blocked | 29% error without scaling factors |
| [LONGCONTEXT_VALIDATION_BLOCKED.txt](validation/LONGCONTEXT_VALIDATION_BLOCKED.txt) | ❌ Blocked | 100× slower than target |

### Session Summaries

| Report | Description |
|--------|-------------|
| [SESSION_OCT25_VALIDATION_SUMMARY.txt](validation/SESSION_OCT25_VALIDATION_SUMMARY.txt) | Expert discipline, systematic validation |
| [SECURITY_AUDIT_REPORT.md](validation/SECURITY_AUDIT_REPORT.md) | Security vulnerabilities audit |
| [SECURITY_FIXES_VERIFICATION.md](validation/SECURITY_FIXES_VERIFICATION.md) | Speed verification post-security fixes |

## 📊 Project Reports

**Location**: `reports/`

| Report | Description |
|--------|-------------|
| [CLEANUP_COMPLETE_SUMMARY.txt](reports/CLEANUP_COMPLETE_SUMMARY.txt) | Repository cleanup and organization |
| [EXPERT_ACTIONS_SUMMARY.txt](reports/EXPERT_ACTIONS_SUMMARY.txt) | Dependency stability policy actions |
| [KERNELS_SHIPPED_TODAY.txt](reports/KERNELS_SHIPPED_TODAY.txt) | Production kernel deployment log |
| [ARCHIVE_OLD_FILES_ANALYSIS.md](reports/ARCHIVE_OLD_FILES_ANALYSIS.md) | File archival analysis |

## 📚 Key Documents

### Strategic

- [STRATEGIC_ROADMAP.md](STRATEGIC_ROADMAP.md) - Development phases for NVIDIA/OpenAI benefit
- [EVIDENCE_PACKAGE.md](EVIDENCE_PACKAGE.md) - Comprehensive performance evidence
- [EXCELLENCE_CONFIRMED.md](EXCELLENCE_CONFIRMED.md) - Executive summary of achievements
- [EXPERT_CONFIRMATION.md](EXPERT_CONFIRMATION.md) - Expert review and A+ certification

### Policy & Process

- [DEPENDENCY_STABILITY_POLICY.md](DEPENDENCY_STABILITY_POLICY.md) - Dependency management policy
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines

## 🎓 Attribution & Citations

**Location**: `meta/`

| Document | Description |
|----------|-------------|
| [ATTRIBUTIONS.md](meta/ATTRIBUTIONS.md) | Comprehensive contributor acknowledgments |
| [CITATIONS.bib](meta/CITATIONS.bib) | Academic citations (BibTeX format) |

### Key Contributors

This project stands on the shoulders of giants:
- **PyTorch** (Meta AI) - Deep learning framework
- **Triton** (OpenAI) - GPU programming language
- **FlashAttention** (Dao et al., Stanford) - Efficient attention algorithm
- **EvoEngineer** (Guo et al., City Univ. Hong Kong) - Optimization methodology
- **NVIDIA** - CUDA Toolkit, Nsight Compute, H100 architecture

See [ATTRIBUTIONS.md](meta/ATTRIBUTIONS.md) for exhaustive list.

## 🚀 Getting Started

**Location**: `getting-started/`

- [Quick Start Guide](getting-started/README.md)
- [Installation Instructions](../README.md#-quick-start)
- [Example Usage](../examples/)

## 🔬 Research

**Location**: `research/`

Research notes, paper drafts, and technical analyses.

## 📖 API Documentation

**Location**: `api/`

API reference and usage documentation.

## 🛠️ Developer Guides

**Location**: `guides/`

Developer guides for contributing to FlashCore.

## 📦 Archive

**Location**: `archive/`

Historical documentation, session logs, and deprecated guides.

---

## 🎯 Navigation by Use Case

### I want to...

**Understand the achievement:**
- Start with [../README.md](../README.md)
- Read [EXCELLENCE_CONFIRMED.md](EXCELLENCE_CONFIRMED.md)

**Reproduce the results:**
- [validation/EXPERT_VALIDATION_REPORT.md](validation/EXPERT_VALIDATION_REPORT.md)
- [validation/CROSS_GPU_VALIDATION_REPORT.md](validation/CROSS_GPU_VALIDATION_REPORT.md)
- [../examples/quick_start.py](../examples/quick_start.py)

**Contribute to the project:**
- [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [DEPENDENCY_STABILITY_POLICY.md](DEPENDENCY_STABILITY_POLICY.md)

**Cite this work:**
- [meta/CITATIONS.bib](meta/CITATIONS.bib)
- [meta/ATTRIBUTIONS.md](meta/ATTRIBUTIONS.md)

**Deploy to production:**
- [validation/SECURITY_AUDIT_REPORT.md](validation/SECURITY_AUDIT_REPORT.md)
- [../scripts/deployment/](../scripts/deployment/)

---

## 📝 Documentation Standards

All documentation follows:
- **Honesty about limitations** - blocked kernels documented with root cause
- **Reproducibility** - 18K measurements, multiple GPUs, public scripts
- **Expert methodology** - systematic validation, proper test criteria
- **Open source** - Apache 2.0 license, comprehensive attribution

**Principle**: "Deeds not words" - all claims backed by validated measurements.

---

Built with ❤️ by GOATnote Inc. | Standing on the shoulders of PyTorch, Triton, FlashAttention, and the entire CUDA ecosystem.

