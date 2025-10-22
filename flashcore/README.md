# FlashCore: High-Performance Fused Attention Kernels

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.2+-blue)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-green)]()

**Goal**: Achieve ≥15× speedup over baseline PyTorch attention on NVIDIA L4 GPUs  
**Status**: v0.1 - Phase 0 (Baseline Validation) 🚀  
**Last Updated**: October 21, 2025

---

## 🎯 Project Overview

FlashCore is an open-source repository demonstrating systematic GPU kernel optimization through:
- **Fused attention kernels** with FlashAttention-style tiling
- **Tensor Core utilization** (NVIDIA WMMA)
- **Evolutionary optimization** (EvoEngineer methodology)
- **Reproducible research** infrastructure (tests, benchmarks, profiling)

### Standing on Giants' Shoulders

FlashCore builds upon:
- **FlashAttention** & **FlashAttention-2** (tiling algorithms)
- **EvoEngineer** (systematic optimization framework)
- **periodicdent42** (proven infrastructure)
- **robust-kbench** (rigorous evaluation methodology)

---

## 🚀 Quick Start

### Prerequisites

```bash
# NVIDIA GPU with CUDA 12.2+
nvidia-smi

# Python 3.10+ with PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

### Installation

```bash
# Clone repository
cd flashcore/

# Build baseline kernel
python build.py

# Run tests (15 test cases)
pytest tests/test_correctness.py -v

# Benchmark performance
python benchmarks/benchmark_latency.py --shape mission --iters 100
```

---

## 📊 Performance Results

### Baseline (v0.1)

**Mission Shape**: `B=1, H=8, S=512, D=64` on NVIDIA L4

| Metric | Value | Notes |
|--------|-------|-------|
| **Latency (p50)** | ~1500 µs | Baseline (scalar, no WMMA) |
| **vs PyTorch SDPA** | 0.017× | 58× slower (starting point) |
| **vs Target (<58 µs)** | 26× away | Need 26× speedup |
| **Correctness** | ✅ 15/15 PASS | max_err=0.052 |

### Roadmap (Planned Performance)

| Version | Phase | Optimization | Target Latency | vs Baseline | Status |
|---------|-------|--------------|----------------|-------------|--------|
| v0.1 | Phase 0 | Baseline (scalar) | ~1500 µs | 1.0× | ✅ DONE |
| v0.2 | Phase 1 | WMMA Tensor Cores | ~150 µs | 10× | 🔄 Next |
| v0.3 | Phase 2 | FlashAttention Fusion | <58 µs | 26× | ⏳ Planned |
| v0.4 | Phase 3 | Warp Specialization | ~20 µs | 75× | ⏳ Stretch |
| v1.0 | Phase 4 | Evolutionary Search | ~15 µs | 100× | 🚀 Bonus |

**Primary Goal**: v0.3 (<58 µs, ≥15× vs 870 µs old PyTorch) → **Phase 2**

---

## 🏗️ Architecture

### Repository Structure

```
flashcore/
├── kernels/                    # CUDA kernel implementations
│   ├── flashcore_baseline.cu  # v0.1: Scalar baseline
│   ├── bindings.cpp            # PyTorch C++ bindings
│   └── (future: wmma, fused, optimized kernels)
│
├── tests/                      # Correctness validation
│   └── test_correctness.py    # 15 test cases (5 shapes × 3 seeds)
│
├── benchmarks/                 # Performance measurement
│   └── benchmark_latency.py   # 100-run medians, PyTorch comparison
│
├── profiling/                  # Hardware analysis
│   └── (future: NCU scripts, roofline analysis)
│
├── search/                     # Evolutionary optimization
│   └── (future: autotune, LLM-driven search)
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # Technical design
│   ├── BASELINE_REPORT.md      # Phase 0 results
│   └── (future: phase reports, tutorials)
│
└── scripts/                    # Automation
    └── (future: validation pipelines)
```

### Design Principles

1. **Modular**: Each optimization phase = separate kernel file
2. **Reproducible**: JSON artifacts with git SHA, environment info
3. **Community-First**: Apache 2.0, educational comments
4. **No Cheating**: Multi-case tests prevent overfitting
5. **Systematic**: EvoEngineer methodology (elite preservation, config search)

---

## 🧪 Testing & Validation

### Correctness Tests

**Multi-Shape Coverage** (prevent overfitting):

| Shape | Config | Seeds | Purpose |
|-------|--------|-------|---------|
| tiny | B=1, H=1, S=32 | 0, 42, 12345 | Quick sanity |
| small | B=1, H=2, S=64 | 0, 42, 12345 | Intermediate |
| medium | B=1, H=4, S=128 | 0, 42, 12345 | Scaling test |
| **mission** | **B=1, H=8, S=512** | **0, 42, 12345** | **Primary target** |
| multi_batch | B=4, H=8, S=256 | 0, 42, 12345 | Batching |

**Accuracy Thresholds** (FP16):
- Max error ≤ 0.06
- Mean error ≤ 0.02
- No NaN/Inf values

### Benchmarking Methodology

**Robust Statistics** (aligned with robust-kbench):
- 100 iterations (median, p90, p99)
- 20 warmup iterations
- CUDA event timing
- PyTorch SDPA baseline comparison

---

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Technical design, algorithm details
- **[BASELINE_REPORT.md](docs/BASELINE_REPORT.md)**: Phase 0 results
- **[Implementation Plan](../FLASHCORE_IMPLEMENTATION_PLAN.md)**: Detailed roadmap
- **[Launch Plan](../FLASHCORE_LAUNCH_PLAN.md)**: Project overview

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) (coming soon) for guidelines.

**Areas for Contribution**:
- Additional kernel optimizations (Phase 1-4)
- Backward pass implementations (dQ, dK, dV)
- Support for other GPUs (A100, RTX series)
- Documentation improvements
- Tutorial notebooks

---

## 📄 License

Apache License 2.0

Copyright 2025 FlashCore Contributors

See [LICENSE](../LICENSE) for full text.

---

## 🎓 Citation

If you use FlashCore in your research, please cite:

```bibtex
@software{flashcore2025,
  title = {FlashCore: High-Performance Fused Attention Kernels},
  author = {FlashCore Contributors},
  year = {2025},
  url = {https://github.com/yourusername/flashcore},
  note = {Open-source CUDA kernel optimization framework}
}
```

---

## 🔗 Related Projects

- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)**: Original fused attention algorithm
- **[EvoEngineer](https://arxiv.org/abs/2510.03760)**: LLM-driven kernel optimization
- **[robust-kbench](https://github.com/.../)**: Rigorous kernel benchmarking
- **[periodicdent42](https://github.com/yourusername/periodicdent42)**: Parent project

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/flashcore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/flashcore/discussions)
- **Email**: your.email@example.com

---

## 🏆 Acknowledgments

FlashCore stands on the shoulders of giants:
- Tri Dao et al. (FlashAttention)
- Guo et al. (EvoEngineer)
- PyTorch Team (SDPA implementation)
- NVIDIA (CUDA, WMMA, profiling tools)
- periodicdent42 contributors

---

**Status**: Phase 0 Complete ✅ → Phase 1 In Progress 🔄  
**Next Milestone**: v0.2 (WMMA Tensor Cores) → Target: ~150 µs

**Let's optimize! 🚀**

