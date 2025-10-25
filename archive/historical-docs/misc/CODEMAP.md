# Autonomous R&D Intelligence Layer - Code Map

**Purpose**: Production AI platform for optimizing physical R&D experiments using dual Gemini models (Flash/Pro), RL agents (PPO+ICM), and Bayesian Optimization.

---

## 🎯 Entry Points

### Runtime
- **API**: `app/src/api/main.py` - FastAPI backend serving AI reasoning endpoints
- **Web UI**: `app/static/` - Static HTML/CSS/JS (Tailwind) for querying AI
- **Analytics**: `app/static/analytics.html` - Dashboard for validation results

### CUDA Kernels & Benchmarks
- **Main Workspace**: `cudadent42/bench/` - Kernel dev, benchmarks, build scripts
- **Kernels**: `cudadent42/bench/kernels/` - FlashAttention variants (v2, v3, WMMA, TC)
- **Key Kernel**: `cudadent42/bench/kernels/fa_s512_v3.cu` (805 lines, see region markers)

### Scripts
- **Benchmarking**: `scripts/bench_*.py`, `scripts/profile_*.py`
- **CI Gates**: `scripts/ci_gates.py`, `scripts/ci/`
- **Database**: `scripts/init_database.py`, `scripts/generate_test_data.py`
- **Validation**: `scripts/validate_*.py`

---

## 🛠 Developer Flows

```bash
# Setup (first time)
make setup              # Install all deps (Python, CUDA build tools)

# Development loop
make lint               # Ruff (Python)
make typecheck          # Mypy
make test               # Pytest (< 60s, CPU-only by default)

# CUDA/GPU workflows (requires L4 GPU)
make bench              # Quick benchmark (S=512, D=64) → artifacts/bench/latest.json
make bench-correctness  # Validate kernels vs PyTorch SDPA
make profile            # Nsight Compute → artifacts/profile/

# Format
make format             # Ruff + Black
```

---

## 🚫 No-Edit Zones

| Directory | Rule | Reason |
|-----------|------|--------|
| `docs/` | Read-only (see `.cursor/rules/no_edit.md`) | Session logs, historical reports |
| `infra/` | Read-only (see `.cursor/rules/no_edit.md`) | Production infrastructure |
| `ext/flash-attention-2/` | Ignore (submodule) | External code, not editable |
| `third_party/` | Ignore | Vendored dependencies |
| `artifacts/**`, `data/**` | Ignore (`.cursorignore`) | Generated outputs, large files |

---

## 📐 Supported Kernel Configurations

### FlashAttention S=512 (fa_s512_v3.cu)
- **Shapes**: B=1-4, H=8-16, S=512, D=64
- **Tile Configs**: `(BLOCK_M, BLOCK_N, NUM_WARPS, STAGES)`
  - Default: `(64, 64, 4, 2)` - Balanced
  - Large Tile: `(128, 64, 8, 2)` - Higher throughput (needs SMEM opt)
- **Build**: `cd cudadent42/bench && python build_v3_release.py`
- **Baseline**: PyTorch SDPA @ 47.10 μs (L4, B=2, H=8, S=512, D=64)

### Target: Beat PyTorch SDPA (Phase 3)
- **Current**: fa_s512.cu @ ~321 μs (6.8× slower)
- **Goal**: < 47 μs (EvoEngineer optimization in progress)

---

## 📂 Architecture

```
periodicdent42/
├── app/                    # FastAPI backend + static UI
│   ├── src/api/            # API endpoints
│   ├── src/reasoning/      # RL agents, Bayesian opt
│   ├── src/services/       # Vertex AI, Cloud SQL, telemetry
│   └── static/             # Web UI
├── cudadent42/             # CUDA kernel workspace
│   └── bench/              # Benchmarks, kernels, build scripts
├── scripts/                # Dev/bench/profile helpers
├── tests/                  # Unit/integration/chaos tests
├── docs/                   # 🚫 Read-only (session logs)
├── infra/                  # 🚫 Read-only (production infra)
├── ext/                    # External submodules (ignore)
└── CODEMAP.md              # ← You are here
```

---

## 🔗 Quick Links
- [README.md](README.md) - Project overview, setup instructions
- [KERNEL_FOCUS.md](KERNEL_FOCUS.md) - 🎯 **START HERE** for kernel optimization work
- [CONTRIBUTING.md](CONTRIBUTING.md) - Dev guidelines
- [Session Logs](docs/archive/session_logs/) - Historical optimization attempts & lessons learned
- [CI/CD](.github/workflows/) - Automated testing & deployment

---

**Last Updated**: 2025-10-16  
**Maintainer**: GOATnote Autonomous Research Lab (b@thegoatnote.com)

