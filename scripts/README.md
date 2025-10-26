# FlashCore Scripts

Utility scripts for deployment, benchmarking, and repository management.

## 📂 Directory Structure

```
scripts/
├── benchmarking/      # Performance validation scripts
├── deployment/        # GPU deployment and connection utilities
├── archive_experimental_code.sh
└── close_all_dependabot_prs.sh
```

## 🚀 Deployment

**Location**: `deployment/`

| Script | Purpose |
|--------|---------|
| `deploy_all_kernels.sh` | Deploy all validated kernels to remote GPU |
| `reconnect_h100.sh` | Reconnect to H100 GPU after restart |
| `verify_runpod_startup.sh` | Systematic GPU startup verification |

### Quick Start

```bash
# Connect to H100
./scripts/deployment/reconnect_h100.sh

# Verify GPU is online
./scripts/deployment/verify_runpod_startup.sh

# Deploy kernels
./scripts/deployment/deploy_all_kernels.sh
```

## 📊 Benchmarking

**Location**: `benchmarking/`

| Script | Purpose |
|--------|---------|
| `benchmark_multihead_h100.sh` | Multi-head attention validation (H=8-128) |
| `validate_fp8_h100.py` | FP8 precision testing (blocked) |
| `validate_longcontext_h100.py` | Long-context testing (blocked) |

### Running Benchmarks

```bash
# Multi-head attention (production)
./scripts/benchmarking/benchmark_multihead_h100.sh

# Note: FP8 and long-context are blocked (see docs/validation/)
```

## 🛠️ Repository Management

| Script | Purpose |
|--------|---------|
| `archive_experimental_code.sh` | Archive experimental code |
| `close_all_dependabot_prs.sh` | Manage Dependabot PRs per stability policy |

## 📝 Notes

- All deployment scripts require active RunPod H100 connection
- Benchmarking scripts use `torch.allclose(rtol=1e-3, atol=2e-3)` for correctness validation
- See `docs/validation/` for detailed validation reports

## 🔗 Related Documentation

- [Validation Reports](../docs/validation/)
- [Deployment Guide](../docs/getting-started/)
- [Dependency Stability Policy](../docs/DEPENDENCY_STABILITY_POLICY.md)

