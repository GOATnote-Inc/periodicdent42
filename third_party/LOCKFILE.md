# Third-Party Tool Lockfile

## Purpose
Pin exact versions of optimization tools for reproducible builds.

## Tools

### EvoEngineer
- **Version**: 0.1.0
- **Commit**: initial (internal implementation)
- **Date**: 2025-10-14
- **Location**: `third_party/evoengineer/`
- **Description**: Evolutionary CUDA kernel parameter optimization framework
- **Components**:
  - `optimizer.py`: Parameter search space and candidate management
  - `evaluator.py`: Benchmark execution and correctness validation
  - `mutator.py`: Parameter mutation strategies

### robust-kbench
- **Version**: 0.1.0
- **Commit**: initial (internal implementation)
- **Date**: 2025-10-14
- **Location**: `third_party/robust_kbench/`
- **Description**: Statistically-rigorous CUDA kernel micro-benchmarking
- **Components**:
  - `config.py`: Benchmark configuration and shape grids
  - `runner.py`: Benchmark execution with statistical rigor
  - `reporter.py`: Multi-format report generation (JSON/CSV/Markdown)

## Verification

```bash
# Verify tool installations
python3 -c "from third_party.evoengineer import KernelOptimizer; print('✓ EvoEngineer OK')"
python3 -c "from third_party.robust_kbench import BenchmarkRunner; print('✓ robust-kbench OK')"
```

## Future Migration

When these tools mature:
1. Extract to separate repositories
2. Convert to git submodules
3. Update commit hashes here
4. Add `git submodule add <url> third_party/<tool>` commands

## Update History

- **2025-10-14**: Initial internal implementation (Phase 1)

