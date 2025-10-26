# FlashCore Repository Guide

**Last Updated**: October 26, 2025  
**Status**: Production-ready, professional structure

---

## Repository Structure

### Root Directory (Minimal - Industry Standard)

```
periodicdent42/
├── README.md                 # Main entry point
├── LICENSE                   # Apache 2.0
├── setup.py                  # Package setup
├── pyproject.toml            # Build configuration
├── Makefile                  # Build automation
├── Justfile                  # Task runner
├── flashcore/                # Source code
├── tests/                    # Test suite
├── docs/                     # Documentation
├── examples/                 # Usage examples
├── scripts/                  # Utility scripts
├── config/                   # Configuration files
└── archive/                  # Historical artifacts
```

**Philosophy**: Keep root minimal and navigable, matching standards of Triton, PyTorch, and NVIDIA repositories.

---

## Source Code (`flashcore/`)

```
flashcore/
├── __init__.py
├── fast/
│   └── attention_production.py    # Main kernels (200 lines)
├── llama_integration.py           # LLaMA integration (339 lines)
├── torch_ops.py                   # PyTorch operators
├── benchmark/                     # Benchmarking tools
└── tests/                         # Unit tests
```

**Key Files**:
- `fast/attention_production.py`: Production Triton kernels
  - `attention()`: Basic FlashAttention
  - `attention_with_kv_cache()`: KV cache + GQA + causal support

---

## Tests (`tests/`)

```
tests/
├── test_kv_cache_correctness.py    # KV cache validation (4 tests)
├── test_gqa_correctness.py         # GQA validation (5 tests)
├── test_causal_correctness.py      # Causal masking validation (5 tests)
└── test_llama31_validation.py      # LLaMA integration tests
```

**Status**: 15/15 tests pass on NVIDIA H100

**Run Tests**:
```bash
python tests/test_kv_cache_correctness.py
python tests/test_gqa_correctness.py
python tests/test_causal_correctness.py
```

---

## Documentation (`docs/`)

### User-Facing Documentation

```
docs/
├── CUDA_COOKBOOK.md              # Best practices and optimization
├── architecture.md               # System design
├── getting-started/              # Installation and quick start
└── guides/                       # Usage guides
```

### Implementation Details

```
docs/implementation/
├── PHASE1_KV_CACHE_SPEC.md
├── PHASE2_GQA_SPEC.md
├── PHASE3_CAUSAL_SPEC.md
└── PHASE4_LLAMA31_VALIDATION.md
```

### Validation Reports

```
docs/validation/
└── TEST_SUITE_COMPLETE.md        # Complete test results (15/15 pass)
```

### Archive (Progress Documentation)

```
docs/archive/
├── progress/                      # Development progress docs (14 files)
│   ├── CACHE_BUG_FIX_COMPLETE.md
│   ├── CACHE_DEEP_DIVE_COMPLETE.md
│   ├── EDGE_CASE_OPTIMIZATION_STATUS.md
│   └── ...
├── PROFESSIONAL_CV_2025.md        # Private CV
├── REPOSITORY_STRUCTURE.md        # Old structure doc
└── SECURITY_FRAMEWORK_ASSESSMENT.md
```

**Note**: Archive contains historical development documentation, useful for understanding the evolution but not required for usage.

---

## Scripts (`scripts/`)

```
scripts/
├── deployment/
│   └── deploy_llama_validation_h100.sh    # H100 deployment automation
└── benchmarking/                          # Performance benchmarks
```

---

## Examples (`examples/`)

```
examples/
├── quick_start.py                # Basic usage example
└── README.md                     # Example documentation
```

---

## Configuration (`config/`)

```
config/
├── requirements.txt              # Core dependencies
├── requirements-dev.lock         # Development dependencies
└── requirements-full.lock        # Full dependency lock
```

---

## Key Design Principles

### 1. Minimal Root

**Why**: Professional repos (Triton, PyTorch, NVIDIA) keep root clean
- Easy to navigate
- Clear entry point (README.md)
- No clutter from progress docs

**Before**: 15+ markdown files in root  
**After**: 1 markdown file (README.md)

### 2. Organized Documentation

**Structure**:
- User-facing → `docs/` (top level)
- Implementation → `docs/implementation/`
- Validation → `docs/validation/`
- Archive → `docs/archive/` (historical)

### 3. Clear Separation

**Code**: `flashcore/` (source) + `tests/` (validation)  
**Docs**: `docs/` (documentation)  
**Tools**: `scripts/` (utilities)  
**Config**: `config/` (configuration)

---

## Navigation Guide

### I want to...

**Use FlashCore in my project**:
→ Start with `README.md`
→ Then `examples/quick_start.py`
→ Then `docs/guides/`

**Understand the implementation**:
→ Read `docs/architecture.md`
→ Read `docs/implementation/PHASE*.md`
→ Study `flashcore/fast/attention_production.py`

**Run tests**:
→ See `tests/README.md` (if exists) or run tests directly
→ Check `docs/validation/TEST_SUITE_COMPLETE.md` for expected results

**Optimize performance**:
→ Read `docs/CUDA_COOKBOOK.md`
→ Use `scripts/benchmarking/` tools

**Integrate with LLaMA**:
→ See `flashcore/llama_integration.py`
→ Read `docs/implementation/PHASE4_LLAMA31_VALIDATION.md`

**Understand the development process**:
→ Check `docs/archive/progress/` for historical context

---

## Comparison: Before vs After

### Before (Cluttered)

```
Root: 15+ markdown files
- CACHE_BUG_FIX_COMPLETE.md
- CACHE_DEEP_DIVE_COMPLETE.md
- EDGE_CASE_OPTIMIZATION_STATUS.md
- H100_DEPLOYMENT_STATUS.md
- PHASE1_STATUS.md
- PHASE4_COMPLETE_SUMMARY.md
- ... (9 more files)

Problem: Hard to navigate, unprofessional first impression
```

### After (Clean)

```
Root: 1 markdown file
- README.md (professional entry point)

Progress docs: docs/archive/progress/
Scripts: scripts/deployment/
Validation: docs/validation/

Result: Professional, easy to navigate, matches industry standards
```

---

## Industry Standard Comparison

### Triton

```
triton/
├── README.md
├── LICENSE
├── setup.py
├── python/
├── test/
└── docs/
```

### PyTorch

```
pytorch/
├── README.md
├── LICENSE
├── setup.py
├── torch/
├── test/
└── docs/
```

### FlashCore (Now)

```
periodicdent42/
├── README.md
├── LICENSE
├── setup.py
├── flashcore/
├── tests/
└── docs/
```

**Result**: ✅ Matches professional standards

---

## Maintenance

### Adding New Features

1. Implement in `flashcore/`
2. Add tests in `tests/`
3. Update `README.md` if user-facing
4. Document in `docs/implementation/`

### Updating Documentation

1. User guides → `docs/guides/`
2. Architecture changes → `docs/architecture.md`
3. Implementation details → `docs/implementation/`
4. Test results → `docs/validation/`

### Archiving Progress Docs

1. Development notes → `docs/archive/progress/`
2. Keep root clean
3. Link from main docs if needed

---

## Summary

**Structure**: Professional, minimal root, organized subdirectories  
**Standards**: Matches Triton, PyTorch, NVIDIA best practices  
**Navigation**: Clear, intuitive, easy to find information  
**Maintenance**: Scalable structure for future growth  

**Status**: Production-ready open-source repository ✅

---

**Last Updated**: October 26, 2025  
**Repository**: https://github.com/GOATnote-Inc/periodicdent42

