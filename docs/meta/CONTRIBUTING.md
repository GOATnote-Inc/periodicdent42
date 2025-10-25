# Contributing to FlashCore

Thank you for your interest in contributing to FlashCore! This document provides guidelines for contributions.

---

## 🌟 Ways to Contribute

### Code Contributions
- **Performance optimizations** (new GPU architectures, block sizes)
- **Correctness improvements** (additional test cases, bug fixes)
- **Platform support** (new GPUs, operating systems)
- **API enhancements** (new features, better interfaces)

### Documentation
- **Tutorials** (getting started guides, use cases)
- **Examples** (Jupyter notebooks, sample applications)
- **API documentation** (docstrings, references)
- **Performance guides** (tuning tips, benchmarking)

### Testing
- **Validation on new hardware** (report results)
- **Edge case testing** (unusual configurations)
- **Continuous integration** (CI improvements)

### Community
- **Bug reports** (GitHub Issues)
- **Feature requests** (well-motivated proposals)
- **Answering questions** (helping other users)

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/periodicdent42.git
cd periodicdent42

# Add upstream remote
git remote add upstream https://github.com/GOATnote-Inc/periodicdent42.git
```

### 2. Set Up Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies (linting, testing)
pip install -r requirements-dev.txt

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

---

## 📝 Development Guidelines

### Code Style

**Python**:
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 88)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints where appropriate

**CUDA/C++**:
- Follow [CUDA C++ Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- Use consistent indentation (4 spaces)
- Comment non-obvious optimizations

**Documentation**:
- Use [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings
- Include type annotations
- Provide usage examples

### Example Docstring

```python
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64
) -> torch.Tensor:
    """
    Scaled dot-product attention with FlashAttention optimization.
    
    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        block_m: Block size for queries (default: 64)
        block_n: Block size for keys/values (default: 64)
        
    Returns:
        Output tensor [B, H, N, D]
        
    Raises:
        ValueError: If tensor shapes are incompatible
        
    Example:
        >>> q = torch.randn(16, 8, 512, 64, device='cuda', dtype=torch.float16)
        >>> output = attention(q, q, q)
        >>> output.shape
        torch.Size([16, 8, 512, 64])
    """
    ...
```

---

## 🧪 Testing

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Correctness validation
python flashcore/benchmark/expert_validation.py

# Performance benchmarks
python flashcore/benchmark/benchmark.py
```

### Writing Tests

**Correctness Tests**:
```python
import torch
from flashcore.fast.attention_production import attention

def test_correctness_basic():
    """Test numerical correctness against PyTorch SDPA."""
    q = torch.randn(16, 8, 512, 64, device='cuda', dtype=torch.float16)
    k, v = q.clone(), q.clone()
    
    # Reference
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    
    # Our implementation
    out = attention(q, k, v)
    
    # Check
    assert torch.allclose(out, ref, rtol=1e-3, atol=2e-3)
```

**Performance Tests**:
```python
def test_performance_target():
    """Test that kernel meets < 5 μs target on H100."""
    q = torch.randn(16, 8, 512, 64, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(100):
        _ = attention(q, q, q)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = attention(q, q, q)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # μs
    
    median_us_per_seq = sorted(times)[len(times)//2] / 16
    assert median_us_per_seq < 5.0, f"Expected < 5 μs, got {median_us_per_seq:.2f} μs"
```

---

## 📋 Pull Request Process

### 1. Prepare Your Changes

```bash
# Run linters
black flashcore/
isort flashcore/
flake8 flashcore/

# Run tests
pytest tests/ -v

# Update documentation if needed
```

### 2. Commit Guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <subject>

git commit -m "feat(kernel): add support for A100 GPU"
git commit -m "fix(validation): correct numerical tolerance in tests"
git commit -m "docs(readme): update installation instructions"
git commit -m "perf(attention): optimize block size for L4"
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `perf`: Performance improvement
- `test`: Adding tests
- `refactor`: Code refactoring
- `style`: Code style changes
- `chore`: Build/tooling changes

### 3. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- **Clear title** (following Conventional Commits)
- **Description** of changes
- **Motivation** (why is this needed?)
- **Testing** (how was it validated?)
- **Screenshots/results** (if applicable)

### 4. PR Review Process

**Reviewers will check**:
- ✅ Code follows style guidelines
- ✅ Tests pass (CI)
- ✅ Documentation is updated
- ✅ Commits are clean and well-described
- ✅ No breaking changes (or clearly documented)

**Response time**: We aim to review PRs within 3-5 business days.

---

## 🎯 Contribution Areas

### High Priority
- ✅ **Additional GPU support** (A100, H200, A6000)
- ✅ **Extended sequence lengths** (1024, 2048, 4096)
- ✅ **Causal attention** variant
- ✅ **FP8 precision** support

### Medium Priority
- 📊 **Benchmarking suite** improvements
- 📖 **Tutorial notebooks** (Jupyter)
- 🧪 **Stress tests** (edge cases)
- 🔧 **Build system** improvements

### Community Requests
Check [GitHub Issues](https://github.com/GOATnote-Inc/periodicdent42/issues) for current requests.

---

## 🔬 Validation Standards

### Performance Claims

If contributing performance improvements:
1. **Measure on real hardware** (not emulated)
2. **Use device-time** (CUDA events, not host time)
3. **Report statistics** (median, p95, p99 over 100+ trials)
4. **Compare to baseline** (PyTorch SDPA)
5. **Test multiple configurations** (different B, S, H, D)

### Example Performance Report

```markdown
## Performance Results

**Hardware**: NVIDIA A100 40GB  
**Configuration**: B=16, H=8, S=512, D=64  
**Trials**: 1000  

| Metric | Value |
|--------|-------|
| P50 | 4.23 μs/seq |
| P95 | 4.45 μs/seq |
| P99 | 4.78 μs/seq |
| vs SDPA | 0.95× (5% faster) |

**Correctness**: ✅ torch.allclose (rtol=1e-3, atol=2e-3)
```

### Correctness Standards

**Required**:
- ✅ Pass `torch.allclose(out, ref, rtol=1e-3, atol=2e-3)`
- ✅ Test at least 3 different sequence lengths
- ✅ Test at least 2 different batch sizes
- ✅ Validate on real hardware

---

## 🏛️ Attribution

### Adding Dependencies

When adding new dependencies:
1. Update `requirements.txt` or `requirements-dev.txt`
2. Add attribution to `ATTRIBUTIONS.md`
3. Check license compatibility (must be compatible with Apache 2.0)
4. Document in PR description

### Citing Research

When implementing ideas from papers:
1. Add citation to `CITATIONS.bib`
2. Reference in code comments
3. Acknowledge in `ATTRIBUTIONS.md`
4. Credit in PR description

---

## 📜 Code of Conduct

### Our Pledge

We pledge to make participation in FlashCore a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience
- Nationality, personal appearance
- Race, religion, sexual orientation

### Our Standards

**Positive behavior**:
- ✅ Using welcoming and inclusive language
- ✅ Being respectful of differing viewpoints
- ✅ Gracefully accepting constructive criticism
- ✅ Focusing on what's best for the community
- ✅ Showing empathy towards others

**Unacceptable behavior**:
- ❌ Trolling, insulting/derogatory comments
- ❌ Public or private harassment
- ❌ Publishing others' private information
- ❌ Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to:
- **Email**: b@thegoatnote.com
- **GitHub**: Open a confidential issue

All complaints will be reviewed and investigated promptly and fairly.

---

## 💡 Tips for First-Time Contributors

### Good First Issues

Look for issues labeled:
- `good first issue` - Beginner-friendly
- `help wanted` - Community contributions welcome
- `documentation` - Documentation improvements

### Getting Help

**Before asking**:
1. Read the [documentation](docs/)
2. Search [existing issues](https://github.com/GOATnote-Inc/periodicdent42/issues)
3. Check [discussions](https://github.com/GOATnote-Inc/periodicdent42/discussions)

**How to ask**:
- Be specific about your problem
- Include error messages and logs
- Mention your environment (GPU, CUDA version, PyTorch version)
- Show what you've tried

---

## 🎓 Learning Resources

### CUDA Programming
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- *Programming Massively Parallel Processors* by Kirk & Hwu

### Attention Mechanisms
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023)

### Triton
- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

---

## 📊 Benchmarking Guidelines

### Hardware Access

**Don't have a GPU?**
- Use free credits: [Google Colab](https://colab.research.google.com/) (limited T4)
- Cloud credits: [RunPod](https://runpod.io/), [Lambda Labs](https://lambdalabs.com/)
- Academic: Many universities provide GPU clusters

**Reporting limitations**:
- ℹ️ Clearly state hardware limitations
- ℹ️ Mark results as "preliminary" if on low-end hardware
- ℹ️ Welcome validation from others with better hardware

---

## 🔐 Security

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities.

Instead:
1. Email: b@thegoatnote.com
2. Include: Description, impact, reproduction steps
3. Expect: Response within 48 hours

### Security Considerations

When contributing:
- ✅ No hardcoded credentials
- ✅ No timing side-channels in crypto-related code
- ✅ Validate all inputs
- ✅ Document security assumptions

---

## 📞 Contact

**General questions**: [GitHub Discussions](https://github.com/GOATnote-Inc/periodicdent42/discussions)  
**Bug reports**: [GitHub Issues](https://github.com/GOATnote-Inc/periodicdent42/issues)  
**Security**: b@thegoatnote.com  
**Code of Conduct**: b@thegoatnote.com

---

## 🙏 Thank You!

Every contribution, no matter how small, makes FlashCore better. We appreciate your time and effort!

<p align="center">
  <strong>Happy Contributing! 🚀</strong>
</p>
