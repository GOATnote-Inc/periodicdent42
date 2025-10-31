# Contributing to BlackwellSparseK

Thank you for your interest in contributing to BlackwellSparseK! This project builds on incredible work from the research and open-source communities.

## 🎓 Attribution & Citations

**BlackwellSparseK stands on the shoulders of giants.** If you contribute, please maintain this spirit of attribution.

### Core Dependencies & Citations

This project builds upon:

1. **SparseK** (arXiv:2406.16747)
   - Paper: "SparseK: Learned Sparse Attention for Efficient LLM Inference"
   - Authors: Mingjie Sun, et al.
   - Citation: `@article{sun2024sparsek, title={SparseK: Learned Sparse Attention for Efficient LLM Inference}, author={Sun, Mingjie and others}, journal={arXiv preprint arXiv:2406.16747}, year={2024}}`

2. **NVIDIA CUTLASS** (4.3.0+)
   - Library: https://github.com/NVIDIA/cutlass
   - License: BSD 3-Clause
   - Thanks to the NVIDIA CUTLASS team for world-class GEMM kernels

3. **Meta xFormers** (0.0.22.post2)
   - Library: https://github.com/facebookresearch/xformers
   - License: BSD 3-Clause
   - Thanks to Meta FAIR for memory-efficient attention primitives

4. **vLLM** (0.11.0)
   - Library: https://github.com/vllm-project/vllm
   - Authors: UC Berkeley Sky Computing Lab
   - License: Apache 2.0
   - Thanks to the vLLM team for production serving infrastructure

5. **FlashAttention & FlashAttention-2**
   - Papers: Dao et al., Stanford/Princeton
   - Inspiration for our tiling and online softmax algorithms

### When Contributing

**Always**:
- ✅ Cite original papers when implementing algorithms
- ✅ Acknowledge libraries and frameworks you use
- ✅ Link to original source code when adapting implementations
- ✅ Respect original licenses (BSD, Apache, MIT)

**Never**:
- ❌ Copy code without attribution
- ❌ Claim others' work as your own
- ❌ Remove existing attribution comments

## 🤖 Ethical AI Guidelines

BlackwellSparseK is designed for beneficial AI applications, particularly in robotics and LLM inference.

### Encouraged Use Cases
- ✅ Robotics manipulation (RoboNet, RT-2, etc.)
- ✅ LLM inference optimization
- ✅ Edge AI deployment
- ✅ Research and education
- ✅ Open-source projects

### Discouraged Use Cases
- ❌ Autonomous weapons or harmful AI systems
- ❌ Surveillance without consent
- ❌ Discriminatory applications
- ❌ Privacy-violating systems

If you're uncertain about a use case, please open a discussion.

## 📝 How to Contribute

### 1. Code Contributions

**Good First Issues**:
- Add support for new GPU architectures (Ada, Blackwell, Rubin)
- Optimize kernels for specific sequence lengths
- Improve test coverage
- Add benchmarks for new models

**Process**:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes with clear commit messages
4. Add tests for new functionality
5. Run `pytest tests/` and ensure all pass
6. Submit a pull request with description

**Code Standards**:
- Python: Follow PEP 8, use `black` formatter
- CUDA: Comment complex kernels, cite algorithms
- Tests: Add unit tests for new features
- Docs: Update README if adding features

### 2. Documentation Contributions

We welcome:
- Tutorials and guides
- Performance optimization tips
- Deployment examples
- Translation to other languages

### 3. Research Contributions

If you publish research using BlackwellSparseK:
- Please cite this repository and underlying papers
- Share your findings via PRs or discussions
- Help us improve based on your insights

### 4. Bug Reports

**Before reporting**:
- Check existing issues
- Verify on latest version
- Test on clean environment

**Include**:
- GPU model and driver version
- CUDA version
- PyTorch version
- Minimal reproducible example
- Error messages and logs

### 5. Feature Requests

Open an issue with:
- Use case description
- Proposed API (if applicable)
- Performance expectations
- Willingness to contribute implementation

## 💰 Bounty Program

We offer bounties for significant contributions:

| Contribution | Bounty | Status |
|--------------|--------|--------|
| Add Rubin R100 support (sm_100+) | $500 | Open |
| Blackwell B200 optimizations | $300 | Open |
| FP8 E4M3 sparse attention | $400 | Open |
| vLLM PagedAttention integration | $600 | Open |
| HuggingFace Transformers integration | $500 | Open |

**Claiming a bounty**:
1. Comment on the issue to claim it
2. Submit PR within 30 days
3. Pass review and CI tests
4. Receive bounty via GitHub Sponsors or Open Collective

## 🔒 Security

See [SECURITY_NOTICE.md](SECURITY_NOTICE.md) for guidelines.

**If you find a security vulnerability**:
- **DO NOT** create a public issue
- Email: security@blackwellsparsek.dev (or create private security advisory)
- Include: Description, impact, steps to reproduce
- We respond within 24 hours

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License (see [LICENSE](LICENSE)).

**Additional requirement**: Please add attribution comment to any substantial code contribution:
```python
# Contributed by [Your Name] <your.email@example.com>
# Based on [Original Work] by [Original Authors]
```

## 🌟 Recognition

All contributors are recognized in:
- [README.md](README.md) Contributors section
- Release notes for their contributions
- Academic papers citing this work (with permission)

## 📞 Getting Help

- **General questions**: GitHub Discussions
- **Bug reports**: GitHub Issues
- **Real-time chat**: Discord (coming soon)
- **Email**: contribute@blackwellsparsek.dev

## 🙏 Thank You

Every contribution, no matter how small, helps advance open-source AI for robotics and beyond.

**Top Contributors** (updated monthly):
- [Your Name Here] - First contribution!

---

**By contributing to BlackwellSparseK, you're joining a community committed to ethical, attributed, and impactful open-source AI.**

