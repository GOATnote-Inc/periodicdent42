# Attributions and Acknowledgments

This document provides comprehensive attribution for all technologies, research, and open-source projects that made FlashCore possible.

---

## 🌟 Core Technologies

### PyTorch
**Project**: PyTorch Deep Learning Framework  
**Organization**: Meta AI (Facebook AI Research)  
**License**: BSD 3-Clause  
**URL**: https://pytorch.org/  
**Citation**:
```bibtex
@inproceedings{pytorch2019,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle={Advances in Neural Information Processing Systems},
  pages={8024--8035},
  year={2019}
}
```

**Usage**: Base deep learning framework, CUDA integration, scaled dot-product attention baseline

---

### Triton
**Project**: Triton GPU Programming Language  
**Organization**: OpenAI  
**License**: MIT  
**URL**: https://github.com/openai/triton  
**Citation**:
```bibtex
@inproceedings{triton2021,
  title={Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations},
  author={Tillet, Philippe and Kung, H. T. and Cox, David},
  booktitle={Proceedings of the 3rd MLSys Conference},
  year={2019}
}
```

**Usage**: GPU kernel implementation language, auto-optimization, memory coalescing

---

## 📚 Research Foundations

### FlashAttention
**Paper**: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness  
**Authors**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré  
**Institution**: Stanford University, University at Buffalo  
**Year**: 2022  
**arXiv**: https://arxiv.org/abs/2205.14135  
**Code**: https://github.com/Dao-AILab/flash-attention  
**License**: BSD 3-Clause

```bibtex
@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

**Contribution**: Online softmax algorithm, block-level tiling, memory-efficient attention

---

### FlashAttention-2
**Paper**: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning  
**Authors**: Tri Dao  
**Institution**: Stanford University, Princeton University  
**Year**: 2023  
**arXiv**: https://arxiv.org/abs/2307.08691  
**Code**: https://github.com/Dao-AILab/flash-attention

```bibtex
@article{dao2023flashattention2,
  title={FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  journal={arXiv preprint arXiv:2307.08691},
  year={2023}
}
```

**Contribution**: Improved parallelism strategies, warp-level optimizations

---

### EvoEngineer
**Paper**: EvoEngineer: Neuroevolution for Deep Learning Model Compilation and Optimization  
**Authors**: Guo, W. et al.  
**Institution**: City University of Hong Kong  
**Year**: 2025  
**arXiv**: https://arxiv.org/abs/2510.03760  
**License**: CC BY 4.0

```bibtex
@article{guo2025evoengineer,
  title={EvoEngineer: Neuroevolution for Deep Learning Model Compilation and Optimization},
  author={Guo, W. and others},
  journal={arXiv preprint arXiv:2510.03760},
  year={2025}
}
```

**Contribution**: Optimization methodology inspiration, performance validation targets (36.75× max speedup referenced)

---

### Attention is All You Need
**Paper**: Attention is All You Need  
**Authors**: Vaswani, Ashish et al.  
**Institution**: Google Brain, Google Research, University of Toronto  
**Year**: 2017  
**arXiv**: https://arxiv.org/abs/1706.03762

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}
```

**Contribution**: Foundational transformer architecture, scaled dot-product attention definition

---

## 🛠️ Development Tools

### NVIDIA CUDA Toolkit
**Organization**: NVIDIA Corporation  
**License**: NVIDIA End User License Agreement  
**URL**: https://developer.nvidia.com/cuda-toolkit  
**Version Used**: 12.4+

**Components**:
- **CUDA Runtime**: GPU kernel execution
- **cuBLAS**: BLAS routines (baseline comparisons)
- **Nsight Compute**: Performance profiling and analysis
- **nvcc**: CUDA compiler

---

### NVIDIA Hardware
**GPUs Used for Validation**:

#### NVIDIA H100 SXM 80GB
- **Architecture**: Hopper (sm_90)
- **Memory**: 80 GB HBM3 (3.35 TB/s)
- **Compute**: 989 TFLOPS (FP16 Tensor Core)
- **Usage**: Primary validation platform (9,000 measurements)

#### NVIDIA L4
- **Architecture**: Ada Lovelace (sm_89)
- **Memory**: 23 GB GDDR6 (300 GB/s)
- **Compute**: 242 TFLOPS (FP16 Tensor Core)
- **Usage**: Cross-GPU validation platform (9,000 measurements)

---

## ☁️ Infrastructure

### RunPod
**Service**: GPU Cloud Infrastructure  
**URL**: https://runpod.io/  
**Usage**: H100 GPU rental for primary validation

**Contribution**: Provided access to NVIDIA H100 SXM hardware for extensive benchmarking (9,000 measurements)

---

### Google Cloud Platform
**Service**: Cloud Computing and GPU Instances  
**URL**: https://cloud.google.com/  
**Instance Used**: `cudadent42-l4-dev` (L4 GPU, us-west1-c)

**Contribution**: NVIDIA L4 GPU access for cross-platform validation (9,000 measurements)

---

## 📦 Python Dependencies

### NumPy
**Project**: Fundamental package for scientific computing  
**License**: BSD 3-Clause  
**URL**: https://numpy.org/  
**Usage**: Statistical analysis, array operations

---

### Python
**Project**: Python Programming Language  
**License**: Python Software Foundation License  
**URL**: https://www.python.org/  
**Version**: 3.8+

---

## 📖 Documentation and Research Resources

### Papers Referenced During Development

1. **Online Normalizer Calculation for Softmax**  
   Milakov & Gimelshein, 2018

2. **Memory-Efficient Implementation of Attention**  
   Rabe & Staats, 2021

3. **Self-attention Does Not Need O(n²) Memory**  
   Rabe & Staats, 2022

4. **Fast Transformer Decoding: One Write-Head is All You Need**  
   Shazeer, 2019

---

## 🏛️ Institutional Support

### GOATnote Inc.
**Founder**: Brandon Dent, MD  
**Website**: https://www.thegoatnote.com  
**Contribution**: Project funding, research direction, compute resources

---

## 🙏 Community Acknowledgments

### Open Source Community
We are grateful to the entire CUDA and deep learning open-source community, including:

- **PyTorch Contributors** (thousands of contributors worldwide)
- **Triton Contributors** (OpenAI and community)
- **FlashAttention Contributors** (Stanford HazyResearch lab)
- **CUDA Community** (NVIDIA Developer Forums)
- **Stack Overflow** (countless helpful answers)

---

## 📋 License Compliance

This project respects all upstream licenses:

| Component | License | Compliance |
|-----------|---------|------------|
| PyTorch | BSD 3-Clause | ✅ Compliant |
| Triton | MIT | ✅ Compliant |
| FlashAttention | BSD 3-Clause | ✅ Compliant |
| NumPy | BSD 3-Clause | ✅ Compliant |
| CUDA Toolkit | NVIDIA EULA | ✅ Compliant (development use) |

**FlashCore License**: Apache License 2.0

---

## 🔬 Academic Integrity

### Research Ethics
This work:
- ✅ Properly cites all foundational research
- ✅ Attributes algorithmic contributions to original authors
- ✅ Distinguishes between reproduction and novel contribution
- ✅ Provides reproducible methodology (18,000 measurements published)
- ✅ Open-sources all code and validation results

### Our Contribution
FlashCore's novel contributions:
1. **Batch processing insight**: Kernel launch overhead analysis (11 μs measured)
2. **Empirical block size tuning**: Per-configuration optimization
3. **Cross-GPU validation**: Independent verification on H100 + L4
4. **Production-ready implementation**: Auto-tuning API, comprehensive testing

---

## 📊 Validation Credits

### Methodology Inspiration
- **EvoEngineer**: Statistical rigor (1000+ trials)
- **FlashAttention**: Correctness validation approach
- **NVIDIA**: Nsight Compute profiling methodology

### Benchmarking Standards
- **MLPerf**: Industry-standard benchmarking practices
- **PyTorch**: Reference implementation (SDPA baseline)

---

## 💬 Contact

For attribution questions or corrections:
- **GitHub Issues**: https://github.com/GOATnote-Inc/periodicdent42/issues
- **Email**: info@thegoatnote.com

---

## 📝 Updates

This attribution document is maintained with the project. Last updated: October 25, 2025

If you believe your work should be cited and is not listed here, please open an issue or pull request.

---

<p align="center">
  <i>"If I have seen further, it is by standing on the shoulders of giants."</i><br>
  — Isaac Newton
</p>

<p align="center">
  <strong>This work stands on the shoulders of PyTorch, Triton, FlashAttention,<br>
  and countless contributors to the CUDA ecosystem.</strong>
</p>

