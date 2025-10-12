# Reproducibility Playbook

The CUDA reboot folder is designed to rebuild the FlashAttention-Science and fused MoE kernels from scratch and validate their performance against modern HPC baselines.

## Hardware

- **GPU:** NVIDIA H100 SXM (80 GB) or H100 PCIe (80 GB). A100 80 GB can be used with the `--preset ampere-a100` flag but will not support FP8 paths.
- **CPU:** Dual AMD EPYC 7xx3 or Intel Xeon Ice Lake with AVX-512.
- **Memory:** 256 GB system RAM.

## Software Stack

| Component | Version | Notes |
|-----------|---------|-------|
| Ubuntu | 22.04 LTS | Kernel 5.15+ |
| CUDA Toolkit | 12.3.2 | Install from NVIDIA network repo |
| NVIDIA Driver | 535.129.03 | Matching CUDA 12.3 |
| Python | 3.10.13 | Managed via Conda |
| PyTorch | 2.2.1+cu123 | `pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu123` |
| flash-attn | 2.3.3.post1 | Provides SOTA baseline |
| triton | 2.1.0 | Required by flash-attn |
| numpy | 1.26 | |
| scipy | 1.11 | |
| pytest | 7.4 | For correctness tests |

## Environment Setup

```bash
# Clone repository
cd ~/workspace
git clone https://github.com/GOATnote-Inc/periodicdent42.git
cd periodicdent42

# Create environment
conda create -n cudadent42 python=3.10 -y
conda activate cudadent42

# Install CUDA toolkit + compilers
conda install -c nvidia cuda-toolkit=12.3 -y

# Core dependencies
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu123
pip install flash-attn==2.3.3.post1 triton==2.1.0
pip install -r cudadent42/requirements.txt

# Build CUDA extensions
cd cudadent42
python setup.py clean
python setup.py build_ext --inplace

# Validate correctness
pytest tests/test_attention_correctness.py -v
pytest tests/test_warp_specialized.py -v -k "attention or moe"
```

## Benchmark Execution

```bash
# From repository root
git checkout cuda-reboot
conda activate cudadent42

# Attention benchmark
python cuda_reboot/benchmarks/run_attention_benchmarks.py \
    --preset hopper-h100 \
    --output cuda_reboot/benchmarks/results/flash_attention_h100_latest.json

# MoE benchmark
python cuda_reboot/benchmarks/run_moe_benchmarks.py \
    --preset hopper-h100 \
    --output cuda_reboot/benchmarks/results/fused_moe_h100_latest.json
```

Both scripts print markdown and JSON summaries, and they write raw timing samples to disk for auditability.

## Verification Checklist

1. `torch.cuda.is_available()` returns `True` and GPU reports `H100`.
2. `flashmoe_science` extension loads successfully (`python -c "import flashmoe_science"`).
3. Attention and MoE correctness tests pass.
4. Benchmarks complete with speedup ≥1.8× vs PyTorch baselines and ≥1.2× vs flash-attn / DeepSpeed baselines.

Record the Git commit, GPU SKU, driver, and benchmark outputs for provenance.
