# FlashMoE-Science: Quick Start Guide

**Get started with CUDA kernel development in 5 minutes**

---

## 🎯 What You're Building

A production-grade CUDA kernel library featuring:
- **FlashAttention-Science**: 2x faster attention with FA4 warp specialization
- **Fused MoE**: 4x faster mixture-of-experts (256 experts)
- **Framework Integration**: vLLM, SGLang, TorchTitan, Megatron-LM
- **Scientific Benchmarks**: Materials discovery applications

**Goal**: Demonstrate world-class CUDA expertise for Periodic Labs

---

## 🚀 5-Minute Setup

### 1. Navigate to Project
```bash
cd /Users/kiteboard/periodicdent42/flashmoe-science
```

### 2. Create Environment
```bash
conda create -n flashmoe python=3.10 cuda-toolkit=12.3 -c nvidia
conda activate flashmoe
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123
pip install -r requirements.txt
```

### 4. Build Extensions
```bash
python setup.py build_ext --inplace
```

### 5. Run Tests (will fail initially - expected!)
```bash
pytest tests/ -v
```

**Why tests fail**: Kernel stub returns identity operation (not implemented yet).  
**That's OK!** The build system works, now you implement the kernel.

---

## 📚 What to Read

### Start Here (30 minutes)
1. **`README.md`** - Project overview (5 min)
2. **`PROJECT_STATUS.md`** - What's built, what's next (10 min)
3. **`DEVELOPMENT_GUIDE.md`** - Implementation guide (15 min, sections 1-2)

### When Implementing (ongoing)
- **`DEVELOPMENT_GUIDE.md` Phase 1** - Step-by-step kernel implementation
- **FlashAttention paper** - Section 3.1 (online softmax algorithm)
- **FlashAttention-2 GitHub** - Reference implementation

---

## 🔨 Implementation Workflow

### Week 1-2: Core Kernel

**Edit**:
```bash
# Open kernel file
code python/flashmoe_science/csrc/flash_attention_science.cu
# or: vim, emacs, nano, etc.
```

**Build**:
```bash
python setup.py build_ext --inplace
```

**Test**:
```bash
pytest tests/test_attention_correctness.py -v
```

**Profile**:
```bash
ncu --set full --export profile python benchmarks/attention_benchmarks.py
```

**Iterate**: Repeat until tests pass and performance targets met.

---

## 🎯 Week 1 Goals

### Day 1-3: Basic Tiling
- Implement `load_kv_tile()` in flash_attention_science.cu
- Implement `compute_qk_matmul()` (Q @ K^T)
- Implement `compute_softmax()` (naive version)
- Implement `compute_attention_v()` (attention @ V)
- **Target**: First test passes (even if slow)

### Day 4-6: Online Softmax
- Implement `online_softmax_update()` (already stubbed)
- Replace naive softmax with online algorithm
- Test numerical stability
- **Target**: All tests pass, no NaN/Inf

### Day 7-9: Warp Specialization
- Refactor kernel for warpgroup parallelism
- Separate MMA, Softmax, Correction work
- **Target**: 1.5x speedup

### Day 10-12: Async Pipeline
- Add async memory copies
- Overlap compute + load
- **Target**: 2x+ total speedup

### Day 13-14: Optimize
- Profile with Nsight Compute
- Tune based on data
- **Target**: >90% occupancy, >80% bandwidth

---

## 🐛 Troubleshooting

### "CUDA extensions not available"
```bash
python setup.py build_ext --inplace --force
python -c "from flashmoe_science import _C; print('✓')"
```

### Tests fail with large errors
- Check online softmax implementation
- Compare intermediate values with PyTorch
- Ensure proper scaling factors

### Kernel crash
```bash
CUDA_LAUNCH_BLOCKING=1 pytest tests/ -v
cuda-memcheck python -c "import tests.test_attention_correctness"
```

### Low performance
```bash
# Profile first!
ncu --set full --export profile python benchmarks/...
ncu-ui profile.ncu-rep
# Then optimize based on metrics (see DEVELOPMENT_GUIDE.md)
```

---

## 📊 Success Metrics

### Week 2 Target (MVP)
- [ ] All tests pass (<1e-2 max error)
- [ ] 2x+ speedup vs PyTorch SDPA
- [ ] >90% SM occupancy
- [ ] Nsight profile looks good

### Week 4 Target (Portfolio-Ready)
- [ ] vLLM integration working
- [ ] Scientific benchmarks 2.5x+ faster
- [ ] 3 blog posts published
- [ ] Demo video recorded

---

## 💡 Pro Tips

1. **Start simple**: Basic tiling first, optimize later
2. **Test incrementally**: After every change
3. **Reference FA2**: Their code is production-quality
4. **Profile first**: Never guess bottlenecks
5. **Ask for help**: GPU MODE Discord is great

---

## 📞 Resources

- **Documentation**: `DEVELOPMENT_GUIDE.md` (comprehensive)
- **Paper**: FlashAttention (https://arxiv.org/abs/2205.14135)
- **Reference**: FA2 GitHub (https://github.com/Dao-AILab/flash-attention)
- **Community**: GPU MODE Discord (https://discord.gg/gpumode)
- **CUDA Docs**: https://docs.nvidia.com/cuda/

---

## 🚀 Next Steps

1. ✅ Read `PROJECT_STATUS.md` (understand what's built)
2. ✅ Read `DEVELOPMENT_GUIDE.md` Phase 1, Step 1
3. 🚧 Implement basic tiling (Day 1-3)
4. ⏳ Get first test passing
5. ⏳ Continue with online softmax (Day 4-6)

---

**You've got this!** The foundation is solid, documentation is comprehensive, and the path forward is clear. Now go write some CUDA code. 🚀

**Project**: FlashMoE-Science  
**Status**: Foundation complete, ready for implementation  
**Week 1 Focus**: Basic tiling → Online softmax → Warp specialization  
**Goal**: 2x speedup by end of Week 2

