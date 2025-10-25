# FlashCore Agent Operating Manual

**Version**: 1.0  
**Date**: October 21, 2025  
**Purpose**: Ground rules for AI agents working on FlashCore GPU kernel optimization

---

## 🎯 Mission & Constraints

### Hardware
- **GPU**: NVIDIA L4 (GCP instance) - Ada architecture, sm_89
- **Rule**: Always use GPU for all heavy tasks (compile, test, bench, profile)
- **Never**: Fall back to CPU without explicit permission
- **Policy**: Engineering time >> GPU cost → keep GPU alive, don't idle

### Toolchain
- **CUDA**: 12.2 (required)
- **PyTorch**: 2.x with CUDA support (required)
- **Nsight Compute**: Installed and in PATH (ncu command available)
- **Compiler**: nvcc, torch.utils.cpp_extension must be accessible

### Repository
- **Root**: `/Users/kiteboard/periodicdent42/`
- **Active Project**: `flashcore/` (new kernel development)
- **Historical**: `cudadent42/` (reference implementations)
- **Current Branch**: Check with `git status -sb`

### Project Phase
- ✅ **Phase 0**: Baseline verified (1500 µs, correct)
- 🔄 **Phase 1**: WMMA Tensor Cores (target: ~150 µs, 10× speedup)
- ⏳ **Phase 2**: FlashAttention fusion (target: <58 µs, ≥15× speedup) **← PRIMARY GOAL**
- ⏳ **Phase 3**: Warp specialization (stretch: ~20 µs)
- ⏳ **Phase 4**: Evolutionary search (bonus: ~15 µs)

### Compilation Target
- **Architecture**: sm_89 (L4, Ada)
- **Default Flags**: `-O3 --use_fast_math -lineinfo -Xptxas -v -arch=sm_89`
- **Environment**: `CUDA_ARCH=8.9`, `TORCH_CUDA_ARCH_LIST=8.9`

---

## 🚨 Mandatory Preflight (Before Any Heavy Operation)

**Run before**: compile, profile, benchmark, EvoEngineer sweep, or any GPU work

```bash
bash scripts/preflight.sh
```

**Checks**:
1. ✅ GPU hardware (nvidia-smi)
2. ✅ CUDA toolkit (nvcc)
3. ✅ Nsight Compute (ncu)
4. ✅ PyTorch CUDA (torch.cuda.is_available())
5. ✅ Environment variables (CUDA_HOME, CUDA_ARCH, etc.)

**Auto-Repair**: If checks fail, agent must:
1. Source `scripts/env_cuda_l4.sh`
2. Fix PATH, CUDA_HOME, LD_LIBRARY_PATH as needed
3. Re-run preflight
4. Show one-screen diff of changes

**Exit Codes**:
- `0`: All pass ✅ → proceed
- `1`: Warnings ⚠️ → proceed with caution
- `2`: Critical errors ❌ → STOP, fix before proceeding

---

## 🔧 Operating Heuristics

### Test-Driven Development (TDD)
1. **Write tests first** (or use existing 15-case suite)
2. **Implement code** (small, focused changes)
3. **Run tests** (pytest tests/test_correctness.py -v)
4. **Benchmark** (python benchmarks/benchmark_latency.py --shape mission)
5. **Profile** (ncu with TC utilization metrics)
6. **Iterate** (repeat until success criteria met)

### Incremental Changes
- **Prefer**: Small diffs (50-200 lines per change)
- **Explain**: Plan before implementation (why, what, how)
- **Verify**: Test after each change
- **Summarize**: Report timing (p50/p90/p99), error metrics, resource usage

### Agent Mode Best Practices
- Use **Agent mode** for multi-file edits + terminal commands
- Keep context window efficient (summarize diffs, don't paste full files)
- Log all command outputs (capture in logs/)
- Report failures immediately with diagnostic info

### Before Compile/Bench/Profile
```bash
# 1. Re-run preflight
bash scripts/preflight.sh

# 2. Ensure GPU keep-alive active
bash scripts/keepalive.sh

# 3. Source environment
source scripts/env_cuda_l4.sh

# 4. Proceed with operation
```

---

## 📊 Success Criteria by Phase

### Phase 1 (WMMA) - Current Target
- ✅ **PTXAS**: ≤120 registers, ≤64 KB shared memory, 0 spills
- ✅ **Correctness**: All 15 tests pass (max_err ≤ 0.06)
- ✅ **Performance**: <200 µs on mission shape (≥7× vs baseline)
- ✅ **NCU**: Tensor Core utilization ≥50%
- ✅ **Deliverables**: kernels/flashcore_wmma.cu, docs/PHASE1_REPORT.md, benchmark JSONs

### Phase 2 (Fusion) - Primary Goal
- ✅ **Performance**: <58 µs (≥15× vs 870 µs old PyTorch baseline)
- ✅ **Correctness**: All 15 tests pass
- ✅ **NCU**: DRAM throughput >50% of peak (memory-bound expected)
- ✅ **Deliverables**: kernels/flashcore_fused.cu, docs/PHASE2_REPORT.md

### Phase 3 (Advanced) - Stretch
- ✅ **Performance**: 15-30 µs (competitive with FlashAttention-2)
- ✅ **NCU**: <10 thread-block barriers (vs 48 baseline)
- ✅ **Correctness**: Maintained

---

## 🛠️ Standard Operations

### Build Kernel
```bash
cd /Users/kiteboard/periodicdent42/flashcore
source scripts/env_cuda_l4.sh
python build.py
```

### Run Tests (15 cases)
```bash
pytest tests/test_correctness.py -v
# Or specific shape:
pytest tests/test_correctness.py::test_correctness[mission-0] -v
```

### Benchmark
```bash
python benchmarks/benchmark_latency.py --shape mission --iters 100 --out results/phase1_wmma.json
```

### Profile with NCU
```bash
ncu --set full --launch-skip 10 --launch-count 1 \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    python benchmarks/benchmark_latency.py --shape mission --iters 1 \
    > logs/ncu_phase1.txt
```

### Check GPU Status
```bash
nvidia-smi
tail -f logs/gpu_dmon.log
```

---

## 📁 Key Files Reference

### Kernels
- `kernels/flashcore_baseline.cu` - Phase 0 baseline (scalar)
- `kernels/flashcore_wmma.cu` - Phase 1 WMMA (create next)
- `kernels/flashcore_fused.cu` - Phase 2 fusion (future)
- `kernels/bindings.cpp` - PyTorch C++ wrapper

### Infrastructure
- `build.py` - Build system (PyTorch C++ extensions)
- `tests/test_correctness.py` - 15 test cases
- `benchmarks/benchmark_latency.py` - Performance measurement
- `scripts/preflight.sh` - **Run before all heavy ops**
- `scripts/env_cuda_l4.sh` - Environment setup
- `scripts/keepalive.sh` - GPU keep-alive (tmux + dmon)

### Documentation
- `README.md` - Project overview
- `docs/ARCHITECTURE.md` - Technical design
- `docs/GETTING_STARTED.md` - Setup guide
- `docs/PHASE1_WMMA_GUIDE.md` - WMMA implementation guide
- `AGENTS.md` - **This file** (agent operating manual)

### Reference Implementations (periodicdent42)
- `../cudadent42/bench/kernels/fa_minimal.cu` - Baseline source
- `../cudadent42/bench/kernels/fa_wmma_qkt.cu` - WMMA Q@K^T reference
- `../cudadent42/bench/kernels/fa_phase5_wmma.cu` - Full WMMA (has bugs)

---

## 🔒 Safety & Correctness

### Before Committing Code
- ✅ All tests pass
- ✅ No NaN/Inf in outputs
- ✅ PTXAS reports no spills
- ✅ Benchmark results saved to JSON
- ✅ Git commit message references phase and success criteria

### Error Handling
- **Compilation errors**: Check PTXAS output, adjust resource usage
- **Test failures**: Check max_err, investigate numerical stability
- **Performance regression**: Profile with NCU, identify bottleneck
- **GPU OOM**: Reduce batch size or tile size, check memory allocator config

### Logging Policy
- **All commands**: Capture output to logs/
- **Benchmark runs**: Save to results/ with timestamp + git SHA
- **NCU profiles**: Save to logs/ncu_*.txt with descriptive names
- **Git commits**: Include performance summary in commit message

---

## 🎓 Learning Resources

### Papers
- **FlashAttention**: https://arxiv.org/abs/2205.14135 (tiling algorithm)
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691 (warp-level optimization)
- **EvoEngineer**: https://arxiv.org/abs/2510.03760 (LLM-driven kernel optimization)

### NVIDIA Documentation
- **WMMA Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
- **L4 Specs**: 242 TFLOPS FP16 (Tensor Cores), 300 GB/s memory bandwidth
- **Nsight Compute**: https://docs.nvidia.com/nsight-compute/

### Internal References
- `FLASHCORE_LAUNCH_PLAN.md` - Complete project plan
- `FLASHCORE_IMPLEMENTATION_PLAN.md` - Task-by-task breakdown
- `FLASHCORE_KERNEL_AUDIT.md` - Baseline selection rationale

---

## 🚀 Quick Start for New Agent Session

```bash
# 1. Navigate to project
cd /Users/kiteboard/periodicdent42/flashcore

# 2. Run preflight
bash scripts/preflight.sh

# 3. Start GPU keep-alive (if not running)
bash scripts/keepalive.sh

# 4. Check current status
git status -sb
cat README.md | head -50

# 5. Review phase guide (Phase 1 = WMMA)
cat docs/PHASE1_WMMA_GUIDE.md | head -100

# 6. Proceed with current phase task
```

---

## 📞 Escalation

**If preflight fails and auto-repair doesn't work**:
1. Document exact error messages
2. Check CUDA installation: `ls -la /usr/local/cuda*`
3. Check PyTorch build: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
4. Request human intervention with diagnostic info

**If GPU becomes unavailable**:
1. Do NOT proceed with CPU fallback
2. Document the issue
3. Request human to restore GPU access
4. Wait for confirmation before resuming

---

## ✅ Agent Checklist (Every Session)

- [ ] Read this file (AGENTS.md)
- [ ] Run preflight (`bash scripts/preflight.sh`)
- [ ] Check git status
- [ ] Review current phase guide
- [ ] Ensure GPU keep-alive running
- [ ] Understand success criteria for current phase
- [ ] Follow TDD loop (test → code → test → bench → profile)
- [ ] Document all changes (commit messages, reports)
- [ ] Save artifacts (JSONs, NCU logs) before ending session

---

**Version**: 1.0  
**Last Updated**: October 21, 2025  
**Status**: Active - Phase 1 (WMMA) in progress

**🎯 Current Goal**: Implement WMMA for Q·K^T and P·V, achieve <200 µs, TC util ≥50%**

