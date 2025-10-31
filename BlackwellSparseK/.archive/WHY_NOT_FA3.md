# Why Not FlashAttention-3?

**BlackwellSparseK Benchmark Philosophy**

---

## 🎯 **Our Position**

FlashAttention-3 (FA3) is a **dense** Hopper-optimized attention kernel that chases peak dense TFLOPs via asynchrony, warp specialization, and FP8. That's great for dense baselines—but it **doesn't exercise learnable sparsity** or KV paging, and therefore doesn't reflect SparseK's advantage on long-context robotics LLM workloads.

We still report FA3 for a **dense ceiling reference**, but our proof is against:

1. **PyTorch SDPA (dense floor)** - Production baseline
2. **xFormers Sparse (structured)** - Industry structured-sparse reference  
3. **vLLM PagedAttention (serving)** - End-to-end throughput

---

## 📊 **The Right Measuring Sticks**

| Baseline | Type | Purpose |
|----------|------|---------|
| **PyTorch SDPA** | Dense | Production floor - beat this to be viable |
| **xFormers Sparse** | Structured Sparse | Industry peer - beat this to show sparsity advantage |
| **vLLM PagedAttention** | Serving | End-to-end throughput - real-world deployment |
| ~~FlashAttention-3~~ | Dense (Hopper-tuned) | Dense ceiling reference only |

---

## 🔬 **Why FA3 Isn't the Right Target**

### **1. FA3 is Dense-Only**
- FA3 optimizes for **100% attention computation**
- SparseK uses **learnable sparse patterns** (e.g., 20-50% density)
- Comparing dense vs sparse is comparing apples to oranges

### **2. FA3 is Hopper-Specific**
- FA3 is tuned for H100's specific warp scheduler and TMA
- SparseK targets H100 + B200 + Rubin (multi-generation)
- FA3's techniques (warp specialization, async copy) are architecture-specific

### **3. FA3 Doesn't Exercise Sparse Patterns**
- SparseK's value proposition: **learn which tokens to attend to**
- FA3 computes all token pairs (full O(n²) complexity)
- SparseK: O(n²) → O(n × s) where s is sparsity ratio

### **4. Real Workloads Use Sparsity**
- Long-context LLMs: Most tokens don't need full attention
- Robotics: Attend to salient observations, not all history
- vLLM: KV cache paging naturally creates sparse patterns

---

## 📈 **Our Benchmark Tiers**

| Tier | Target | Comparator | Rationale |
|------|--------|------------|-----------|
| **T1** | ≤ 3.820 µs/head (H=96, L=4096, B=16) | **PyTorch SDPA** | Dense production floor |
| **T2** | < 3.0 µs/head | **xFormers Sparse** | Beat structured-sparse industry ref |
| **T3** | < 2.0 µs/head + faster policy convergence | **vLLM + SparseK** | End-to-end robotics LLM throughput |

**Target Configuration**:
- Batch: 16
- Heads: 96 (GPT-4 scale)
- Sequence: 4096 tokens
- Head Dim: 128

---

## 🚀 **What We Measure**

### **Micro-Benchmark** (`make bench`)
```bash
python3 benchmarks/perf.py --run micro
```

**Outputs**:
- PyTorch SDPA (dense floor): X.XX μs/iter
- xFormers Sparse (structured): X.XX μs/iter
- BlackwellSparseK (learnable): X.XX μs/iter

**Success**: SparseK < SDPA (beat dense floor)

### **Nsight Profiling** (`make bench-profile`)
```bash
make bench-profile
# Exports to: benchmarks/metrics.json
```

**Metrics**:
- SM throughput (% of peak)
- Warp active (% of peak)
- DRAM throughput (% of peak)
- L2 throughput (% of peak)

### **End-to-End** (vLLM serving)
```bash
# Run in separate process
python3 benchmarks/vllm_bench.py --sparsek
```

**Metrics**:
- Prompt tokens/second
- Generation tokens/second
- KV cache hit rate
- Total throughput (tokens/s)

---

## 📚 **References**

### **FlashAttention-3**
- **Blog**: [PyTorch - FlashAttention-3](https://pytorch.org/blog/flashattention-3/)
- **Published**: July 11, 2024
- **Focus**: Dense Hopper optimization, not sparse patterns

### **xFormers**
- **Repo**: [facebookresearch/xformers](https://github.com/facebookresearch/xformers)
- **Docs**: [xFormers Operators](https://facebookresearch.github.io/xformers/components/ops.html)
- **Sparse Support**: LowerTriangularMask, BlockDiagonalMask, AttentionBias

### **vLLM**
- **Repo**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Docs**: [vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics.html)
- **PagedAttention**: KV cache management for long contexts

### **SparseK Paper**
- **Title**: Efficient Sparse Attention for Long-Range Transformers
- **Authors**: Sun et al.
- **arXiv**: [2406.16747](https://arxiv.org/abs/2406.16747)
- **Key Idea**: Learn which tokens to attend to (adaptive sparsity)

---

## ✅ **Summary**

**FA3 is a fantastic dense kernel** - it's the state-of-the-art for 100% dense attention on Hopper.

**But SparseK's value is in sparsity** - learning which 20-50% of tokens to attend to while maintaining accuracy.

**Therefore**:
- ✅ Use PyTorch SDPA as floor (beat dense baseline)
- ✅ Use xFormers Sparse as peer (beat structured sparse)
- ✅ Use vLLM as end-to-end (prove serving value)
- ⚠️  Use FA3 as reference only (dense ceiling, not target)

---

**Benchmark Command**:
```bash
cd BlackwellSparseK
make bench           # Micro-benchmark vs SDPA/xFormers
make bench-profile   # Nsight metrics → JSON
```

**Target**: < 3.820 μs/head (beat SDPA) = **TIER 1 PASS**

---

**Last Updated**: October 30, 2025  
**By**: BlackwellSparseK Team

