# EvoEngineer SDPA Optimization

**Mission**: Achieve **< 5 μs** for fused SDPA (B=1, H=8, S=512, D=64) on L4 (sm_89)

**Current baseline**: PyTorch SDPA @ 25.94 μs → Need **5.2× speedup**

---

## Repository Layout

```
evo-sdpa/
  ├── 00_task.md                 # task context (I1)
  ├── 01_generate.md             # EvoEngineer-Free generator (exploration, low tokens)
  ├── 02_refine_with_insights.md # EvoEngineer-Insight (harvest + apply I3)
  ├── 03_elite_loop.md           # EvoEngineer-Full (I2 + I3 + elite population)
  ├── kernels/
  │    ├── sdpa_fused.cu         # candidate kernel (fwd)
  │    └── runtime.hpp           # launchers, param structs
  ├── bench/
  │    └── bench_sdpa.py         # compile+correctness+perf harness
  ├── prompts/
  │    └── snippets.md           # reusable prompt blocks (PTX, cp.async, etc.)
  ├── nsight/
  │    └── metrics.txt           # NCU metric set
  └── README.md                  # this file
```

---

## Workflow

### Phase 1: Exploration (EvoEngineer-Free)
1. Use `01_generate.md` to generate diverse initial kernels
2. Run `bench/bench_sdpa.py` for each candidate
3. Keep any that pass correctness and beat PyTorch

### Phase 2: Refinement (EvoEngineer-Insight)
1. Profile best kernels with NCU (`nsight/metrics.txt`)
2. Extract optimization insights
3. Use `02_refine_with_insights.md` to apply insights

### Phase 3: Elite Loop (EvoEngineer-Full)
1. Maintain Top-K performers (K=3-5)
2. Use `03_elite_loop.md` to propose children
3. Select strictly on measured perf + correctness
4. Iterate until target achieved

---

## Quick Start

```bash
cd evo-sdpa
python bench/bench_sdpa.py
```

---

## References

- **EvoEngineer Paper**: arXiv:2510.03760v1 [cs.LG] 04 Oct 2025
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691
- **CUTLASS**: https://github.com/NVIDIA/cutlass

---

**Status**: Phase 1 (Exploration) - Ready to generate first candidate

