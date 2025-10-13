# Phase 1: Profile-Driven Optimization Plan

**Date**: October 12, 2025  
**Objective**: Profile FA-1 kernel, identify top 3 bottlenecks, fix systematically  
**Target**: 0.5-1.0× vs PyTorch (portfolio-ready)  
**Timeline**: 12-17 hours ($60-85 @ $5/hr for profiling work)

---

## 📊 Current Baseline (FA-1)

**Performance**: 1.8 ms @ S=128 (measured in Session N+6)  
**vs PyTorch**: 0.05 ms (36× slower)  
**vs Expected**: Should be 0.5-1.0× on L4

**Hardware**: L4 GPU
- Compute: 30 TFLOPs (FP16)
- Memory: 300 GB/s bandwidth
- SMEM: 48 KB per SM
- Architecture: Ampere (SM89)

**Current Config**:
- Threads: 256 (8 warps)
- Tiles: 64×64×64 (M×N×K)
- SMEM usage: ~40 KB (within 48 KB limit)

---

## 🔬 Phase 1: Profiling (4-6 hours, $2.40)

### Step 1: Basic Profiling
```bash
# SSH to GPU
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a

# Setup environment
cd ~/periodicdent42/cudadent42
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/kiteboard/periodicdent42/cudadent42:$PYTHONPATH

# Run basic profile
ncu --set full --target-processes all \
    --kernel-name "flash_attention_forward_kernel" \
    -o profile_fa1_basic \
    python3 benches/bench_correctness_and_speed.py

# Generate reports
ncu -i profile_fa1_basic.ncu-rep --csv > profile_fa1.csv
ncu -i profile_fa1_basic.ncu-rep --page raw > profile_fa1_raw.txt
ncu -i profile_fa1_basic.ncu-rep --page details > profile_fa1_details.txt
```

### Step 2: Key Metrics to Capture

**Memory Metrics**:
```bash
ncu --metrics \
  dram__bytes_read.sum,dram__bytes_write.sum,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  -o profile_fa1_memory \
  python3 benches/bench_correctness_and_speed.py
```

**Compute Metrics**:
```bash
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
  smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio \
  -o profile_fa1_compute \
  python3 benches/bench_correctness_and_speed.py
```

### Step 3: Roofline Analysis
```bash
ncu --set roofline -o profile_fa1_roofline \
    python3 benches/bench_correctness_and_speed.py
```

---

## 📈 Expected Bottlenecks (Hypotheses to Test)

### Hypothesis 1: Memory Bound (Most Likely)
**Symptoms**:
- DRAM throughput < 70% of peak (300 GB/s)
- High memory stall cycles
- Low compute utilization

**Evidence to Look For**:
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` < 70%
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` > 0.5

**Likely Fixes**:
1. Non-coalesced memory access (stride issues)
2. Missing `__restrict__` on pointers
3. Redundant global memory reads

### Hypothesis 2: Shared Memory Bank Conflicts
**Symptoms**:
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` > 0
- Throughput drops during SGEMM-like loops

**Evidence to Look For**:
- Bank conflicts in smem_Q, smem_K, smem_V access
- Conflicts during S matrix writes

**Likely Fixes**:
1. Add padding: `smem_Q[TILE_SIZE_M][TILE_SIZE_K + 1]`
2. Use XOR swizzling for 2D arrays
3. Reorder access patterns

### Hypothesis 3: Low Occupancy
**Symptoms**:
- `sm__warps_active.avg.pct_of_peak_sustained_active` < 50%
- Register pressure (too many registers per thread)
- SMEM pressure (too much shared memory)

**Evidence to Look For**:
- Occupancy calculator shows <50% theoretical occupancy
- Register spilling to local memory

**Likely Fixes**:
1. Reduce tile sizes (64→32)
2. Reduce register usage (fewer loop unrolls)
3. Launch more blocks (reduce block size)

---

## 🛠️ Phase 2: Fix Top 3 Bottlenecks (6-8 hours, $3.60-4.80)

### Fix Template (Repeat for Each Bottleneck)

**1. Identify**:
- Metric: [name] = [current value]
- Target: [expected value]
- Gap: [percentage]

**2. Hypothesize**:
- Root cause: [explanation]
- Evidence: [profiler metric]

**3. Implement**:
```cpp
// BEFORE
[old code]

// AFTER
[new code with comment explaining fix]
```

**4. Validate**:
- Re-profile: [metric] = [new value]
- Speedup: [percentage improvement]
- Correctness: [test result]

**5. Document**:
```markdown
| Fix | Metric Improved | Before | After | Speedup |
|-----|-----------------|--------|-------|---------|
| [name] | [metric] | [value] | [value] | [%] |
```

---

## 📊 Phase 3: Ablation Study (2-3 hours, $1.20-1.80)

### Goal: Show Causal Relationship

Create a table showing incremental improvements:

| Version | Speedup vs PyTorch | Memory BW | Occupancy | Key Change |
|---------|-------------------|-----------|-----------|------------|
| Baseline (Session N+6) | 0.027× | ? | ? | Original |
| +restrict | ? | ? | ? | Added `__restrict__` to all pointers |
| +coalescing | ? | ? | ? | Fixed Q/K/V load patterns |
| +no-bank-conflicts | ? | ? | ? | Padded shared memory arrays |
| +warp-specialization | ? | ? | ? | Separate warps for load vs compute |
| **Final** | **0.5-1.0×** | **>200 GB/s** | **>50%** | **All fixes combined** |

### Documentation for Portfolio

Create `cudadent42/OPTIMIZATION_REPORT.md`:
```markdown
# Flash Attention Optimization Case Study

## Objective
Optimize Flash Attention kernel on L4 GPU to match PyTorch SDPA performance.

## Initial State
- Performance: 1.8 ms @ S=128 (0.027× vs PyTorch)
- Hardware: NVIDIA L4 (300 GB/s, 30 TFLOPs FP16)

## Methodology
1. Profile with Nsight Compute to identify bottlenecks
2. Fix top 3 issues incrementally
3. Validate each fix with profiler metrics
4. Document causal relationship

## Bottlenecks Identified
[From profiling session - fill in after Phase 1]

## Fixes Applied
[Ablation table - fill in after Phase 2]

## Final Results
- Performance: X ms @ S=128 (Y× vs PyTorch)
- Memory bandwidth: X GB/s (Y% of peak)
- Occupancy: X% (Y% of theoretical)

## Key Learnings
[What worked, what didn't, why]

## References
- Nsight Compute reports: `artifacts/profiles/`
- Code: `cudadent42/python/flashmoe_science/csrc/flash_attention_science.cu`
- Tests: `cudadent42/benches/bench_correctness_and_speed.py`
```

---

## 🎯 Success Criteria

### Minimum Viable Portfolio Piece (MVP)
✅ Nsight Compute profile captured and analyzed  
✅ Top 3 bottlenecks identified with evidence  
✅ At least 2/3 bottlenecks fixed  
✅ Speedup: 0.5-1.0× vs PyTorch (10-20× improvement from baseline)  
✅ Documentation: Optimization report with ablation table  
✅ Tests: All correctness tests passing  

### Stretch Goals
⭐ All 3 bottlenecks fixed  
⭐ Speedup: 0.8-1.0× vs PyTorch  
⭐ Comparison to flash-attn reference implementation  
⭐ Blog post or GitHub README showcasing methodology  

---

## 💰 Budget

| Phase | Time | GPU Cost | Engineer Cost | Total |
|-------|------|----------|---------------|-------|
| Profiling | 4-6 hours | $1.20-1.80 | $20-30 | $21.20-31.80 |
| Fixing | 6-8 hours | $1.80-2.40 | $30-40 | $31.80-42.40 |
| Ablation | 2-3 hours | $0.60-0.90 | $10-15 | $10.60-15.90 |
| **Total** | **12-17 hours** | **$3.60-5.10** | **$60-85** | **$63.60-90.10** |

**vs Split-K**: $90 vs $1,304 (93% savings) ✅  
**vs Expected ROI**: Portfolio-ready piece for <$100 ✅

---

## 🔧 Tools & References

### Nsight Compute
- Docs: https://docs.nvidia.com/nsight-compute/
- Profiling guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- Metric reference: https://docs.nvidia.com/nsight-compute/NsightComputeCli/

### L4 GPU Specs
- Datasheet: https://resources.nvidia.com/en-us-l4-datasheet
- Compute: 30 TFLOPs (FP16), 15 TFLOPs (FP32)
- Memory: 300 GB/s bandwidth, 24 GB VRAM
- SMEM: 48 KB per SM (static limit)

### Flash Attention Reference
- Paper: https://arxiv.org/abs/2205.14135
- Code: https://github.com/Dao-AILab/flash-attention
- L4 optimizations: Check `csrc/flash_attn/src/` for SM89 tuning

### CUDA Performance Guidelines
- Occupancy calculator: https://docs.nvidia.com/cuda/cuda-occupancy-calculator/
- Best practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Memory coalescing: https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/

---

## 📝 Session Logs

### Session 1: Initial Profiling (Planned)
**Date**: TBD  
**Goal**: Capture baseline metrics  
**Duration**: 2-3 hours  
**Cost**: $10-15  

**Checklist**:
- [ ] Start GPU
- [ ] Run `ncu --set full` profile
- [ ] Generate CSV/TXT reports
- [ ] Identify top 3 bottlenecks
- [ ] Document findings in this file
- [ ] Stop GPU

### Session 2: Fix Bottleneck #1 (Planned)
**Date**: TBD  
**Goal**: Fix highest-impact issue  
**Duration**: 2-3 hours  
**Cost**: $10-15  

**Checklist**:
- [ ] Implement fix
- [ ] Re-profile
- [ ] Validate speedup
- [ ] Document in ablation table
- [ ] Commit changes

### Session 3-4: Fix Bottlenecks #2-3 (Planned)
[Repeat pattern for remaining bottlenecks]

### Session 5: Ablation Study & Documentation (Planned)
**Date**: TBD  
**Goal**: Create portfolio piece  
**Duration**: 2-3 hours  
**Cost**: $10-15  

**Checklist**:
- [ ] Create ablation table
- [ ] Write optimization report
- [ ] Update README with results
- [ ] Create before/after comparison
- [ ] Push to GitHub

---

## 🚀 Getting Started

### Immediate Next Steps (30 minutes)

1. **Start GPU**:
```bash
gcloud compute instances start cudadent42-l4-dev --zone=us-central1-a
```

2. **Verify Nsight Compute installed**:
```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a --command="ncu --version"
```

3. **Run quick profile** (5 iterations):
```bash
gcloud compute ssh cudadent42-l4-dev --zone=us-central1-a --command="
  cd ~/periodicdent42/cudadent42 && \
  export PATH=/usr/local/cuda/bin:\$PATH && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH && \
  export PYTHONPATH=/home/kiteboard/periodicdent42/cudadent42:\$PYTHONPATH && \
  ncu --set full --target-processes all -o profile_quick python3 benches/bench_correctness_and_speed.py
"
```

4. **Download profile**:
```bash
gcloud compute scp cudadent42-l4-dev:~/periodicdent42/cudadent42/profile_quick.ncu-rep ./artifacts/profiles/ --zone=us-central1-a
```

5. **Analyze locally** (if you have Nsight Compute GUI):
```bash
ncu-ui ./artifacts/profiles/profile_quick.ncu-rep
```

---

**Status**: 📋 **READY TO START**  
**Next Action**: Start GPU and run initial profile  
**Expected Time to Portfolio-Ready**: 12-17 hours  
**Expected Cost**: $63.60-90.10

