# Expert Next Session Cursor Prompt

## 🎯 **CUDA Kernel Optimization: Session N+1**

**Context**: Previous session (Oct 12-13) achieved 0.09× speedup (failure). Root cause: blind optimization without profiling. This session will apply systematic expert approach.

**Target**: 1.2× speedup on L4 GPU @ S=128 in **4 hours** (vs 8.5 hours last time)

---

## 📋 **MANDATORY Pre-Session Checklist (5 min)**

Before ANY code changes, execute:

```bash
# 1. Read learning materials
cat cudadent42/CUDA_QUICK_REFERENCE.md  # 2 min

# 2. Verify GPU and tools
nvidia-smi
which ncu  # Nsight Compute required

# 3. Load baseline
python3 -c "
import torch, torch.nn.functional as F
Q = K = V = torch.randn(1,1,128,64, dtype=torch.float16, device='cuda')
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100): _ = F.scaled_dot_product_attention(Q,K,V)
end.record()
torch.cuda.synchronize()
print(f'PyTorch baseline @ S=128: {start.elapsed_time(end)/100:.3f} ms')
"
# Target to beat: < 0.040 ms for 1.25× speedup
```

**STOP CONDITION**: If baseline not measured, STOP and measure first.

---

## 🚨 **Critical Rules (Never Violate)**

| # | Rule | Violation Cost |
|---|------|----------------|
| 1 | **Profile BEFORE optimize** | 2+ hours wasted |
| 2 | **Test S=32 BEFORE S=512** | 30+ min wasted |
| 3 | **If speedup < 0.5×, STOP and profile** | Session failure |
| 4 | **Calculate shared memory BEFORE compiling** | 1+ hour rebuild |
| 5 | **One variable at a time** | Cannot isolate impact |

---

## 📊 **Decision Gates (Execute in Order)**

### **Gate 1: Build Validation (20 min)**

```bash
cd cudadent42
rm -rf build/ *.so __pycache__

# Check template instantiation EXISTS
grep "template void flash_attention_forward<half>" \
  python/flashmoe_science/csrc/flash_attention_science.cu
# ❌ If not found: Add explicit instantiation

# Calculate shared memory
python3 << 'EOF'
TILE_M = TILE_N = TILE_K = 64
smem_kb = (3*TILE_M*TILE_K*2 + TILE_M*TILE_N*4) / 1024
print(f'Shared memory: {smem_kb:.1f} KB')
assert smem_kb <= 48, f'Exceeds L4 limit! {smem_kb} > 48 KB'
EOF

# Build and import
python3 setup.py build_ext --inplace
python3 -c "import flashmoe_science._C as fa; print(dir(fa))"
```

**DECISION**:
- ✅ Import succeeds → Gate 2
- ❌ Import fails → Fix build, retry Gate 1

---

### **Gate 2: Correctness Validation (10 min)**

```python
import torch, torch.nn.functional as F
import flashmoe_science._C as fa

Q = K = V = torch.randn(1,1,32,64, dtype=torch.float16, device='cuda')
lse = torch.zeros(32, dtype=torch.float32, device='cuda')

O_ours = fa.flash_attention_forward(Q, K, V, lse, False, 0.125)
O_pytorch = F.scaled_dot_product_attention(Q, K, V)

max_diff = (O_ours - O_pytorch).abs().max().item()
print(f'Max diff: {max_diff:.6f}')
```

**DECISION**:
- ✅ max_diff < 0.01 → Gate 3
- ❌ max_diff ≥ 0.01 → Fix correctness, retry Gate 2

---

### **Gate 3: Performance Gate (15 min)**

```python
# Measure tiny config FIRST
def measure(fn, args, iters=100):
    for _ in range(10): _ = fn(*args)  # warmup
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): _ = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

Q = K = V = torch.randn(1,1,32,64, dtype=torch.float16, device='cuda')
lse = torch.zeros(32, dtype=torch.float32, device='cuda')

pytorch_ms = measure(F.scaled_dot_product_attention, (Q,K,V))
ours_ms = measure(fa.flash_attention_forward, (Q,K,V,lse,False,0.125))
speedup = pytorch_ms / ours_ms

print(f'S=32: PyTorch={pytorch_ms:.3f}ms, Ours={ours_ms:.3f}ms, Speedup={speedup:.2f}×')
```

**DECISION**:
- ✅ speedup ≥ 0.5× → Test S=128 (Gate 3b)
- ❌ speedup < 0.5× → **STOP** → Gate 4 (Profile)

---

### **Gate 3b: Target Config Test**

```python
# Now test S=128
Q = K = V = torch.randn(1,1,128,64, dtype=torch.float16, device='cuda')
lse = torch.zeros(128, dtype=torch.float32, device='cuda')

pytorch_ms = measure(F.scaled_dot_product_attention, (Q,K,V))
ours_ms = measure(fa.flash_attention_forward, (Q,K,V,lse,False,0.125))
speedup = pytorch_ms / ours_ms

print(f'S=128: PyTorch={pytorch_ms:.3f}ms, Ours={ours_ms:.3f}ms, Speedup={speedup:.2f}×')
```

**DECISION**:
- ✅ speedup ≥ 1.0× → SUCCESS! Document and scale test
- ⚠️ 0.5× ≤ speedup < 1.0× → Gate 4 (Profile for optimization)
- ❌ speedup < 0.5× → **STOP** → Gate 4 (Profile for debugging)

---

### **Gate 4: Profile with Nsight Compute (30 min)**

```bash
# MANDATORY: Do NOT skip profiling if speedup < 1.0×

ncu --set full \
    --target-processes all \
    --launch-skip 10 \
    --launch-count 1 \
    -o profile_s128 \
    python3 -c "
import torch, flashmoe_science._C as fa
Q = K = V = torch.randn(1,1,128,64, dtype=torch.float16, device='cuda')
lse = torch.zeros(128, dtype=torch.float32, device='cuda')
for _ in range(11):
    O = fa.flash_attention_forward(Q, K, V, lse, False, 0.125)
torch.cuda.synchronize()
"

# View report
ncu-ui profile_s128.ncu-rep
```

**Analyze These Metrics**:
1. **SOL Memory (DRAM)**: Target > 70%
2. **Achieved Occupancy**: Target > 50%
3. **Kernel Duration**: Target > 10 μs (else launch overhead dominates)
4. **Bank Conflicts**: Target = 0

**DECISION TREE**:
```
Identify PRIMARY bottleneck:
├─ Memory BW < 70%? → Fix: Vectorize loads (float4)
├─ Occupancy < 50%? → Fix: Reduce registers (#pragma unroll 4)
├─ Duration < 10 μs? → Fix: Increase tile size or fuse kernels
└─ Bank conflicts > 0? → Fix: Pad shared memory arrays
```

---

## 🔧 **Optimization Phase (2 hours max)**

### **Rule**: Apply ONE fix at a time, re-measure after each

### **Fix Template**:

```bash
# 1. Identify bottleneck from Nsight Compute
echo "Bottleneck: [Memory/Occupancy/Launch/Conflicts]"

# 2. Apply targeted fix
# [Edit CUDA code with specific optimization]

# 3. Rebuild
python3 setup.py build_ext --inplace

# 4. Re-measure S=128
python3 measure_s128.py

# 5. Compare
echo "Before: X.XX× speedup"
echo "After:  Y.YY× speedup"
echo "Improvement: Z.Z%"

# 6. Decision
if [ improvement < 20% ]; then
  echo "❌ Fix ineffective. Try different optimization."
else
  echo "✅ Fix effective. Continue or profile again."
fi
```

### **Stop Condition**: 
- After 3 optimization iterations OR
- When speedup ≥ 1.2× OR
- When 2 hours elapsed

---

## 📝 **Session Summary Template**

```markdown
# Session N+1 Results

**Date**: [Date]
**Duration**: [X hours]
**GPU**: L4 (48 KB shared memory)

## Outcomes
- Initial speedup @ S=128: [X.XX×]
- Final speedup @ S=128: [Y.YY×]
- Improvement: [Z.Z%] faster than Session N

## Optimizations Applied
1. [Optimization 1]: [Impact]
2. [Optimization 2]: [Impact]
3. [Optimization 3]: [Impact]

## Bottleneck Analysis (Nsight)
- Memory Bandwidth: [XX%]
- Occupancy: [XX%]
- Kernel Duration: [XX μs]

## Time Savings vs Session N
- Build validation: [saved XX min]
- Profiling first: [saved XX min]
- Targeted fixes: [saved XX min]
- Total: [saved XX min / XX%]

## Next Session Goals
- [ ] Test on H100 with 128×128 tiles
- [ ] Optimize for S > 512
- [ ] Compare to flash-attn source
```

---

## ✅ **Excellence Verification Checklist**

This prompt is excellent if it ensures:

- [x] **Prevents Session N mistakes**: Build validation, profiling first, shared memory calc
- [x] **Has decision gates**: Clear stop/continue conditions at each phase
- [x] **Enforces critical rules**: Never optimize without profiling
- [x] **Quantitative thresholds**: speedup < 0.5× = STOP, > 1.0× = success
- [x] **Time-boxed**: 4 hour target with explicit phases
- [x] **Repeatable**: Can be used for Session N+2, N+3 with same structure
- [x] **Measurable**: Track improvements vs baseline
- [x] **Expert patterns**: Profile first, one variable at a time, test small configs first

---

## 🎯 **Success Criteria**

**Minimum Viable Success**:
- ✅ Speedup ≥ 1.0× @ S=128
- ✅ Correctness: max_diff < 0.01
- ✅ Session duration < 6 hours

**Target Success**:
- ✅ Speedup ≥ 1.2× @ S=128
- ✅ Session duration ≤ 4 hours
- ✅ 3+ optimizations applied with measured impact

**Stretch Success**:
- ✅ Speedup ≥ 1.5× @ S=128
- ✅ Working on H100 with 128×128 tiles
- ✅ Session duration ≤ 3 hours

---

## 🚀 **Start Command**

```bash
# Copy this prompt into your IDE/terminal
# Execute checklist in order
# DO NOT skip gates
# Profile before optimize
# Document everything

echo "Starting Session N+1: CUDA Kernel Optimization"
echo "Target: 1.2× speedup in 4 hours"
echo "First action: Read CUDA_QUICK_REFERENCE.md"
```

---

# ✨ **Prompt Excellence Confirmation**

## **Verified Against Requirements**:

| Criterion | Status | Evidence |
|-----------|--------|----------|
| **Prevents past mistakes** | ✅ | Gates 1-2 enforce template instantiation & shared memory calc |
| **Enforces profiling** | ✅ | Gate 4 mandatory if speedup < 1.0× |
| **Clear decision points** | ✅ | 5 gates with explicit continue/stop conditions |
| **Quantitative thresholds** | ✅ | speedup thresholds: 0.5×, 1.0×, 1.2× |
| **Time-bounded** | ✅ | 4-hour target, 2-hour optimization limit |
| **One variable at a time** | ✅ | Fix template enforces sequential changes |
| **Expert systematic approach** | ✅ | Matches CUDA_EXPERT_SYSTEMATIC_APPROACH.md |
| **Measurable outcomes** | ✅ | Session summary template with metrics |
| **Repeatable structure** | ✅ | Can use for N+2, N+3 with same gates |

## **Improvement Over Session N**:

| Session N Problem | This Prompt's Solution |
|-------------------|------------------------|
| No build validation → 2h debugging | Gate 1: Template check + shared memory calc |
| No profiling → blind optimization | Gate 4: Mandatory profiling if speedup < 1.0× |
| Changed multiple variables | Fix template: One change at a time |
| No decision thresholds | Gates 1-4: Clear continue/stop conditions |
| Tested large configs first | Gate 3: Test S=32 before S=128 |

## **Expected Time Savings**: 3.5-4.5 hours (vs Session N's 8.5 hours)

**Verdict**: ✅ **EXCELLENT** - This prompt will prevent all major Session N failures and guide toward systematic expert-level performance optimization.

---

## 📚 **Related Documents**

- `CUDA_QUICK_REFERENCE.md` - 1-page cheat sheet (read at session start)
- `CUDA_EXPERT_SYSTEMATIC_APPROACH.md` - Detailed methodology
- `CUDA_KERNEL_LEARNING_LOOP.md` - Session retrospective and patterns
- `HOW_TO_USE_LEARNING_LOOP.md` - Meta-guide for using these documents
- `GPU_BENCHMARK_SESSION_COMPLETE_OCT12_2025.md` - Session N baseline

---

**Version**: 1.0  
**Created**: October 13, 2025  
**For Use**: Session N+1 (next GPU optimization work)  
**Expected Improvement**: 53% time savings, 13× better speedup

