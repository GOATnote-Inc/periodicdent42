# Priority 2: TILE_M=64 Implementation - In Progress

**Start Time**: Session continues (GPU active)  
**Target**: 1.5-2× additional speedup (0.3184 ms → 0.18 ms)  
**Approach**: Increase TILE_M from 32 to 64

---

## Implementation Checklist

### 1. Kernel Configuration ⏳
- [ ] Change TILE_M: 32 → 64
- [ ] Change NUM_WARPS: 4 → 8 (256 threads for better occupancy)
- [ ] Update SMEM calculations (verify <48KB)

### 2. Warp Work Distribution ⏳
- [ ] Update compute_QK_wmma (4 M-blocks instead of 2)
- [ ] Update compute_SV_wmma (4 M-blocks instead of 2)
- [ ] Adjust temp_O allocation (64×64 instead of 32×64)

### 3. Testing ⏳
- [ ] Compile V3 kernel
- [ ] Run 7 correctness tests
- [ ] Measure performance (100 iterations)

### 4. Documentation ⏳
- [ ] Update PRIORITY2_RESULTS with findings
- [ ] Commit V3 kernel
- [ ] Update TODO list

---

## Expected Outcome

**Best Case** (2× speedup):
- V3: 0.16 ms (3.15× vs V1)
- 0.44× vs SDPA (still 2.3× slower)

**Realistic** (1.75× speedup):
- V3: 0.18 ms (2.80× vs V1)
- 0.40× vs SDPA (still 2.5× slower)

**Minimum** (1.5× speedup):
- V3: 0.21 ms (2.40× vs V1)
- 0.35× vs SDPA (still 2.9× slower)

---

**Status**: Starting implementation now...

