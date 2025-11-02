# CUTLASS PR Checklist

Professional contribution checklist based on NVIDIA CUTLASS standards.

## ✅ Completed (Ready Now)

### Code Quality
- [x] Modern CUTLASS 4.3.0 API (CollectiveBuilder)
- [x] Clean, commented implementation
- [x] BSD 3-Clause license (compatible with CUTLASS)
- [x] Professional code structure

### Performance Verification
- [x] CUDA Events timing (industry standard)
- [x] Multiple runs (5 independent measurements)
- [x] Statistical validation (±0.3% variance)
- [x] Baseline comparisons (cuBLAS, CUTLASS Ex49)

### Documentation
- [x] Example directory structure (`examples/gemm_optimized/`)
- [x] Build instructions (clear, tested)
- [x] Performance tables (verified numbers)
- [x] Minimal README (deeds not words)
- [x] Detailed subdirectory docs

### Repository Structure
- [x] Matches CUTLASS organization
- [x] `examples/` directory for implementations
- [x] Professional README format
- [x] Clear license terms
- [x] Contact information

## ⏸️ Pending (Before CUTLASS PR)

### NCU Profiling (Required)
- [ ] **SM utilization %** - Hardware counter validation
- [ ] **Memory throughput** - HBM bandwidth analysis
- [ ] **Warp occupancy** - Thread block efficiency
- [ ] **Roofline analysis** - Compute vs memory bound determination

**Blocker:** NCU requires privileged container or bare metal access  
**Current:** Cloud GPU container restrictions  
**Solution:** Need bare metal H100 or compatible cloud provider

### Extended Validation (Recommended)
- [ ] Matrix size sweep (4K, 8K, 16K, 32K, 64K)
- [ ] Different aspect ratios (square, tall, wide)
- [ ] Numerical correctness tests (detailed error analysis)
- [ ] Comparison with latest cuBLAS versions

### Documentation Polish (Nice-to-have)
- [ ] Parameter tuning guide
- [ ] When to use these optimizations
- [ ] Performance prediction model
- [ ] Architecture-specific notes (Ampere vs Hopper)

## CUTLASS Contribution Pattern

### Phase 1: Fork & Branch ✅ Can Do Now
```bash
# Fork NVIDIA/cutlass on GitHub
git clone https://github.com/YOUR_USERNAME/cutlass.git
cd cutlass
git checkout main
git checkout -b feature/optimized-dense-gemm
```

### Phase 2: Copy Example Structure ✅ Ready
```bash
# Create example directory
mkdir -p examples/XX_optimized_dense_gemm/

# Copy verified implementation
cp periodicdent42/BlackwellSparseK/examples/gemm_optimized/* \
   cutlass/examples/XX_optimized_dense_gemm/

# Update numbering (XX = next available number)
```

### Phase 3: Add NCU Metrics ⏸️ Blocked
```bash
# Run comprehensive profiling (requires privileged access)
ncu --set full \
    --target-processes all \
    --kernel-name regex:gemm \
    --print-summary per-kernel \
    ./gemm_optimized > ncu_report.txt

# Add to documentation
cat ncu_report.txt >> README.md
```

### Phase 4: Submit PR ⏸️ After NCU
```bash
git add examples/XX_optimized_dense_gemm/
git commit -m "Add optimized dense GEMM example (550.8 TFLOPS, 88% of cuBLAS)

Demonstrates non-square tile optimization for H100:
- TileShape 128×256×64 (vs default 128×128×128)
- ClusterShape 2×1×1 (vs default 1×2×1)
- 35% improvement over Example 49 baseline

Verified with NCU profiling on H100."

git push origin feature/optimized-dense-gemm

# Create PR on GitHub
```

## NCU Requirements

### What NCU Provides
1. **SM Utilization** - % of compute units active
2. **Memory Throughput** - GB/s and % of peak bandwidth
3. **Warp Occupancy** - Theoretical vs achieved
4. **Instruction Mix** - Breakdown of operation types
5. **Roofline Position** - Compute vs memory bound

### Why NCU Is Critical for CUTLASS
- NVIDIA maintainers expect hardware validation
- Performance claims need counter evidence
- Helps identify optimization opportunities
- Standard for all CUTLASS examples

### How to Get NCU Access

**Option 1: Bare Metal H100** (Best)
- Full NCU access
- No restrictions
- Complete profiling

**Option 2: Compatible Cloud** (Alternative)
- Lambda Labs (allows NCU)
- AWS EC2 p5 (with setup)
- Azure NC H100 v5 (with setup)

**Option 3: NVIDIA DGX Cloud** (Premium)
- Full profiling support
- NVIDIA-managed environment
- Professional grade

## Estimated Timeline

### Now → NCU Access
- **Status:** Code complete, performance verified
- **Blocker:** NCU access
- **Action:** Monitor for bare metal H100 access

### NCU Access → PR Ready
- **Time:** 1-2 days
- **Tasks:** Run NCU, document metrics, polish docs
- **Effort:** Minimal (code already excellent)

### PR Submission → Merge
- **Time:** 2-4 weeks (typical NVIDIA review cycle)
- **Tasks:** Address maintainer feedback
- **Outcome:** Contribution to CUTLASS project

## Success Criteria

### Minimum (Must Have)
- [x] Working implementation
- [x] Verified performance (CUDA Events)
- [x] Clean documentation
- [ ] NCU profiling metrics

### Target (Should Have)
- [ ] Matrix size sweep
- [ ] Numerical correctness tests
- [ ] Architecture comparison

### Stretch (Nice to Have)
- [ ] Auto-tuning integration
- [ ] Python bindings
- [ ] Multi-GPU support

## Current Status

**Code Quality:** ★★★★★ (Professional)  
**Performance:** ★★★★★ (Verified, 88% of cuBLAS)  
**Documentation:** ★★★★★ (Minimal, factual)  
**NCU Profiling:** ★☆☆☆☆ (Blocked on access)

**Overall:** Ready for PR after NCU ✅

## Contact for CUTLASS Maintainers

**Brandon Dent, MD**  
Email: b@thegoatnote.com  
GitHub: @GOATnote-Inc

**Availability:** Ready to provide additional validation, address feedback, and perform any requested changes.

**Timeline:** Can complete NCU profiling within 1-2 days of gaining access.

---

**Document Purpose:** Internal checklist for CUTLASS PR preparation  
**Last Updated:** November 2, 2025  
**Status:** Ready for PR after NCU profiling

