#!/bin/bash
# Nsight Compute Profiling Script for V3 Kernel
# Captures detailed performance metrics and generates bottleneck analysis

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================================
# Configuration
# ============================================================================

# Output directory
DATE_STR=$(date +%Y-%m-%d)
OUTPUT_DIR="${REPO_ROOT}/benchmarks/l4/${DATE_STR}/nsight_compute"
mkdir -p "${OUTPUT_DIR}"

# Kernel name pattern to profile
KERNEL_PATTERN="flash_attention_s512_v3"

# Shapes to profile (canonical + V3 specialized)
declare -a SHAPES=(
    "B=1,H=8,S=512,D=64,causal=False"    # v3_small
    "B=4,H=16,S=512,D=64,causal=True"    # v3_medium_causal
    "B=8,H=16,S=512,D=64,causal=False"   # v3_large
)

# Nsight Compute metrics to collect
# Full set: warp occupancy, SM busy, memory, bank conflicts, etc.
NCU_METRICS="smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,\
smsp__inst_executed.sum,\
dram__bytes.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,\
l1tex__data_bank_conflicts_pipe_lsu.sum,\
smsp__sass_branch_targets_threads_divergent.sum"

echo "================================================================================"
echo "Nsight Compute Profiling - V3 Kernel Bottleneck Analysis"
echo "================================================================================"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output: ${OUTPUT_DIR}"
echo "Kernel Pattern: ${KERNEL_PATTERN}"
echo "Shapes: ${#SHAPES[@]}"
echo "================================================================================"

# ============================================================================
# Helper: Run profiling for a single shape
# ============================================================================

profile_shape() {
    local shape_config=$1
    local shape_name=$(echo "${shape_config}" | tr '=,' '__')
    
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Profiling: ${shape_config}"
    echo "--------------------------------------------------------------------------------"
    
    # Parse shape config
    local B=$(echo "${shape_config}" | grep -oP 'B=\K\d+')
    local H=$(echo "${shape_config}" | grep -oP 'H=\K\d+')
    local S=$(echo "${shape_config}" | grep -oP 'S=\K\d+')
    local D=$(echo "${shape_config}" | grep -oP 'D=\K\d+')
    local causal=$(echo "${shape_config}" | grep -oP 'causal=\K\w+')
    
    # Create subdirectory for this shape
    local shape_dir="${OUTPUT_DIR}/${shape_name}"
    mkdir -p "${shape_dir}"
    
    # Create Python script to run kernel once
    local python_script="${shape_dir}/run_kernel.py"
    cat > "${python_script}" <<PYEOF
import torch
from cudadent42.bench.fa_s512_v3 import flash_attention_s512_v3_forward

# Shape config
B, H, S, D = ${B}, ${H}, ${S}, ${D}
causal = ${causal}

# Create inputs
Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(5):
    flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)

# Profile run
torch.cuda.synchronize()
output = flash_attention_s512_v3_forward(Q, K, V, is_causal=causal, config_id=1)
torch.cuda.synchronize()

print(f"✓ Kernel executed: B={B}, H={H}, S={S}, D={D}, causal={causal}")
PYEOF
    
    # Run NCU profiling
    local qdrep_file="${shape_dir}/profile.ncu-rep"
    local txt_file="${shape_dir}/profile.txt"
    local csv_file="${shape_dir}/profile.csv"
    
    echo "Running ncu..."
    ncu \
        --set full \
        --target-processes all \
        --kernel-name-base function \
        --kernel-regex "${KERNEL_PATTERN}" \
        --export "${qdrep_file%.ncu-rep}" \
        --log-file "${txt_file}" \
        --csv \
        --page details \
        python3 "${python_script}" > "${csv_file}" 2>&1 || true
    
    # Check if profiling succeeded
    if [ -f "${qdrep_file}" ]; then
        echo "✓ Profile captured: ${qdrep_file}"
        
        # Generate text summary
        ncu --import "${qdrep_file}" --page details > "${shape_dir}/summary.txt" 2>&1 || true
        echo "✓ Summary: ${shape_dir}/summary.txt"
    else
        echo "✗ Profiling failed for ${shape_config}"
    fi
}

# ============================================================================
# Main: Profile all shapes
# ============================================================================

for shape in "${SHAPES[@]}"; do
    profile_shape "${shape}"
done

# ============================================================================
# Generate bottleneck analysis
# ============================================================================

echo ""
echo "================================================================================"
echo "Generating Bottleneck Analysis"
echo "================================================================================"

ANALYSIS_FILE="${OUTPUT_DIR}/bottleneck_analysis.md"

cat > "${ANALYSIS_FILE}" <<'MDEOF'
# Nsight Compute Bottleneck Analysis

**Date**: $(date '+%Y-%m-%d %H:%M:%S')  
**Kernel**: flash_attention_s512_v3  
**GPU**: NVIDIA L4 (sm_89)

---

## Profiled Shapes

MDEOF

# List profiled shapes
for shape in "${SHAPES[@]}"; do
    shape_name=$(echo "${shape}" | tr '=,' '__')
    shape_dir="${OUTPUT_DIR}/${shape_name}"
    
    if [ -f "${shape_dir}/profile.ncu-rep" ]; then
        echo "- ✓ ${shape}" >> "${ANALYSIS_FILE}"
    else
        echo "- ✗ ${shape} (profiling failed)" >> "${ANALYSIS_FILE}"
    fi
done

cat >> "${ANALYSIS_FILE}" <<'MDEOF'

---

## Key Metrics to Analyze

When reviewing `.ncu-rep` files in Nsight Compute UI, focus on:

### 1. **SM Utilization**
- **Metric**: `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed`
- **Target**: ≥70% for compute-bound kernels
- **If Low**: Indicates under-occupancy or memory bottleneck

### 2. **Warp Occupancy**
- **Metric**: `sm__warps_active.avg.pct_of_peak_sustained_active`
- **Target**: ≥50% for good parallelism
- **If Low**: Check register usage, shared memory allocation, block size

### 3. **Memory Throughput**
- **DRAM**: `dram__bytes.sum`
- **L2**: `lts__t_bytes.sum`
- **L1**: `l1tex__t_bytes.sum`
- **If DRAM-bound**: Optimize data reuse, use shared memory staging

### 4. **Memory Coalescing**
- **Metric**: `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`
- **Target**: ≥80% (good coalescing)
- **If Low**: Fix memory access patterns (stride-1 access, aligned loads)

### 5. **Shared Memory Bank Conflicts**
- **Metric**: `l1tex__data_bank_conflicts_pipe_lsu.sum`
- **Target**: 0 or near-zero
- **If High**: Pad shared memory, change access patterns

### 6. **Branch Divergence**
- **Metric**: `smsp__sass_branch_targets_threads_divergent.sum`
- **Target**: Minimize (< 10% of total branches)
- **If High**: Reduce conditional branches, use predication

### 7. **Compute Throughput**
- **FP16**: `smsp__sass_thread_inst_executed_op_hadd_pred_on.sum`
- **FP32**: `smsp__sass_thread_inst_executed_op_dadd_pred_on.sum`
- **If Low**: Not compute-bound, likely memory-bound

---

## Common Bottlenecks & Fixes

| Symptom | Root Cause | Fix |
|---------|------------|-----|
| **Low SM busy (<30%)** | Under-occupancy | Increase block size, reduce register usage, reduce SMEM per block |
| **High DRAM traffic** | Poor data reuse | Use shared memory, tile data, prefetch with cp.async |
| **Low coalescing (<50%)** | Strided access | Transpose data layout, use vectorized loads (ld.global.v4) |
| **High bank conflicts** | Shared memory stride | Pad SMEM arrays, change access patterns |
| **High branch divergence** | Conditional branches in warps | Use predication, uniform control flow |
| **Long instruction replay** | RAW hazards, dependency stalls | Reorder instructions, increase ILP |

---

## Prioritized Hypotheses (To Be Updated After Review)

1. **[PLACEHOLDER]** - Review `.ncu-rep` files and update with actual findings
2. **[PLACEHOLDER]** - List top 3 bottlenecks in priority order
3. **[PLACEHOLDER]** - Propose concrete fixes for each

**Action**: Open `.ncu-rep` files in Nsight Compute GUI and update this section.

---

## Next Steps

1. **Review** `.ncu-rep` files in Nsight Compute UI
2. **Identify** top 3 bottlenecks from metrics above
3. **Update** this document with concrete findings
4. **Implement** fixes (Phase 6: Inversion thinking, Phase 7: Expert polish)
5. **Re-profile** to validate improvements

MDEOF

echo "✓ Analysis template: ${ANALYSIS_FILE}"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "================================================================================"
echo "✅ Phase 5: Nsight Compute Profiling Complete"
echo "================================================================================"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Files generated:"
find "${OUTPUT_DIR}" -type f \( -name "*.ncu-rep" -o -name "*.txt" -o -name "*.md" \) -exec echo "  {}" \; 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. Download .ncu-rep files from GPU:"
echo "     gcloud compute scp --recurse cudadent42-l4-dev:~/periodicdent42/benchmarks/l4/${DATE_STR}/nsight_compute/ ."
echo "  2. Open in Nsight Compute GUI (requires NVIDIA Nsight Compute installed locally)"
echo "  3. Update bottleneck_analysis.md with findings"
echo "  4. Proceed to Phase 6: Inversion thinking based on identified bottlenecks"
echo "================================================================================"

exit 0

