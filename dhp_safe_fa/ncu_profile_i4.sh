#!/bin/bash
# NCU profiling for I4 kernel (burn methodology)
# This will identify performance bottlenecks

echo "🔬 NCU Profiling: DHP I4 Kernel"
echo "================================"
echo ""

# Check NCU availability
if ! command -v ncu &> /dev/null; then
    echo "❌ Error: ncu not found. Install Nsight Compute."
    exit 1
fi

# Create profiling script
cat > /tmp/profile_i4.py << 'EOF'
import torch
import dhp_i4_kernel

B, H, S, D = 4, 16, 1024, 64

torch.manual_seed(42)
Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')

scale = 1.0 / (D ** 0.5)
scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
scores_flat = scores.reshape(B*H, S, S)
V_flat = V.reshape(B*H, S, D)

# Warmup
for _ in range(10):
    _ = dhp_i4_kernel.forward(scores_flat, V_flat, S, S)
torch.cuda.synchronize()

# Profile this
out = dhp_i4_kernel.forward(scores_flat, V_flat, S, S)
torch.cuda.synchronize()
EOF

# Run NCU with burn-style metrics
echo "Running NCU (this will take ~30 seconds)..."
echo ""

sudo ncu \
    --metrics smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
launch__registers_per_thread,\
launch__occupancy_limit_blocks,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    --target-processes all \
    python3 /tmp/profile_i4.py 2>&1 | \
    grep -E "(gpu__time|sm__throughput|dram__|launch__|l1tex__|smsp__sass)" | \
    head -20

echo ""
echo "✅ NCU Profile Complete"

