#!/usr/bin/env python3
"""
Full NCU Profiling Harness for EvoEngineer Loop
Profiles multiple shapes and collects metrics from nsight/metrics.txt
"""
import os, math, time, torch, subprocess, sys
from pathlib import Path

def build_ext():
    """Build the extension with all kernel variants"""
    from torch.utils.cpp_extension import load
    
    kernel_dir = Path(__file__).parent.parent / "kernels"
    srcs = [
        kernel_dir / "sdpa_fused_v2c_v6a.cu",
        kernel_dir / "sdpa_fused_v2c_v7a.cu",
        kernel_dir / "sdpa_fused_bindings.cpp",
        kernel_dir / "runtime.hpp",
    ]
    
    return load(
        name="sdpa_fused_ext",
        sources=[str(s) for s in srcs if s.exists()],
        extra_cuda_cflags=[
            "-O3", "--generate-code=arch=compute_89,code=sm_89",
            "--use_fast_math", "-lineinfo", "-Xptxas", "-v", "-std=c++17"
        ],
        verbose=True
    )

def ref_sdpa(q, k, v, causal, dropout_p=0.0):
    """Reference PyTorch SDPA"""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=causal
    )

def profile_case(B, H, L, d, causal=False, dtype=torch.float16, kernel_name="sdpa_fused_v2c_v7a_kernel"):
    """Profile a single shape with NCU"""
    print(f"\n{'━'*80}")
    print(f"📊 Profiling: B={B}, H={H}, L={L}, d={d}, causal={causal}")
    print(f"{'━'*80}\n")
    
    # Build extension
    print("[Building extension...]")
    mod = build_ext()
    
    # Prepare tensors
    q = torch.randn(B, H, L, d, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, L, d, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, L, d, device="cuda", dtype=dtype) * 0.1
    O = torch.empty_like(q)
    scale = 1.0 / math.sqrt(d)
    
    # Warmup
    print("[Warmup...]")
    for _ in range(5):
        mod.sdpa_fused_forward(q, k, v, O, causal, scale)
        torch.cuda.synchronize()
    
    # NCU command
    metrics_file = Path(__file__).parent.parent / "nsight" / "metrics.txt"
    output_dir = Path(__file__).parent.parent / "ncu_results"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"ncu_B{B}_H{H}_L{L}_d{d}_causal{int(causal)}.csv"
    
    env = os.environ.copy()
    env["CUDA_MODULE_LOADING"] = "LAZY"
    
    cmd = [
        "sudo", "/usr/local/cuda/bin/ncu",
        "--target-processes", "all",
        "--kernel-name-base", "function",
        f"--kernel-name=regex:{kernel_name}",
        "--metrics", ",".join([
            "sm__pipe_tensor_active.avg.pct_of_peak_sustained_active",
            "smsp__inst_executed_pipe_tensor.sum",
            "smsp__inst_executed_pipe_lsu.sum",
            "smsp__gmio_stall.avg.pct",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "lts__t_sectors_srcunit_tex_op_read.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
            "smsp__warps_active.avg.pct_of_peak_sustained_active",
            "sm__cycles_elapsed.avg"
        ]),
        "--csv",
        sys.executable, "-c",
        f"""
import torch, math, sys
sys.path.insert(0, '{Path(__file__).parent}')
from bench_sdpa import build_ext

mod = build_ext()
q = torch.randn({B},{H},{L},{d}, device='cuda', dtype=torch.float16) * 0.1
k = torch.randn({B},{H},{L},{d}, device='cuda', dtype=torch.float16) * 0.1
v = torch.randn({B},{H},{L},{d}, device='cuda', dtype=torch.float16) * 0.1
O = torch.empty_like(q)
scale = 1.0 / math.sqrt({d})

torch.cuda.synchronize()
mod.sdpa_fused_forward(q, k, v, O, {causal}, scale)
torch.cuda.synchronize()
print('[NCU profiling complete]', file=sys.stderr)
        """
    ]
    
    print(f"[Launching NCU... output: {output_file}]")
    with open(output_file, 'w') as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"⚠️  NCU returned non-zero: {result.returncode}")
        print(f"stderr: {result.stderr[:500]}")
    else:
        print(f"✅ NCU complete: {output_file}")
    
    return output_file

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    # Mission + stress shapes
    shapes = [
        (1, 8, 512, 64, False),   # Mission shape
        (2, 8, 2048, 64, True),   # Long sequence
        (2, 8, 2048, 128, True),  # Wide head
    ]
    
    print("\n" + "="*80)
    print("🔬 EvoEngineer NCU Profiling Session")
    print("="*80)
    
    results = []
    for shape in shapes:
        try:
            output = profile_case(*shape)
            results.append((shape, output))
        except Exception as e:
            print(f"❌ Error profiling {shape}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("📊 Profiling Session Complete")
    print("="*80)
    print(f"\n✅ Profiled {len(results)}/{len(shapes)} shapes")
    print("\nResults:")
    for shape, output in results:
        print(f"  {shape} → {output}")
    
    print("\nNext: Run parse_ncu_full.py to extract I3 insights")

