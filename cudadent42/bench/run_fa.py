#!/usr/bin/env python3
"""Benchmark FlashAttention or xFormers as fallback."""
import torch
import json
import sys
import subprocess
from pathlib import Path

def get_version_info():
    """Extract exact versions."""
    cuda_version = torch.version.cuda
    driver_output = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                   capture_output=True, text=True)
    driver = driver_output.stdout.strip() if driver_output.returncode == 0 else "unknown"
    
    props = torch.cuda.get_device_properties(0)
    sm = props.major * 10 + props.minor
    
    cudnn_version = torch.backends.cudnn.version()
    
    return {
        "torch": torch.__version__,
        "cuda": cuda_version,
        "driver": driver,
        "cudnn": cudnn_version,
        "gpu_name": torch.cuda.get_device_name(0),
        "sm": sm,
    }

def try_import_flash_attn():
    """Try to import flash_attn, return None if unavailable."""
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func, "flash_attn"
    except ImportError:
        return None, None

def try_import_xformers():
    """Try to import xformers, return None if unavailable."""
    try:
        import xformers.ops as xops
        return xops.memory_efficient_attention, "xformers"
    except ImportError:
        return None, None

def benchmark_config(attn_func, impl_name, B, H, S, d, dtype_str, causal, warmup=200, iters=500, seed=42):
    """Benchmark single configuration."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16
    
    # Create inputs (flash_attn expects [B, S, H, d])
    if impl_name == "flash_attn":
        Q = torch.randn(B, S, H, d, dtype=dtype, device='cuda', requires_grad=False)
        K = torch.randn(B, S, H, d, dtype=dtype, device='cuda', requires_grad=False)
        V = torch.randn(B, S, H, d, dtype=dtype, device='cuda', requires_grad=False)
    else:  # xformers expects [B, H, S, d]
        Q = torch.randn(B, H, S, d, dtype=dtype, device='cuda', requires_grad=False)
        K = torch.randn(B, H, S, d, dtype=dtype, device='cuda', requires_grad=False)
        V = torch.randn(B, H, S, d, dtype=dtype, device='cuda', requires_grad=False)
    
    scale = 1.0 / (d ** 0.5)
    
    # Warmup
    for _ in range(warmup):
        if impl_name == "flash_attn":
            _ = attn_func(Q, K, V, softmax_scale=scale, causal=causal)
        else:  # xformers
            _ = attn_func(Q, K, V, scale=scale)
    torch.cuda.synchronize()
    
    # Timed runs with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times_ms = []
    for _ in range(iters):
        start_event.record()
        if impl_name == "flash_attn":
            O = attn_func(Q, K, V, softmax_scale=scale, causal=causal)
        else:  # xformers
            O = attn_func(Q, K, V, scale=scale)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    
    mean_ms = sum(times_ms) / len(times_ms)
    std_ms = (sum((t - mean_ms)**2 for t in times_ms) / len(times_ms)) ** 0.5
    
    queries_per_sec = (B * H * S) / (mean_ms / 1000)
    
    return {
        "impl": impl_name,
        "B": B,
        "H": H,
        "S": S,
        "d": d,
        "dtype": dtype_str,
        "causal": causal,
        "ms_mean": mean_ms,
        "ms_std": std_ms,
        "qps": queries_per_sec,
        "seed": seed,
    }

def main():
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA not available"}))
        sys.exit(1)
    
    version_info = get_version_info()
    
    # Try to find available implementation
    attn_func, impl_name = try_import_flash_attn()
    if attn_func is None:
        attn_func, impl_name = try_import_xformers()
    
    if attn_func is None:
        print(json.dumps({"error": "Neither flash_attn nor xformers available"}))
        sys.exit(1)
    
    # Fixed workload
    configs = [
        (1, 1, 128, 64, "fp16", False),
        (8, 4, 128, 64, "fp16", False),
        (32, 8, 128, 64, "fp16", False),
        (32, 8, 256, 64, "fp16", False),
    ]
    
    # Add bf16 if supported (SM >= 80)
    if version_info["sm"] >= 80:
        configs.extend([
            (1, 1, 128, 64, "bf16", False),
            (8, 4, 128, 64, "bf16", False),
            (32, 8, 128, 64, "bf16", False),
            (32, 8, 256, 64, "bf16", False),
        ])
    
    results = []
    for B, H, S, d, dtype, causal in configs:
        try:
            result = benchmark_config(attn_func, impl_name, B, H, S, d, dtype, causal)
            result.update(version_info)
            
            # Get git commit
            try:
                commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                       capture_output=True, text=True, cwd=Path(__file__).parent.parent).stdout.strip()
                result["commit"] = commit
            except:
                result["commit"] = "unknown"
            
            results.append(result)
            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"error": f"Config {B},{H},{S},{d},{dtype}: {str(e)}"}), file=sys.stderr)
    
    # Write to file
    output_path = Path(__file__).parent / "out" / f"{impl_name}_results.jsonl"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()

