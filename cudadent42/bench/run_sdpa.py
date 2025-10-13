#!/usr/bin/env python3
"""Benchmark PyTorch SDPA with flash backend."""
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

def benchmark_config(B, H, S, d, dtype_str, causal, warmup=200, iters=500, seed=42):
    """Benchmark single configuration."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16
    
    # Create inputs
    Q = torch.randn(B, H, S, d, dtype=dtype, device='cuda', requires_grad=False)
    K = torch.randn(B, H, S, d, dtype=dtype, device='cuda', requires_grad=False)
    V = torch.randn(B, H, S, d, dtype=dtype, device='cuda', requires_grad=False)
    
    scale = 1.0 / (d ** 0.5)
    
    # Warmup
    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=causal)
    torch.cuda.synchronize()
    
    # Timed runs with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times_ms = []
    for _ in range(iters):
        start_event.record()
        O = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=causal)
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    
    mean_ms = sum(times_ms) / len(times_ms)
    std_ms = (sum((t - mean_ms)**2 for t in times_ms) / len(times_ms)) ** 0.5
    
    queries_per_sec = (B * H * S) / (mean_ms / 1000)
    
    return {
        "impl": "pytorch_sdpa",
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
        result = benchmark_config(B, H, S, d, dtype, causal)
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
    
    # Write to file
    output_path = Path(__file__).parent / "out" / "sdpa_results.jsonl"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    main()

