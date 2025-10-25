#!/usr/bin/env python3
# Copyright 2025 GOATnote Inc.
# Security: Fixed shell injection vulnerability (removed shell=True)

import subprocess, json, os, yaml, datetime, pathlib, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ART = ROOT/"artifacts"; ART.mkdir(parents=True, exist_ok=True)

def sh(cmd): 
    """Execute command safely without shell=True to prevent injection"""
    if isinstance(cmd, str):
        cmd = cmd.split()
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    gpu = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    clocks = sh("nvidia-smi -q -d CLOCK")
    commit = sh("git rev-parse HEAD") if (ROOT/".git").exists() else "N/A"
    
    # Get Python version
    python_ver = sh("python3 -V")
    
    # Get PyTorch info (fallback on error)
    try:
        pytorch_ver = subprocess.run(["python3", "-c", "import torch; print(torch.__version__)"], 
                                     capture_output=True, text=True, check=False).stdout.strip()
    except: pytorch_ver = "N/A"
    
    try:
        cuda_ver = subprocess.run(["python3", "-c", "import torch; print(torch.version.cuda)"],
                                  capture_output=True, text=True, check=False).stdout.strip()
    except: cuda_ver = "N/A"
    
    try:
        triton_ver = subprocess.run(["python3", "-c", "import triton; print(triton.__version__)"],
                                    capture_output=True, text=True, check=False).stdout.strip()
    except: triton_ver = "N/A"
    
    # Try nvcc in common locations
    nvcc_ver = sh("nvcc --version") or sh("/usr/local/cuda/bin/nvcc --version") or "N/A"
    
    try:
        cudnn_ver = subprocess.run(["python3", "-c", "import torch; print(torch.backends.cudnn.version())"],
                                   capture_output=True, text=True, check=False).stdout.strip()
    except: cudnn_ver = "N/A"
    
    env = {
        "timestamp": datetime.datetime.now().isoformat(),
        "gpu_info": gpu,
        "clocks": clocks,
        "python": python_ver,
        "pytorch": pytorch_ver,
        "cuda_from_torch": cuda_ver,
        "triton": triton_ver,
        "nvcc": nvcc_ver,
        "cudnn": cudnn_ver,
        "commit": commit,
        "seed": 17,
        "commands": {
            "repro": "bash scripts/repro.sh",
            "bench": "bash scripts/bench.sh",
            "profile": "bash scripts/profile.sh"
        }
    }
    (ART/"manifest.yaml").write_text(yaml.safe_dump(env, sort_keys=False))
if __name__=="__main__":
    main()

