#!/usr/bin/env python3
import subprocess, json, os, yaml, datetime, pathlib, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ART = ROOT/"artifacts"; ART.mkdir(parents=True, exist_ok=True)

def sh(cmd): 
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

def main():
    gpu = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    clocks = sh("nvidia-smi -q -d CLOCK")
    commit = sh("git rev-parse HEAD") if (ROOT/".git").exists() else "N/A"
    env = {
        "timestamp": datetime.datetime.now().isoformat(),
        "gpu_info": gpu,
        "clocks": clocks,
        "python": sh("python3 -V"),
        "pytorch": sh("python3 -c 'import torch; print(torch.__version__)'"),
        "cuda_from_torch": sh("python3 -c 'import torch; print(torch.version.cuda)'"),
        "triton": sh("python3 -c 'import importlib; print(importlib.import_module(\"triton\").__version__)' || true"),
        "nvcc": sh("nvcc --version || /usr/local/cuda/bin/nvcc --version || true"),
        "cudnn": sh("python3 -c 'import torch; print(torch.backends.cudnn.version())'"),
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

