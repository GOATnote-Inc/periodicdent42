# cudadent42/bench/common/env_lock.py
"""
Environment Locking and Fingerprinting Module

Ensures reproducible benchmarks by:
- Locking dtype to FP16
- Disabling TF32 for matmul and cuDNN
- Enabling deterministic algorithms
- Recording complete environment fingerprint

Usage:
    from cudadent42.bench.common.env_lock import lock_environment, write_env
    lock_environment()
    write_env("cudadent42/bench/artifacts/env.json")
"""

import os
import json
import platform
from pathlib import Path
from typing import Dict, Any, Optional

import torch


def lock_environment() -> None:
    """
    Lock the environment to reproducible settings.
    
    Sets:
    - Default dtype: torch.float16
    - TF32: Disabled for both matmul and cuDNN
    - Matmul precision: high
    - Deterministic algorithms: Enabled (warn only for unsupported ops)
    - CUBLAS workspace: Configured for determinism
    - Python hash seed: 0 (for dictionary ordering)
    """
    # Lock dtype to FP16
    torch.set_default_dtype(torch.float16)
    
    # Disable TF32 (can silently change results)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Set explicit precision
    torch.set_float32_matmul_precision("high")
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Configure CUBLAS for determinism
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Set Python hash seed for reproducible dict ordering
    os.environ.setdefault("PYTHONHASHSEED", "0")
    
    print("✓ Environment locked: FP16, no TF32, deterministic")


def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed GPU information.
    
    Returns:
        Dict with GPU properties or CPU fallback info
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": "CPU",
            "compute_capability": None,
            "memory_total_gb": 0,
            "driver_version": None,
            "device_id": None,
            "multi_processor_count": None
        }
    
    device_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)
    
    return {
        "available": True,
        "name": torch.cuda.get_device_name(device_id),
        "compute_capability": torch.cuda.get_device_capability(device_id),
        "memory_total_gb": props.total_memory / (1024**3),
        "driver_version": torch.version.cuda,
        "device_id": device_id,
        "multi_processor_count": props.multi_processor_count
    }


def env_fingerprint() -> Dict[str, Any]:
    """
    Create comprehensive environment fingerprint.
    
    Returns:
        Dict with complete environment state for reproducibility
    """
    gpu_info = get_gpu_info()
    
    # Get numpy version
    import numpy as np
    numpy_version = np.__version__
    
    # Get scipy version (optional dependency)
    scipy_version = "not_installed"
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:
        pass
    
    # cuDNN version (if available)
    cudnn_version = None
    if torch.cuda.is_available():
        try:
            cudnn_version = torch.backends.cudnn.version()
        except Exception:
            pass
    
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "cuda_compiled_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "gpu": gpu_info,
        "default_dtype": str(torch.get_default_dtype()),
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'not_set'),
        "pythonhashseed": os.environ.get('PYTHONHASHSEED', 'not_set'),
        "packages": {
            "numpy": numpy_version,
            "scipy": scipy_version,
        },
        "host": {
            "node": platform.node(),
            "os": platform.system(),
            "os_version": platform.version(),
            "arch": platform.machine()
        }
    }


def write_env(path: str = "cudadent42/bench/artifacts/env.json") -> None:
    """
    Write environment fingerprint to JSON file.
    
    Args:
        path: Output path for JSON file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    fingerprint = env_fingerprint()
    
    with open(path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    
    print(f"✓ Environment fingerprint written to: {path}")


if __name__ == "__main__":
    # Test module
    print("Testing environment locking module...")
    lock_environment()
    
    print("\nEnvironment fingerprint:")
    fingerprint = env_fingerprint()
    print(json.dumps(fingerprint, indent=2))
    
    print("\n✓ Module test complete")

