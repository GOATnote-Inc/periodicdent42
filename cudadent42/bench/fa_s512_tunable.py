#!/usr/bin/env python3
"""
Tunable FA-S512 Kernel Interface

Builds and executes the fa_s512.cu kernel with different configurations.
Handles JIT compilation with tunable parameters.

Author: GOATnote Autonomous Research Lab Initiative
Date: 2025-10-13
"""

import os
import sys
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.utils.cpp_extension

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class FA_S512_Tunable:
    """
    Tunable FlashAttention kernel for S=512 on L4 (SM_89)
    
    Compiles and loads kernel variants with different tile sizes,
    warp counts, pipeline stages, etc.
    """
    
    def __init__(self, cache_dir: str = "/tmp/fa_s512_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.loaded_modules = {}  # config_hash -> torch.utils.cpp_extension module
        
        self.kernel_source = PROJECT_ROOT / "cudadent42/bench/kernels/fa_s512.cu"
        if not self.kernel_source.exists():
            raise FileNotFoundError(f"Kernel source not found: {self.kernel_source}")
    
    def _config_hash(self, config: Dict[str, int]) -> str:
        """Generate hash for config (for caching compiled kernels)"""
        config_str = "_".join(f"{k}{v}" for k, v in sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _build_kernel(self, config: Dict[str, int]) -> torch.utils.cpp_extension.load:
        """
        JIT compile kernel with given config
        
        Args:
            config: Dict with keys:
                - BLOCK_M, BLOCK_N, BLOCK_K
                - NUM_WARPS
                - STAGES
                - UNROLL
                - CP_ASYNC (0/1)
                - SWIZZLE (0/1)
                - HALF2 (0/1)
        
        Returns:
            Loaded torch extension module
        """
        config_hash = self._config_hash(config)
        
        # Check if already loaded
        if config_hash in self.loaded_modules:
            return self.loaded_modules[config_hash]
        
        # Build flags
        nvcc_flags = [
            '-O3',
            '--use_fast_math',
            '-lineinfo',
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
            '-Xcompiler=-fno-omit-frame-pointer',
            '-Xcompiler=-fno-common',
            '-gencode=arch=compute_89,code=sm_89',
            '-std=c++17',
        ]
        
        # Add config as -D flags
        for key, val in config.items():
            nvcc_flags.append(f'-D{key}={val}')
        
        # Module name (unique per config)
        module_name = f"fa_s512_{config_hash}"
        build_dir = self.cache_dir / module_name
        
        try:
            # Load or compile
            module = torch.utils.cpp_extension.load(
                name=module_name,
                sources=[str(self.kernel_source)],
                extra_cuda_cflags=nvcc_flags,
                build_directory=str(build_dir),
                verbose=False
            )
            
            self.loaded_modules[config_hash] = module
            return module
        
        except Exception as e:
            print(f"❌ Build failed for config {config}: {e}")
            return None
    
    def run(
        self,
        config: Dict[str, int],
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        iterations: int = 100,
        warmup: int = 20
    ) -> Tuple[Optional[torch.Tensor], List[float], Dict]:
        """
        Run kernel with given config
        
        Args:
            config: Kernel configuration
            Q, K, V: Input tensors [B, H, S, D] in FP16
            iterations: Number of timing iterations
            warmup: Number of warmup iterations
        
        Returns:
            (output_tensor, latencies_ms, meta)
            output_tensor: Result tensor or None if build failed
            latencies_ms: List of latencies in milliseconds
            meta: Dict with build status, errors, etc.
        """
        meta = {
            'config': config,
            'build_success': False,
            'run_success': False,
            'error': None
        }
        
        # Build kernel
        try:
            module = self._build_kernel(config)
            if module is None:
                meta['error'] = "Build failed"
                return None, [], meta
            
            meta['build_success'] = True
        except Exception as e:
            meta['error'] = f"Build exception: {e}"
            return None, [], meta
        
        # Allocate output
        B, H, S, D = Q.shape
        O = torch.empty_like(Q)
        
        # Warmup
        try:
            for _ in range(warmup):
                module.launch_fa_s512(
                    Q.data_ptr(),
                    K.data_ptr(),
                    V.data_ptr(),
                    O.data_ptr(),
                    B, H, S, D,
                    0  # default stream
                )
            torch.cuda.synchronize()
        except Exception as e:
            meta['error'] = f"Warmup failed: {e}"
            return None, [], meta
        
        # Timing runs
        latencies = []
        try:
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                module.launch_fa_s512(
                    Q.data_ptr(),
                    K.data_ptr(),
                    V.data_ptr(),
                    O.data_ptr(),
                    B, H, S, D,
                    0
                )
                end.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start.elapsed_time(end)
                latencies.append(elapsed_ms)
            
            meta['run_success'] = True
        except Exception as e:
            meta['error'] = f"Runtime failed: {e}"
            return O if 'O' in locals() else None, latencies, meta
        
        return O, latencies, meta


def test_kernel():
    """Quick test of kernel build and execution"""
    print("Testing FA-S512 Tunable Kernel...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("❌ CUDA not available")
        return
    
    # Test config
    config = {
        'BLOCK_M': 128,
        'BLOCK_N': 64,
        'BLOCK_K': 32,
        'NUM_WARPS': 4,
        'STAGES': 2,
        'UNROLL': 1,
        'CP_ASYNC': 1,
        'SWIZZLE': 1,
        'HALF2': 1,
    }
    
    # Create inputs
    B, H, S, D = 32, 8, 512, 64
    Q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    K = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    V = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    
    # Build and run
    kernel = FA_S512_Tunable()
    O, latencies, meta = kernel.run(config, Q, K, V, iterations=10, warmup=3)
    
    if meta['build_success'] and meta['run_success']:
        print(f"✅ Build and run successful")
        print(f"   Median latency: {np.median(latencies):.4f} ms")
        print(f"   Output shape: {O.shape}")
    else:
        print(f"❌ Failed: {meta['error']}")


if __name__ == "__main__":
    test_kernel()

