"""Configuration management for BlackwellSparseK."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """BlackwellSparseK configuration."""
    
    # CUDA architecture
    cuda_arch: str = "90a"  # "90a" for H100, "100" for Blackwell
    auto_detect_arch: bool = True
    
    # Kernel parameters
    block_m: int = 64
    block_n: int = 64
    num_stages: int = 2
    num_warps: int = 4
    
    # Build options
    use_fast_math: bool = True
    enable_profiling: bool = False
    debug_mode: bool = False
    
    # Performance tuning
    persistent_kernel: bool = True
    warp_specialized: bool = True
    use_tma: bool = True  # Tensor Memory Accelerator
    
    # Paths
    cutlass_path: Optional[str] = None
    build_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.cutlass_path is None:
            self.cutlass_path = os.environ.get("CUTLASS_PATH", "/opt/cutlass")
        
        if self.build_dir is None:
            self.build_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "blackwell_sparsek", "build"
            )
        
        # Validate architecture
        if self.cuda_arch not in ["90a", "100"]:
            raise ValueError(f"Unsupported CUDA architecture: {self.cuda_arch}")
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            cuda_arch=os.environ.get("BSK_CUDA_ARCH", "90a"),
            auto_detect_arch=os.environ.get("BSK_AUTO_DETECT", "1") == "1",
            use_fast_math=os.environ.get("BSK_FAST_MATH", "1") == "1",
            enable_profiling=os.environ.get("BSK_PROFILE", "0") == "1",
            debug_mode=os.environ.get("BSK_DEBUG", "0") == "1",
            cutlass_path=os.environ.get("CUTLASS_PATH"),
            build_dir=os.environ.get("BSK_BUILD_DIR"),
        )


# Global config instance
_default_config = None


def get_default_config() -> Config:
    """Get the default global configuration."""
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config


def set_default_config(config: Config):
    """Set the default global configuration."""
    global _default_config
    _default_config = config

