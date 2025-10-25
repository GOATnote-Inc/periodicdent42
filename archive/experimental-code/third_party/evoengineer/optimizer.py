"""
EvoEngineer Optimizer: Parameter search and candidate management
"""
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


@dataclass
class SearchSpace:
    """Defines the parameter search space for kernel optimization"""
    
    # Tile sizes
    BLOCK_M: List[int] = field(default_factory=lambda: [64, 80, 96, 128])
    BLOCK_N: List[int] = field(default_factory=lambda: [32, 64, 80, 128])
    BLOCK_K: List[int] = field(default_factory=lambda: [16, 32, 64])
    
    # Warp configuration
    NUM_WARPS: List[int] = field(default_factory=lambda: [4, 8])
    
    # Pipeline stages
    STAGES: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Other optimizations
    UNROLL: List[int] = field(default_factory=lambda: [1, 2, 4])
    CP_ASYNC: List[bool] = field(default_factory=lambda: [True, False])
    SWIZZLE: List[bool] = field(default_factory=lambda: [True, False])
    HALF2: List[bool] = field(default_factory=lambda: [True, False])
    
    def to_dict(self) -> Dict[str, List[Any]]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def for_l4(cls) -> "SearchSpace":
        """Create L4-specific search space with memory constraints"""
        return cls(
            BLOCK_M=[64, 80, 96],  # Avoid 128 due to SMEM limits
            BLOCK_N=[32, 64],
            BLOCK_K=[16, 32],
            NUM_WARPS=[4, 8],
            STAGES=[2, 3],  # L4 benefits from pipelining
            UNROLL=[1, 2],
            CP_ASYNC=[True],  # Always use cp.async on sm_89
            SWIZZLE=[True],
            HALF2=[True],  # Always use half2 for FP16
        )


@dataclass
class Candidate:
    """Represents a kernel configuration candidate"""
    
    # Parameters
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    NUM_WARPS: int
    STAGES: int
    UNROLL: int
    CP_ASYNC: bool
    SWIZZLE: bool
    HALF2: bool
    
    # Metadata
    generation: int = 0
    parent_hash: Optional[str] = None
    
    # Results (populated after evaluation)
    p50_latency_ms: Optional[float] = None
    p90_latency_ms: Optional[float] = None
    tflops: Optional[float] = None
    occupancy: Optional[float] = None
    register_count: Optional[int] = None
    spills: Optional[int] = None
    correctness_passed: bool = False
    
    @property
    def hash(self) -> str:
        """Unique hash for this configuration"""
        params = f"{self.BLOCK_M}_{self.BLOCK_N}_{self.BLOCK_K}_{self.NUM_WARPS}_" \
                 f"{self.STAGES}_{self.UNROLL}_{int(self.CP_ASYNC)}_{int(self.SWIZZLE)}_{int(self.HALF2)}"
        return hashlib.md5(params.encode()).hexdigest()[:8]
    
    def passes_gates(self) -> bool:
        """Check if candidate passes hard gates"""
        if not self.correctness_passed:
            return False
        if self.spills and self.spills > 0:
            return False
        if self.occupancy and self.occupancy < 0.3:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_compile_flags(self) -> List[str]:
        """Generate compile-time defines for this config"""
        flags = [
            f"-DBLOCK_M={self.BLOCK_M}",
            f"-DBLOCK_N={self.BLOCK_N}",
            f"-DBLOCK_K={self.BLOCK_K}",
            f"-DNUM_WARPS={self.NUM_WARPS}",
            f"-DSTAGES={self.STAGES}",
            f"-DUNROLL={self.UNROLL}",
        ]
        if self.CP_ASYNC:
            flags.append("-DCP_ASYNC=1")
        if self.SWIZZLE:
            flags.append("-DSWIZZLE=1")
        if self.HALF2:
            flags.append("-DHALF2=1")
        return flags
    
    def estimate_smem_bytes(self, head_dim: int = 64) -> int:
        """Estimate shared memory usage in bytes"""
        # K and V tiles (double-buffered if STAGES > 1)
        kv_size = 2 * self.STAGES * self.BLOCK_N * head_dim * 2  # FP16 = 2 bytes
        
        # S and temp_O (if not register-only)
        s_size = self.BLOCK_M * self.BLOCK_N * 4  # FP32
        temp_o_size = self.BLOCK_M * head_dim * 4  # FP32
        
        # Add padding if swizzled
        padding = 128 if self.SWIZZLE else 0
        
        return kv_size + s_size + temp_o_size + padding


class KernelOptimizer:
    """Evolutionary optimizer for kernel parameters"""
    
    def __init__(self, search_space: SearchSpace, output_dir: Path):
        self.search_space = search_space
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.population: List[Candidate] = []
        self.leaderboard: List[Candidate] = []
        self.generation = 0
    
    def initialize_population(self, size: int = 20) -> List[Candidate]:
        """Generate initial population with diverse configs"""
        import random
        
        population = []
        for _ in range(size):
            candidate = Candidate(
                BLOCK_M=random.choice(self.search_space.BLOCK_M),
                BLOCK_N=random.choice(self.search_space.BLOCK_N),
                BLOCK_K=random.choice(self.search_space.BLOCK_K),
                NUM_WARPS=random.choice(self.search_space.NUM_WARPS),
                STAGES=random.choice(self.search_space.STAGES),
                UNROLL=random.choice(self.search_space.UNROLL),
                CP_ASYNC=random.choice(self.search_space.CP_ASYNC),
                SWIZZLE=random.choice(self.search_space.SWIZZLE),
                HALF2=random.choice(self.search_space.HALF2),
                generation=0,
            )
            # Prune candidates that exceed L4 SMEM limit (48KB)
            if candidate.estimate_smem_bytes() <= 49152:
                population.append(candidate)
        
        self.population = population
        return population
    
    def update_leaderboard(self, candidate: Candidate):
        """Add candidate to leaderboard if it's competitive"""
        if not candidate.passes_gates():
            return
        
        # Add to leaderboard
        self.leaderboard.append(candidate)
        
        # Sort by p50 latency (lower is better)
        self.leaderboard.sort(key=lambda c: c.p50_latency_ms if c.p50_latency_ms else float('inf'))
        
        # Keep top 10
        self.leaderboard = self.leaderboard[:10]
    
    def save_leaderboard(self):
        """Save leaderboard to JSON"""
        leaderboard_file = self.output_dir / "leaderboard.json"
        data = [c.to_dict() for c in self.leaderboard]
        with open(leaderboard_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_best(self) -> Optional[Candidate]:
        """Get current best candidate"""
        return self.leaderboard[0] if self.leaderboard else None

