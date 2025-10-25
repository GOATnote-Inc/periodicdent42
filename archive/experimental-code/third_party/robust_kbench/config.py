"""
robust-kbench Configuration
"""
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class ShapeConfig:
    """Represents a single benchmark shape configuration"""
    name: str
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    causal: bool
    dtype: str = "float16"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def __str__(self) -> str:
        causal_str = "causal" if self.causal else "noncausal"
        return f"B{self.batch_size}_H{self.num_heads}_S{self.seq_len}_D{self.head_dim}_{causal_str}_{self.dtype}"


@dataclass
class RBKConfig:
    """robust-kbench configuration"""
    
    # Benchmark parameters
    warmups: int = 20
    iterations: int = 100
    
    # Shape grid
    shapes: List[ShapeConfig] = field(default_factory=list)
    
    # Output settings
    output_dir: str = "benchmarks/rbk"
    save_json: bool = True
    save_csv: bool = True
    save_markdown: bool = True
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "RBKConfig":
        """Load configuration from YAML file"""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        # Parse shapes
        shapes = []
        for shape_data in data.get("shapes", []):
            shapes.append(ShapeConfig(**shape_data))
        
        return cls(
            warmups=data.get("warmups", 20),
            iterations=data.get("iterations", 100),
            shapes=shapes,
            output_dir=data.get("output_dir", "benchmarks/rbk"),
            save_json=data.get("save_json", True),
            save_csv=data.get("save_csv", True),
            save_markdown=data.get("save_markdown", True),
        )
    
    def to_yaml(self, yaml_path: Path):
        """Save configuration to YAML file"""
        data = {
            "warmups": self.warmups,
            "iterations": self.iterations,
            "shapes": [s.to_dict() for s in self.shapes],
            "output_dir": self.output_dir,
            "save_json": self.save_json,
            "save_csv": self.save_csv,
            "save_markdown": self.save_markdown,
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, indent=2, default_flow_style=False)
    
    @classmethod
    def default_l4_grid(cls) -> "RBKConfig":
        """Create default L4 benchmark grid"""
        shapes = []
        
        # Standard configurations
        for B in [1, 4, 8]:
            for H in [8, 16]:
                for S in [128, 512, 1024, 2048]:
                    for D in [64, 128]:
                        for causal in [True, False]:
                            for dtype in ["float16", "bfloat16"]:
                                shapes.append(ShapeConfig(
                                    name=f"std_B{B}_H{H}_S{S}_D{D}",
                                    batch_size=B,
                                    num_heads=H,
                                    seq_len=S,
                                    head_dim=D,
                                    causal=causal,
                                    dtype=dtype,
                                ))
        
        # Long sequence stress tests
        for S in [4096, 8192]:
            shapes.append(ShapeConfig(
                name=f"longseq_S{S}",
                batch_size=1,
                num_heads=8,
                seq_len=S,
                head_dim=128,
                causal=True,
                dtype="float16",
            ))
        
        return cls(shapes=shapes)
    
    @classmethod
    def canonical_shapes(cls) -> "RBKConfig":
        """Create canonical benchmark shapes for optimization"""
        shapes = [
            # Canonical 1: Large batch, long sequence
            ShapeConfig(
                name="canonical_1",
                batch_size=4,
                num_heads=16,
                seq_len=2048,
                head_dim=128,
                causal=True,
                dtype="float16",
            ),
            # Canonical 2: Single batch, very long sequence
            ShapeConfig(
                name="canonical_2",
                batch_size=1,
                num_heads=8,
                seq_len=4096,
                head_dim=128,
                causal=True,
                dtype="float16",
            ),
            # Canonical 3: Moderate batch, moderate sequence
            ShapeConfig(
                name="canonical_3",
                batch_size=8,
                num_heads=16,
                seq_len=1024,
                head_dim=64,
                causal=False,
                dtype="float16",
            ),
        ]
        
        return cls(shapes=shapes)

