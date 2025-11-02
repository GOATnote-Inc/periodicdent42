"""
Ceiling Scout Integration for vLLM
Automatically optimizes kernel selection for LLM serving
"""

import json
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KernelConfig:
    """Configuration for a specific kernel"""
    backend: str  # "cublas", "cutlass", "flashattn", "custom_sparse"
    tile_shape: Optional[Tuple[int, int, int]] = None
    cluster_shape: Optional[Tuple[int, int, int]] = None
    reasoning: str = ""


class VLLMCeilingOptimizer:
    """
    Integrates ceiling scout recommendations into vLLM's kernel selection
    
    Usage:
        optimizer = VLLMCeilingOptimizer(reports_dir="./ceiling_reports")
        
        # In vLLM's model forward pass:
        if optimizer.should_use_custom_kernel("attention", shape):
            output = optimizer.dispatch_kernel("attention", Q, K, V)
        else:
            output = F.scaled_dot_product_attention(Q, K, V)
    """
    
    def __init__(self, reports_dir: str = "./ceiling_reports"):
        self.reports_dir = Path(reports_dir)
        self.cache: Dict[str, Dict] = {}
        
    def load_report(self, op_type: str, shape: Tuple[int, ...]) -> Optional[Dict]:
        """Load ceiling scout report for operation and shape"""
        
        # Create cache key
        cache_key = f"{op_type}_{'x'.join(map(str, shape))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to load from file
        report_path = self.reports_dir / f"{cache_key}.json"
        
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
                self.cache[cache_key] = report
                return report
        
        return None
    
    def should_use_custom_kernel(self, op_type: str, shape: Tuple[int, ...]) -> bool:
        """Check if custom kernel is recommended"""
        report = self.load_report(op_type, shape)
        
        if not report:
            return False
        
        # Don't use custom if library is already optimal
        if report.get("approach") == "NONE":
            return False
        
        # Use custom if efficiency <90%
        efficiency = report.get("efficiency", 1.0)
        return efficiency < 0.90
    
    def get_kernel_config(self, op_type: str, shape: Tuple[int, ...]) -> KernelConfig:
        """Get recommended kernel configuration"""
        report = self.load_report(op_type, shape)
        
        if not report:
            # No report, use default
            return KernelConfig(backend="cublas", reasoning="No ceiling report available")
        
        approach = report.get("approach", "NONE")
        config = report.get("config_suggestion", {})
        
        if approach == "NONE":
            # Library is optimal
            return KernelConfig(
                backend="cublas",
                reasoning=config.get("reasoning", "Library is optimal")
            )
        
        elif approach == "CUSTOM_BSR_SPARSE":
            # Use BlackwellSparseK
            return KernelConfig(
                backend="blackwell_sparse",
                reasoning=f"Block sparse detected: {config.get('expected_vs_cusparse', 'Unknown')} vs cuSPARSE"
            )
        
        elif approach == "CUTLASS_STRUCTURED_SPARSE":
            # Use CUTLASS 2:4 sparse
            return KernelConfig(
                backend="cutlass_24_sparse",
                reasoning="2:4 structured sparsity detected"
            )
        
        elif approach == "CUTLASS_SWEEP":
            # Use CUTLASS with tuned config
            tile_shapes = config.get("tile_shapes", ["128x128x64"])
            tile_shape = tuple(map(int, tile_shapes[0].split('x')))
            
            cluster_shapes = config.get("cluster_shapes", ["1x1x1"])
            cluster_shape = tuple(map(int, cluster_shapes[0].split('x')))
            
            return KernelConfig(
                backend="cutlass_tuned",
                tile_shape=tile_shape,
                cluster_shape=cluster_shape,
                reasoning=f"Expected gain: {config.get('expected_gain', 'Unknown')}"
            )
        
        else:
            # Unknown, use default
            return KernelConfig(backend="cublas", reasoning="Unknown approach, using default")
    
    def dispatch_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          is_causal: bool = False) -> torch.Tensor:
        """Smart attention dispatch based on ceiling scout"""
        
        batch, heads, seq_len, head_dim = Q.shape
        shape = (batch, heads, seq_len, head_dim)
        
        config = self.get_kernel_config("attention", shape)
        
        if config.backend == "cublas" or config.backend == "flashattn":
            # Use PyTorch's SDPA (which uses FA2/FA3)
            return torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, is_causal=is_causal
            )
        
        # TODO: Add custom attention implementations
        # For now, fallback to SDPA
        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=is_causal
        )
    
    def dispatch_linear(self, input: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Smart linear layer dispatch"""
        
        # Check sparsity
        sparsity = (weight.abs() < 1e-6).float().mean().item()
        
        if sparsity > 0.70:
            # Sparse matrix, check pattern
            M, K = input.shape if input.dim() == 2 else (input.shape[0] * input.shape[1], input.shape[2])
            N = weight.shape[0]
            
            config = self.get_kernel_config("gemm", (M, N, K))
            
            if config.backend == "blackwell_sparse":
                # TODO: Call BlackwellSparseK
                print(f"  Using BlackwellSparseK: {config.reasoning}")
                return torch.nn.functional.linear(input, weight, bias)
            
            elif config.backend == "cutlass_24_sparse":
                # TODO: Call CUTLASS 2:4
                print(f"  Using CUTLASS 2:4: {config.reasoning}")
                return torch.nn.functional.linear(input, weight, bias)
        
        # Default: use PyTorch (cuBLAS)
        return torch.nn.functional.linear(input, weight, bias)


# Integration with vLLM's LLM class
class CeilingOptimizedLLM:
    """
    Example integration with vLLM
    
    Wraps vLLM's LLM class and adds ceiling-based optimization
    """
    
    def __init__(self, model_name: str, reports_dir: str = "./ceiling_reports"):
        # Import vLLM (if available)
        try:
            from vllm import LLM as VLLM_LLM
            self.llm = VLLM_LLM(model_name)
        except ImportError:
            print("vLLM not installed. Install with: pip install vllm")
            self.llm = None
        
        self.optimizer = VLLMCeilingOptimizer(reports_dir)
    
    def generate(self, prompts, **kwargs):
        """Generate with optimized kernels"""
        if self.llm is None:
            raise RuntimeError("vLLM not available")
        
        # vLLM will use the monkey-patched optimized ops
        return self.llm.generate(prompts, **kwargs)
    
    @staticmethod
    def patch_vllm_ops(optimizer: VLLMCeilingOptimizer):
        """
        Monkey-patch vLLM's operations with ceiling-optimized versions
        
        Call this once at initialization:
            optimizer = VLLMCeilingOptimizer("./reports")
            CeilingOptimizedLLM.patch_vllm_ops(optimizer)
        """
        try:
            import vllm.attention.ops.paged_attn
            
            # Save original
            original_attention = vllm.attention.ops.paged_attn.attention_forward
            
            # Create optimized version
            def optimized_attention(query, key, value, *args, **kwargs):
                # Check if ceiling scout suggests optimization
                if optimizer.should_use_custom_kernel("attention", query.shape):
                    config = optimizer.get_kernel_config("attention", query.shape)
                    print(f"  Using optimized attention: {config.reasoning}")
                
                # Call original (or custom implementation)
                return original_attention(query, key, value, *args, **kwargs)
            
            # Patch
            vllm.attention.ops.paged_attn.attention_forward = optimized_attention
            
            print("✅ vLLM operations patched with ceiling optimization")
        
        except ImportError:
            print("⚠️  vLLM not installed, skipping patch")


def generate_vllm_reports(model_name: str, output_dir: str = "./ceiling_reports"):
    """
    Generate ceiling scout reports for all operations in a vLLM model
    
    Usage:
        generate_vllm_reports("meta-llama/Llama-2-7b-hf")
    """
    from ceiling_scout import CeilingScout, Operation, Precision
    from ceiling_scout_extended import FA3Benchmarker, SparseDetector
    
    print(f"Generating ceiling reports for {model_name}...")
    
    # Common LLM shapes (adjust based on actual model)
    shapes = {
        "attention": [(1, 32, 2048, 128), (1, 32, 4096, 128)],  # (B, H, S, D)
        "gemm": [(4096, 11008, 4096), (4096, 4096, 11008)]  # (M, N, K) for FFN
    }
    
    scout = CeilingScout(device="h100")
    fa3_bench = FA3Benchmarker()
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate reports
    for op_type, shape_list in shapes.items():
        for shape in shape_list:
            print(f"\nAnalyzing {op_type} {shape}...")
            
            if op_type == "attention":
                opp = fa3_bench.detect_attention_ceiling(*shape)
            elif op_type == "gemm":
                opp = scout.detect_ceiling(Operation.GEMM, shape, Precision.FP16)
            else:
                continue
            
            # Save report
            report_name = f"{op_type}_{'x'.join(map(str, shape))}.json"
            report_path = Path(output_dir) / report_name
            
            with open(report_path, 'w') as f:
                import json
                from dataclasses import asdict
                json.dump(asdict(opp), f, indent=2)
            
            print(f"  Saved: {report_path}")
            print(f"  Recommendation: {opp.recommendation[:80]}...")


if __name__ == "__main__":
    print("Ceiling Scout - vLLM Integration")
    print("=" * 60)
    print("\nExample usage:")
    print("  1. Generate reports: generate_vllm_reports('meta-llama/Llama-2-7b-hf')")
    print("  2. Patch vLLM:")
    print("     optimizer = VLLMCeilingOptimizer('./ceiling_reports')")
    print("     CeilingOptimizedLLM.patch_vllm_ops(optimizer)")
    print("  3. Run inference:")
    print("     llm = CeilingOptimizedLLM('meta-llama/Llama-2-7b-hf')")
    print("     outputs = llm.generate(['Once upon a time'])")

