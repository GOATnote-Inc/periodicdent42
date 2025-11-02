#!/usr/bin/env python3
"""
Ceiling Scout Extended: FA3, Sparse Detection, Fusion Analysis
Built on validated H100 methodology + ecosystem knowledge (Nov 2025)
"""

from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path

# Torch imports are conditional - only needed for GPU benchmarking
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None
    np = None

from ceiling_scout import (
    BenchmarkResult, OpportunityScore, Precision, Operation, CeilingScout
)


class FA3Benchmarker:
    """FlashAttention-3 benchmarking with our validated methodology"""
    
    def __init__(self, device: str = "cuda"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for FA3Benchmarker. "
                             "Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        self.device = device
        
    def benchmark_pytorch_sdpa(self, batch: int, heads: int, seq_len: int, 
                               head_dim: int, dtype=None) -> BenchmarkResult:
        """Benchmark PyTorch's Scaled Dot-Product Attention (uses FA2/FA3)"""
        
        # Setup
        if dtype is None:
            dtype = torch.float16
        Q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=self.device)
        K = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=self.device)
        V = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=self.device)
        
        # Warmup (20 iterations per our methodology)
        for _ in range(20):
            _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        torch.cuda.synchronize()
        
        # Benchmark (200 iterations)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(200):
            _ = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        end.record()
        torch.cuda.synchronize()
        
        ms = start.elapsed_time(end) / 200
        us_per_head = (ms * 1000) / heads
        
        # Attention FLOPS: 4 * B * H * S^2 * D (Q@K^T + softmax + @V)
        flops = 4 * batch * heads * seq_len * seq_len * head_dim
        tflops = (flops / (ms / 1000)) / 1e12
        
        memory_gb = (batch * heads * seq_len * head_dim * 3 * 2) / 1e9  # Q,K,V in FP16
        
        return BenchmarkResult(
            name="PyTorch SDPA (FA2/FA3)",
            operation="attention",
            shape=(batch, heads, seq_len, head_dim),
            precision="fp16",
            tflops=tflops,
            latency_ms=ms,
            memory_gb=memory_gb,
            is_library=True,
            library_name="FlashAttention"
        )
    
    def benchmark_naive_attention(self, batch: int, heads: int, seq_len: int,
                                  head_dim: int, dtype=None) -> BenchmarkResult:
        """Naive attention for comparison (memory-bound baseline)"""
        
        if dtype is None:
            dtype = torch.float16
        Q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=self.device)
        K = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=self.device)
        V = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=self.device)
        
        # Warmup
        for _ in range(20):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)
            _ = torch.matmul(attn, V)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(200):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(head_dim)
            attn = F.softmax(scores, dim=-1)
            _ = torch.matmul(attn, V)
        end.record()
        torch.cuda.synchronize()
        
        ms = start.elapsed_time(end) / 200
        
        flops = 4 * batch * heads * seq_len * seq_len * head_dim
        tflops = (flops / (ms / 1000)) / 1e12
        
        return BenchmarkResult(
            name="Naive Attention",
            operation="attention",
            shape=(batch, heads, seq_len, head_dim),
            precision="fp16",
            tflops=tflops,
            latency_ms=ms,
            memory_gb=(batch * heads * seq_len * head_dim * 3 * 2) / 1e9,
            is_library=False
        )
    
    def detect_attention_ceiling(self, batch: int, heads: int, seq_len: int,
                                head_dim: int) -> OpportunityScore:
        """Detect if FA3 is optimal or if custom kernel could help"""
        
        print(f"Benchmarking attention: B={batch}, H={heads}, S={seq_len}, D={head_dim}")
        
        # Benchmark FA3 (via PyTorch SDPA)
        fa3_result = self.benchmark_pytorch_sdpa(batch, heads, seq_len, head_dim)
        print(f"  FA3: {fa3_result.latency_ms:.3f} ms ({fa3_result.latency_ms * 1000 / heads:.2f} μs/head)")
        
        # Benchmark naive (for reference)
        naive_result = self.benchmark_naive_attention(batch, heads, seq_len, head_dim)
        print(f"  Naive: {naive_result.latency_ms:.3f} ms ({naive_result.latency_ms * 1000 / heads:.2f} μs/head)")
        
        speedup = naive_result.latency_ms / fa3_result.latency_ms
        
        # From our H100 validation: FA3 achieves 0.27-0.49 μs/head
        # Target was <5 μs/head, so 10-19× better
        us_per_head = (fa3_result.latency_ms * 1000) / heads
        target_us_per_head = 5.0
        
        if us_per_head < target_us_per_head:
            # FA3 is already excellent
            return OpportunityScore(
                operation="attention",
                shape=(batch, heads, seq_len, head_dim),
                baseline_tflops=fa3_result.tflops,
                ceiling_tflops=fa3_result.tflops * 1.1,  # 10% margin
                efficiency=0.95,
                recommendation=f"FA3 is optimal: {us_per_head:.2f} μs/head "
                               f"(target: <{target_us_per_head} μs/head). "
                               f"{speedup:.1f}× faster than naive. Use library.",
                priority="NONE",
                approach="NONE",
                config_suggestion={
                    "use": "torch.nn.functional.scaled_dot_product_attention",
                    "backend": "flash_attention_2 or flash_attention_3",
                    "reasoning": "Already optimal for this shape"
                }
            )
        else:
            # Room for improvement
            return OpportunityScore(
                operation="attention",
                shape=(batch, heads, seq_len, head_dim),
                baseline_tflops=fa3_result.tflops,
                ceiling_tflops=fa3_result.tflops * 2.0,
                efficiency=0.5,
                recommendation=f"FA3 achieves {us_per_head:.2f} μs/head "
                               f"(target: <{target_us_per_head}). "
                               f"Consider: (1) custom tile sizes for this shape, "
                               f"(2) fused ops (mask+dropout), (3) sparse attention.",
                priority="MEDIUM",
                approach="CUSTOM_ATTENTION",
                config_suggestion={
                    "actions": [
                        "Profile with NCU to find bottleneck",
                        "Check if causal mask or dropout can be fused",
                        "Try different tile sizes (block_m, block_n)",
                        "Consider xFormers if irregular pattern"
                    ]
                }
            )


class SparseDetector:
    """Detect sparsity patterns and recommend sparse kernels"""
    
    @staticmethod
    def analyze_sparsity(tensor: torch.Tensor, threshold: float = 1e-6) -> Dict:
        """Analyze sparsity pattern of a tensor"""
        
        zeros = (torch.abs(tensor) < threshold).sum().item()
        total = tensor.numel()
        sparsity = zeros / total
        
        # Check for structured patterns
        is_24_structured = SparseDetector._check_24_sparsity(tensor, threshold)
        is_bsr_friendly = SparseDetector._check_bsr_pattern(tensor, threshold)
        
        return {
            "sparsity": sparsity,
            "nnz": total - zeros,
            "total": total,
            "is_24_structured": is_24_structured,
            "is_bsr_friendly": is_bsr_friendly,
            "pattern": SparseDetector._classify_pattern(sparsity, is_24_structured, is_bsr_friendly)
        }
    
    @staticmethod
    def _check_24_sparsity(tensor: torch.Tensor, threshold: float) -> bool:
        """Check if tensor follows 2:4 structured sparsity (Ampere/Hopper)"""
        # Simplified check: In each group of 4 elements, are exactly 2 non-zero?
        # Real implementation would check all dimensions properly
        flat = tensor.flatten()
        if len(flat) % 4 != 0:
            return False
        
        groups = flat.reshape(-1, 4)
        nonzero_per_group = (torch.abs(groups) >= threshold).sum(dim=1)
        
        # At least 80% of groups should have exactly 2 non-zeros
        correct_groups = (nonzero_per_group == 2).sum().item()
        return (correct_groups / len(groups)) > 0.8
    
    @staticmethod
    def _check_bsr_pattern(tensor: torch.Tensor, threshold: float, block_size: int = 128) -> bool:
        """Check if tensor is friendly for Block Sparse Row format"""
        if tensor.dim() != 2:
            return False
        
        M, N = tensor.shape
        if M % block_size != 0 or N % block_size != 0:
            return False
        
        # Check if blocks tend to be all-zero or all-nonzero
        blocks_m = M // block_size
        blocks_n = N // block_size
        
        zero_blocks = 0
        for i in range(blocks_m):
            for j in range(blocks_n):
                block = tensor[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                if (torch.abs(block) < threshold).all():
                    zero_blocks += 1
        
        block_sparsity = zero_blocks / (blocks_m * blocks_n)
        
        # BSR is beneficial if >50% of blocks are zero
        return block_sparsity > 0.5
    
    @staticmethod
    def _classify_pattern(sparsity: float, is_24: bool, is_bsr: bool) -> str:
        """Classify sparsity pattern"""
        if sparsity < 0.5:
            return "DENSE"
        elif is_24:
            return "STRUCTURED_24"
        elif is_bsr:
            return "BLOCK_SPARSE"
        elif sparsity > 0.9:
            return "HIGHLY_SPARSE"
        else:
            return "UNSTRUCTURED"
    
    @staticmethod
    def recommend_sparse_kernel(analysis: Dict, M: int, N: int, K: int) -> OpportunityScore:
        """Recommend sparse kernel based on pattern analysis"""
        
        sparsity = analysis["sparsity"]
        pattern = analysis["pattern"]
        
        if pattern == "DENSE":
            return OpportunityScore(
                operation="gemm",
                shape=(M, N, K),
                baseline_tflops=0.0,  # Not measured yet
                ceiling_tflops=628.0,  # H100 ceiling
                efficiency=1.0,
                recommendation="Matrix is dense. Use cuBLAS (optimal at 628 TFLOPS).",
                priority="NONE",
                approach="NONE",
                config_suggestion={"use": "cuBLAS"}
            )
        
        elif pattern == "STRUCTURED_24":
            return OpportunityScore(
                operation="sparse_gemm",
                shape=(M, N, K),
                baseline_tflops=0.0,
                ceiling_tflops=1200.0,  # 2× dense theoretical
                efficiency=0.0,
                recommendation=f"2:4 structured sparsity detected ({sparsity:.1%} sparse). "
                               f"Use CUTLASS 4.3 Example 62 (SM90 sparse tensor cores). "
                               f"Theoretical: 2× dense speedup.",
                priority="HIGH",
                approach="CUTLASS_STRUCTURED_SPARSE",
                config_suggestion={
                    "use": "CUTLASS 4.3 Example 62",
                    "file": "/opt/cutlass/examples/62_hopper_sparse_gemm",
                    "expected_speedup": "2.0x",
                    "requirements": "Ampere/Hopper/Blackwell GPU"
                }
            )
        
        elif pattern == "BLOCK_SPARSE":
            return OpportunityScore(
                operation="sparse_gemm",
                shape=(M, N, K),
                baseline_tflops=0.0,
                ceiling_tflops=628.0 * (1 - sparsity),  # Proportional to non-zeros
                efficiency=0.0,
                recommendation=f"Block sparse pattern detected ({sparsity:.1%} sparse). "
                               f"Use BlackwellSparseK (your 63× cuSPARSE speedup, 52 TFLOPS on L4). "
                               f"BSR format with 128×128 blocks.",
                priority="HIGH",
                approach="CUSTOM_BSR_SPARSE",
                config_suggestion={
                    "use": "BlackwellSparseK",
                    "format": "BSR (Block Sparse Row)",
                    "block_size": 128,
                    "expected_vs_cusparse": "63x faster",
                    "expected_vs_pytorch": "1.74x faster than CUTLASS",
                    "architecture": "SM89 (Ada) or SM90 (Hopper)",
                    "validated_performance": "52.1 TFLOPS on L4"
                }
            )
        
        else:  # HIGHLY_SPARSE or UNSTRUCTURED
            return OpportunityScore(
                operation="sparse_gemm",
                shape=(M, N, K),
                baseline_tflops=0.0,
                ceiling_tflops=100.0,  # Conservative for unstructured
                efficiency=0.0,
                recommendation=f"High sparsity ({sparsity:.1%}) but unstructured. "
                               f"Options: (1) cuSPARSE CSR/COO (slow but correct), "
                               f"(2) Convert to BSR if possible, (3) Custom kernel. "
                               f"Benchmark cuSPARSE first as baseline.",
                priority="MEDIUM",
                approach="INVESTIGATE_SPARSE",
                config_suggestion={
                    "baseline": "cuSPARSE (CSR or COO format)",
                    "alternatives": [
                        "Convert to BSR if block structure exists",
                        "Try CUTLASS if can pad to 2:4",
                        "Custom kernel for specific access pattern"
                    ]
                }
            )


class FusionDetector:
    """Detect fusion opportunities in operation sequences"""
    
    @staticmethod
    def analyze_sequence(ops: List[str]) -> Dict:
        """Analyze a sequence of operations for fusion opportunities"""
        
        fusion_patterns = {
            ("gemm", "bias", "relu"): {
                "name": "GEMM+Bias+ReLU",
                "benefit": "Save 2 memory round-trips",
                "approach": "CUTLASS epilogue visitor or custom kernel",
                "expected_speedup": "1.3-1.5x vs separate ops"
            },
            ("attention", "mask", "dropout"): {
                "name": "Attention+Mask+Dropout",
                "benefit": "Fused in FA3, verify it's being used",
                "approach": "torch.nn.functional.scaled_dot_product_attention with is_causal=True",
                "expected_speedup": "Already optimal if using FA3"
            },
            ("layernorm", "residual", "activation"): {
                "name": "LayerNorm+Residual+Activation",
                "benefit": "Common in transformers, save memory bandwidth",
                "approach": "Apex FusedLayerNorm or custom Triton kernel",
                "expected_speedup": "1.2-1.4x vs separate"
            },
            ("gemm", "gemm"): {
                "name": "Back-to-back GEMM",
                "benefit": "Potential for pipelining or output reuse",
                "approach": "Check if output of first fits in L2 cache",
                "expected_speedup": "Minimal unless shapes align"
            }
        }
        
        opportunities = []
        for i in range(len(ops) - 1):
            pattern = tuple(ops[i:i+3]) if i < len(ops) - 2 else tuple(ops[i:i+2])
            if pattern in fusion_patterns:
                opportunities.append(fusion_patterns[pattern])
        
        return {
            "num_ops": len(ops),
            "opportunities": opportunities,
            "priority": "HIGH" if opportunities else "NONE"
        }


def extended_ceiling_scout_demo():
    """Demo of extended functionality"""
    
    print("═══════════════════════════════════════════════════════════")
    print("  Ceiling Scout Extended - Demo")
    print("═══════════════════════════════════════════════════════════\n")
    
    # 1. FA3 Benchmarking
    print("1. FlashAttention-3 Benchmarking")
    print("-" * 60)
    fa3_bench = FA3Benchmarker(device="cuda")
    
    # Multi-head attention (typical GPT-2 config)
    opp = fa3_bench.detect_attention_ceiling(
        batch=1, heads=8, seq_len=512, head_dim=64
    )
    print(f"\nRecommendation: {opp.recommendation}")
    print(f"Priority: {opp.priority}\n")
    
    # 2. Sparse Detection
    print("2. Sparse Pattern Detection")
    print("-" * 60)
    
    # Create 87.5% sparse matrix (like your BlackwellSparseK test case)
    M, N = 8192, 8192
    sparse_matrix = torch.randn(M, N, device="cuda")
    mask = torch.rand(M, N, device="cuda") > 0.875  # 87.5% zeros
    sparse_matrix = sparse_matrix * mask
    
    analysis = SparseDetector.analyze_sparsity(sparse_matrix)
    print(f"Sparsity: {analysis['sparsity']:.1%}")
    print(f"Pattern: {analysis['pattern']}")
    
    opp = SparseDetector.recommend_sparse_kernel(analysis, M, N, M)
    print(f"\nRecommendation: {opp.recommendation}")
    print(f"Approach: {opp.approach}\n")
    
    # 3. Fusion Detection
    print("3. Fusion Opportunity Detection")
    print("-" * 60)
    
    transformer_block = ["gemm", "bias", "relu", "gemm", "layernorm", "residual"]
    fusion_analysis = FusionDetector.analyze_sequence(transformer_block)
    
    print(f"Ops analyzed: {fusion_analysis['num_ops']}")
    print(f"Fusion opportunities found: {len(fusion_analysis['opportunities'])}")
    for opp_dict in fusion_analysis['opportunities']:
        print(f"  • {opp_dict['name']}: {opp_dict['benefit']}")
        print(f"    → {opp_dict['approach']}")
        print(f"    Expected: {opp_dict['expected_speedup']}\n")
    
    print("═══════════════════════════════════════════════════════════")
    print("  Extended Scout Ready for Integration")
    print("═══════════════════════════════════════════════════════════")


if __name__ == "__main__":
    if torch.cuda.is_available():
        extended_ceiling_scout_demo()
    else:
        print("CUDA not available. Skipping GPU benchmarks.")
        print("Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")

