"""
EvoEngineer: Evolutionary CUDA Kernel Parameter Optimization
Phase 1: Tool Integration

Systematic parameter search using evolutionary strategies for CUDA kernel tuning.
Supports L4 (sm_89) with focus on FlashAttention optimization.
"""

__version__ = "0.1.0"
__commit__ = "initial"  # Will be updated when this becomes a proper git submodule

from .optimizer import KernelOptimizer, SearchSpace, Candidate
from .evaluator import BenchmarkEvaluator, CorrectnessGate
from .mutator import ParameterMutator

__all__ = [
    "KernelOptimizer",
    "SearchSpace",
    "Candidate",
    "BenchmarkEvaluator",
    "CorrectnessGate",
    "ParameterMutator",
]

