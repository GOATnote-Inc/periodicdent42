"""
robust-kbench: Robust CUDA Kernel Micro-Benchmarking
Phase 1: Tool Integration

Repeatable, statistically-rigorous kernel benchmarking with multi-shape testing.
"""

__version__ = "0.1.0"
__commit__ = "initial"

from .runner import BenchmarkRunner, ShapeConfig
from .reporter import BenchmarkReporter
from .config import RBKConfig

__all__ = [
    "BenchmarkRunner",
    "ShapeConfig",
    "BenchmarkReporter",
    "RBKConfig",
]

