# cudadent42/bench/common/memory_tracker.py
"""
GPU Memory Tracking Module

Provides utilities to track CUDA memory usage during benchmark execution.
Supports both decorator and context manager patterns.

Usage as context manager:
    from cudadent42.bench.common.memory_tracker import MemoryTracker
    
    with MemoryTracker() as tracker:
        # Your code here
        pass
    
    print(f"Peak memory: {tracker.peak_mb:.2f} MB")
"""

import torch
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'allocated_mb': self.allocated_mb,
            'reserved_mb': self.reserved_mb,
            'peak_allocated_mb': self.peak_allocated_mb,
            'peak_reserved_mb': self.peak_reserved_mb
        }


class MemoryTracker:
    """
    Context manager for tracking GPU memory usage.
    
    Example:
        with MemoryTracker() as tracker:
            # Run your CUDA code
            tensor = torch.randn(1000, 1000, device='cuda')
        
        print(f"Peak memory: {tracker.peak_mb:.2f} MB")
        print(f"Current memory: {tracker.current_mb:.2f} MB")
    """
    
    def __init__(self, device: Optional[int] = None, reset_peak: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            device: CUDA device ID (None = current device)
            reset_peak: Reset peak memory stats on entry
        """
        self.device = device if device is not None else torch.cuda.current_device()
        self.reset_peak = reset_peak
        
        self.initial_allocated = 0.0
        self.initial_reserved = 0.0
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0
        self.final_allocated = 0.0
        self.final_reserved = 0.0
    
    def __enter__(self):
        """Enter context manager"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for memory tracking")
        
        # Synchronize and clear cache
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        
        # Reset peak stats if requested
        if self.reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Record initial state
        self.initial_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        self.initial_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        # Synchronize before reading final stats
        torch.cuda.synchronize(self.device)
        
        # Record final and peak states
        self.final_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        self.final_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        self.peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        self.peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        
        # Don't suppress exceptions
        return False
    
    @property
    def peak_mb(self) -> float:
        """Peak allocated memory in MB"""
        return self.peak_allocated
    
    @property
    def current_mb(self) -> float:
        """Current allocated memory in MB"""
        return self.final_allocated
    
    @property
    def delta_mb(self) -> float:
        """Change in allocated memory in MB"""
        return self.final_allocated - self.initial_allocated
    
    def get_stats(self) -> MemoryStats:
        """Get complete memory statistics"""
        return MemoryStats(
            allocated_mb=self.final_allocated,
            reserved_mb=self.final_reserved,
            peak_allocated_mb=self.peak_allocated,
            peak_reserved_mb=self.peak_reserved
        )
    
    def summary(self) -> str:
        """Get human-readable summary"""
        return (
            f"Memory Usage:\n"
            f"  Initial: {self.initial_allocated:.2f} MB\n"
            f"  Final:   {self.final_allocated:.2f} MB\n"
            f"  Delta:   {self.delta_mb:+.2f} MB\n"
            f"  Peak:    {self.peak_allocated:.2f} MB"
        )


def check_oom_risk(peak_mb: float, total_mb: float, warn_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Check if memory usage is approaching OOM risk.
    
    Args:
        peak_mb: Peak memory used in MB
        total_mb: Total GPU memory in MB
        warn_threshold: Fraction of total memory to trigger warning (default 0.85)
    
    Returns:
        Dict with risk assessment:
        {
            'is_safe': bool,
            'usage_fraction': float,
            'peak_mb': float,
            'available_mb': float,
            'warning': str or None
        }
    
    Example:
        >>> risk = check_oom_risk(peak_mb=20000, total_mb=24000)
        >>> if not risk['is_safe']:
        ...     print(risk['warning'])
    """
    usage_fraction = peak_mb / total_mb
    available_mb = total_mb - peak_mb
    
    is_safe = usage_fraction < warn_threshold
    warning = None
    
    if not is_safe:
        warning = (
            f"⚠️  High memory usage: {usage_fraction*100:.1f}% ({peak_mb:.0f}/{total_mb:.0f} MB). "
            f"Only {available_mb:.0f} MB available. OOM risk!"
        )
    
    return {
        'is_safe': is_safe,
        'usage_fraction': usage_fraction,
        'peak_mb': peak_mb,
        'total_mb': total_mb,
        'available_mb': available_mb,
        'warning': warning
    }


def get_gpu_memory_info(device: Optional[int] = None) -> Dict[str, float]:
    """
    Get current GPU memory information.
    
    Args:
        device: CUDA device ID (None = current device)
    
    Returns:
        Dict with memory info in MB:
        {
            'allocated': float,
            'reserved': float,
            'free': float,
            'total': float
        }
    """
    if not torch.cuda.is_available():
        return {
            'allocated': 0.0,
            'reserved': 0.0,
            'free': 0.0,
            'total': 0.0
        }
    
    device = device if device is not None else torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    free = total - allocated
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total
    }


if __name__ == "__main__":
    # Test module
    print("Testing memory tracker module...\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping test")
    else:
        print("Test 1: Context manager")
        with MemoryTracker() as tracker:
            # Allocate some memory
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = x @ y
        
        print(tracker.summary())
        print()
        
        print("Test 2: Memory info")
        info = get_gpu_memory_info()
        print(f"GPU Memory:")
        print(f"  Total:     {info['total']:.2f} MB")
        print(f"  Allocated: {info['allocated']:.2f} MB")
        print(f"  Free:      {info['free']:.2f} MB")
        print()
        
        print("Test 3: OOM risk check")
        risk = check_oom_risk(info['allocated'], info['total'])
        print(f"Safe: {risk['is_safe']}")
        print(f"Usage: {risk['usage_fraction']*100:.1f}%")
        if risk['warning']:
            print(risk['warning'])
    
    print("\n✓ Module test complete")

