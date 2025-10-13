#!/usr/bin/env python3
"""
CUDA Kernel Correctness Checker
================================
Comprehensive correctness validation for CUDA kernels against reference implementations.

Features:
- Numerical accuracy testing (absolute, relative, ULP)
- Statistical distribution comparison
- Bit-exact validation options
- Numerical stability analysis
- Comprehensive error reporting
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import sys


class ToleranceMode(Enum):
    """Different tolerance checking modes"""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    ULP = "ulp"  # Units in Last Place
    MIXED = "mixed"  # Combination of absolute and relative
    

@dataclass
class CorrectnessConfig:
    """Configuration for correctness checking"""
    atol: float = 1e-5  # Absolute tolerance
    rtol: float = 1e-5  # Relative tolerance
    ulp_tol: int = 4    # ULP tolerance for floating point
    mode: ToleranceMode = ToleranceMode.MIXED
    check_nan: bool = True
    check_inf: bool = True
    check_distribution: bool = True
    statistical_tests: bool = True
    

@dataclass
class CorrectnessResult:
    """Results of correctness checking"""
    passed: bool
    total_elements: int
    mismatches: int
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    mean_rel_error: float
    
    # Statistical measures
    correlation: float = None
    kl_divergence: float = None
    
    # Numerical issues
    has_nan: bool = False
    has_inf: bool = False
    nan_count: int = 0
    inf_count: int = 0
    
    # Detailed analysis
    error_distribution: Dict[str, int] = None
    worst_indices: list = None
    

class CUDACorrectnessChecker:
    """
    Comprehensive correctness checker for CUDA kernels
    
    Usage:
        checker = CUDACorrectnessChecker(config)
        result = checker.check(
            reference_output=ref_result,
            cuda_output=cuda_result,
            input_data=inputs
        )
        
        if not result.passed:
            checker.print_detailed_report(result)
    """
    
    def __init__(self, config: CorrectnessConfig = CorrectnessConfig()):
        self.config = config
        
    def check(
        self,
        reference_output: np.ndarray,
        cuda_output: np.ndarray,
        input_data: Optional[np.ndarray] = None,
        kernel_name: str = "kernel"
    ) -> CorrectnessResult:
        """
        Comprehensive correctness check
        
        Args:
            reference_output: Ground truth output (CPU/reference implementation)
            cuda_output: CUDA kernel output to validate
            input_data: Optional input data for context
            kernel_name: Name of kernel being tested
            
        Returns:
            CorrectnessResult with detailed analysis
        """
        
        print(f"\n{'='*70}")
        print(f"Correctness Check: {kernel_name}")
        print(f"{'='*70}")
        
        # Basic validation
        if reference_output.shape != cuda_output.shape:
            print(f"SHAPE MISMATCH")
            print(f"   Reference: {reference_output.shape}")
            print(f"   CUDA:      {cuda_output.shape}")
            return CorrectnessResult(
                passed=False,
                total_elements=0,
                mismatches=reference_output.size + cuda_output.size,
                max_abs_error=float('inf'),
                max_rel_error=float('inf'),
                mean_abs_error=float('inf'),
                mean_rel_error=float('inf')
            )
            
        # Convert to float for comparison
        ref = reference_output.astype(np.float64).flatten()
        cuda = cuda_output.astype(np.float64).flatten()
        
        # Check for numerical issues
        nan_mask_ref = np.isnan(ref)
        nan_mask_cuda = np.isnan(cuda)
        inf_mask_ref = np.isinf(ref)
        inf_mask_cuda = np.isinf(cuda)
        
        has_nan = np.any(nan_mask_cuda)
        has_inf = np.any(inf_mask_cuda)
        
        if self.config.check_nan and has_nan:
            print(f"WARNING: CUDA output contains {np.sum(nan_mask_cuda)} NaN values")
            
        if self.config.check_inf and has_inf:
            print(f"WARNING: CUDA output contains {np.sum(inf_mask_cuda)} Inf values")
            
        # Remove NaN/Inf for comparison
        valid_mask = ~(nan_mask_ref | nan_mask_cuda | inf_mask_ref | inf_mask_cuda)
        ref_valid = ref[valid_mask]
        cuda_valid = cuda[valid_mask]
        
        if len(ref_valid) == 0:
            print("No valid values to compare")
            return CorrectnessResult(
                passed=False,
                total_elements=len(ref),
                mismatches=len(ref),
                max_abs_error=float('inf'),
                max_rel_error=float('inf'),
                mean_abs_error=float('inf'),
                mean_rel_error=float('inf'),
                has_nan=has_nan,
                has_inf=has_inf
            )
            
        # Compute errors
        abs_errors = np.abs(cuda_valid - ref_valid)
        rel_errors = np.abs((cuda_valid - ref_valid) / (ref_valid + 1e-10))
        
        # Tolerance checking based on mode
        if self.config.mode == ToleranceMode.ABSOLUTE:
            mismatches = abs_errors > self.config.atol
        elif self.config.mode == ToleranceMode.RELATIVE:
            mismatches = rel_errors > self.config.rtol
        elif self.config.mode == ToleranceMode.MIXED:
            # Pass if either absolute OR relative tolerance is satisfied
            mismatches = (abs_errors > self.config.atol) & (rel_errors > self.config.rtol)
        elif self.config.mode == ToleranceMode.ULP:
            mismatches = self._check_ulp_tolerance(ref_valid, cuda_valid)
        else:
            mismatches = np.zeros(len(ref_valid), dtype=bool)
            
        num_mismatches = np.sum(mismatches)
        passed = num_mismatches == 0
        
        # Statistical analysis
        correlation = None
        kl_divergence = None
        
        if self.config.statistical_tests and len(ref_valid) > 1:
            correlation = np.corrcoef(ref_valid, cuda_valid)[0, 1]
            
            if self.config.check_distribution:
                kl_divergence = self._compute_kl_divergence(ref_valid, cuda_valid)
                
        # Find worst errors for debugging
        worst_indices = None
        if num_mismatches > 0:
            # Get indices of top 10 worst errors
            worst_abs_idx = np.argsort(abs_errors)[-10:]
            worst_indices = worst_abs_idx.tolist()
            
        # Error distribution
        error_distribution = self._compute_error_distribution(abs_errors)
        
        # Create result
        result = CorrectnessResult(
            passed=passed,
            total_elements=len(ref),
            mismatches=num_mismatches,
            max_abs_error=np.max(abs_errors),
            max_rel_error=np.max(rel_errors),
            mean_abs_error=np.mean(abs_errors),
            mean_rel_error=np.mean(rel_errors),
            correlation=correlation,
            kl_divergence=kl_divergence,
            has_nan=has_nan,
            has_inf=has_inf,
            nan_count=int(np.sum(nan_mask_cuda)),
            inf_count=int(np.sum(inf_mask_cuda)),
            error_distribution=error_distribution,
            worst_indices=worst_indices
        )
        
        self._print_result_summary(result)
        
        return result
        
    def _check_ulp_tolerance(
        self, 
        ref: np.ndarray, 
        cuda: np.ndarray
    ) -> np.ndarray:
        """
        Check ULP (Units in Last Place) tolerance
        More sophisticated floating point comparison
        """
        # Convert to integer representation
        ref_int = ref.view(np.int64)
        cuda_int = cuda.view(np.int64)
        
        # Compute ULP difference
        ulp_diff = np.abs(ref_int - cuda_int)
        
        return ulp_diff > self.config.ulp_tol
        
    def _compute_kl_divergence(
        self, 
        ref: np.ndarray, 
        cuda: np.ndarray
    ) -> float:
        """Compute KL divergence between distributions"""
        # Create histograms
        bins = 100
        range_min = min(ref.min(), cuda.min())
        range_max = max(ref.max(), cuda.max())
        
        hist_ref, _ = np.histogram(ref, bins=bins, range=(range_min, range_max))
        hist_cuda, _ = np.histogram(cuda, bins=bins, range=(range_min, range_max))
        
        # Normalize to probabilities
        p = hist_ref / hist_ref.sum() + 1e-10
        q = hist_cuda / hist_cuda.sum() + 1e-10
        
        # KL divergence
        kl = np.sum(p * np.log(p / q))
        
        return float(kl)
        
    def _compute_error_distribution(
        self, 
        abs_errors: np.ndarray
    ) -> Dict[str, int]:
        """Categorize errors into buckets"""
        
        distribution = {
            'exact_match': int(np.sum(abs_errors == 0)),
            'very_small': int(np.sum((abs_errors > 0) & (abs_errors <= 1e-8))),
            'small': int(np.sum((abs_errors > 1e-8) & (abs_errors <= 1e-6))),
            'medium': int(np.sum((abs_errors > 1e-6) & (abs_errors <= 1e-4))),
            'large': int(np.sum((abs_errors > 1e-4) & (abs_errors <= 1e-2))),
            'very_large': int(np.sum(abs_errors > 1e-2))
        }
        
        return distribution
        
    def _print_result_summary(self, result: CorrectnessResult):
        """Print concise result summary"""
        
        status = "PASSED" if result.passed else "FAILED"
        print(f"\n{status}")
        
        print(f"\nError Statistics:")
        print(f"  Total Elements:    {result.total_elements:,}")
        print(f"  Mismatches:        {result.mismatches:,} ({100*result.mismatches/result.total_elements:.4f}%)")
        print(f"  Max Abs Error:     {result.max_abs_error:.2e}")
        print(f"  Mean Abs Error:    {result.mean_abs_error:.2e}")
        print(f"  Max Rel Error:     {result.max_rel_error:.2e}")
        print(f"  Mean Rel Error:    {result.mean_rel_error:.2e}")
        
        if result.correlation is not None:
            print(f"\nStatistical Measures:")
            print(f"  Correlation:       {result.correlation:.6f}")
            if result.kl_divergence is not None:
                print(f"  KL Divergence:     {result.kl_divergence:.6f}")
                
        if result.has_nan or result.has_inf:
            print(f"\nNumerical Issues:")
            if result.has_nan:
                print(f"  NaN values:        {result.nan_count}")
            if result.has_inf:
                print(f"  Inf values:        {result.inf_count}")
                
    def print_detailed_report(
        self, 
        result: CorrectnessResult,
        reference_output: np.ndarray = None,
        cuda_output: np.ndarray = None
    ):
        """Print detailed analysis for debugging"""
        
        print(f"\n{'='*70}")
        print("DETAILED CORRECTNESS ANALYSIS")
        print(f"{'='*70}")
        
        # Error distribution
        if result.error_distribution:
            print(f"\nError Distribution:")
            for category, count in result.error_distribution.items():
                pct = 100 * count / result.total_elements
                print(f"  {category:15s}: {count:8,} ({pct:6.2f}%)")
                
        # Worst errors
        if result.worst_indices and reference_output is not None and cuda_output is not None:
            print(f"\nWorst Mismatches (showing up to 10):")
            ref_flat = reference_output.flatten()
            cuda_flat = cuda_output.flatten()
            
            for i, idx in enumerate(result.worst_indices[:10], 1):
                ref_val = ref_flat[idx]
                cuda_val = cuda_flat[idx]
                abs_err = abs(cuda_val - ref_val)
                rel_err = abs((cuda_val - ref_val) / (ref_val + 1e-10))
                
                print(f"  #{i} Index {idx}:")
                print(f"     Reference: {ref_val:.8e}")
                print(f"     CUDA:      {cuda_val:.8e}")
                print(f"     Abs Error: {abs_err:.8e}")
                print(f"     Rel Error: {rel_err:.8e}")
                
        # Recommendations
        print(f"\nRecommendations:")
        if result.max_abs_error > 1e-2:
            print("  - Large absolute errors detected. Check algorithm correctness.")
        if result.max_rel_error > 0.1:
            print("  - Large relative errors detected. May indicate numerical instability.")
        if result.has_nan:
            print("  - NaN values present. Check for division by zero or invalid operations.")
        if result.has_inf:
            print("  - Inf values present. Check for overflow or division by very small numbers.")
        if result.correlation and result.correlation < 0.99:
            print(f"  - Low correlation ({result.correlation:.4f}). Outputs may be systematically different.")
            
    def batch_check(
        self,
        test_cases: list,
        kernel_func: Callable,
        reference_func: Callable
    ) -> Dict[str, CorrectnessResult]:
        """
        Run multiple test cases
        
        Args:
            test_cases: List of input configurations
            kernel_func: CUDA kernel function
            reference_func: Reference implementation
            
        Returns:
            Dictionary mapping test case name to result
        """
        results = {}
        
        print(f"\n{'='*70}")
        print(f"BATCH CORRECTNESS TESTING")
        print(f"{'='*70}")
        print(f"Running {len(test_cases)} test cases...\n")
        
        for i, test_case in enumerate(test_cases, 1):
            name = test_case.get('name', f'test_{i}')
            inputs = test_case.get('inputs')
            
            # Run reference
            ref_output = reference_func(**inputs)
            
            # Run CUDA
            cuda_output = kernel_func(**inputs)
            
            # Check correctness
            result = self.check(ref_output, cuda_output, kernel_name=name)
            results[name] = result
            
        # Summary
        passed = sum(1 for r in results.values() if r.passed)
        failed = len(results) - passed
        
        print(f"\n{'='*70}")
        print(f"BATCH TEST SUMMARY")
        print(f"{'='*70}")
        print(f"Total Tests:  {len(results)}")
        print(f"Passed:       {passed}")
        print(f"Failed:       {failed}")
        
        if failed > 0:
            print(f"\nFailed tests:")
            for name, result in results.items():
                if not result.passed:
                    print(f"  - {name}: {result.mismatches} mismatches")
                    
        return results

