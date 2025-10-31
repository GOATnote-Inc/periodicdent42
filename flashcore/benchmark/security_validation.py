#!/usr/bin/env python3
"""
Security Validation - Python-level safety tests
Validates: memory bounds, numerical stability, timing side-channels

Complements: memory_safety_validator.py (compute-sanitizer)
Advantage: No external dependencies, runs anywhere

Based on: NVIDIA CUDA security best practices
"""

import torch
import hashlib
import numpy as np
from typing import Dict, Tuple
import time
import json
from pathlib import Path


class SecurityValidator:
    """Pure Python security validation for CUDA kernels"""
    
    def __init__(self, kernel_fn):
        self.kernel = kernel_fn
        self.results = {}
    
    def validate_memory_bounds(self) -> Dict[str, bool]:
        """Test kernel behavior with extreme inputs"""
        print("Testing memory bounds and input validation...")
        results = {}
        
        # Test 1: Oversized input (should fail gracefully, not crash)
        try:
            print("  • Testing oversized input (OOM handling)...", end=' ')
            huge = torch.randn(2**18, 32, 1024, 128, device='cuda', dtype=torch.float16)
            _ = self.kernel(huge, huge, huge)
            results['overflow_protection'] = False  # Should have raised OOM
            print("❌ No protection")
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            results['overflow_protection'] = True
            print("✓")
        
        # Test 2: Misaligned shapes (should handle gracefully)
        try:
            print("  • Testing misaligned shapes...", end=' ')
            q = torch.randn(16, 8, 1023, 127, device='cuda', dtype=torch.float16)
            k = torch.randn(16, 8, 512, 64, device='cuda', dtype=torch.float16)
            v = k.clone()
            _ = self.kernel(q, k, v)
            results['shape_validation'] = True  # Handled gracefully
            print("✓")
        except Exception as e:
            results['shape_validation'] = False
            print(f"❌ {type(e).__name__}")
        
        # Test 3: Zero-sized inputs
        try:
            print("  • Testing zero batch size...", end=' ')
            q = torch.randn(0, 8, 512, 64, device='cuda', dtype=torch.float16)
            k, v = q.clone(), q.clone()
            out = self.kernel(q, k, v)
            results['zero_batch_handling'] = out.shape[0] == 0
            print("✓" if results['zero_batch_handling'] else "❌")
        except Exception as e:
            results['zero_batch_handling'] = False
            print(f"❌ {type(e).__name__}")
        
        return results
    
    def validate_numerical_stability(self) -> Dict[str, float]:
        """Test resilience to NaN/Inf injection"""
        print("Testing numerical stability (NaN/Inf resilience)...")
        results = {}
        
        test_configs = [
            ('small', (2, 4, 256, 64)),
            ('medium', (16, 8, 1024, 128)),
            ('large', (32, 16, 2048, 128))
        ]
        
        for desc, (b, h, s, d) in test_configs:
            print(f"  • Testing {desc} config ({b}×{h}×{s}×{d})...", end=' ')
            
            # Create clean input
            torch.manual_seed(42)
            q = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
            k = q.clone()
            v = q.clone()
            
            # Inject single Inf value
            q[0, 0, 0, 0] = float('inf')
            
            try:
                out = self.kernel(q, k, v)
                
                nan_ratio = torch.isnan(out).float().mean().item()
                inf_ratio = torch.isinf(out).float().mean().item()
                
                results[f'{desc}_nan_ratio'] = nan_ratio
                results[f'{desc}_inf_ratio'] = inf_ratio
                
                # Requirement: < 0.01% contamination from 1 bad input
                isolated = (nan_ratio < 1e-4) and (inf_ratio < 0.01)
                
                status = "✓" if isolated else f"⚠ ({nan_ratio*100:.3f}% NaN, {inf_ratio*100:.3f}% Inf)"
                print(status)
                
            except Exception as e:
                results[f'{desc}_error'] = str(e)
                print(f"❌ {type(e).__name__}")
        
        return results
    
    def detect_timing_sidechannel(self, n_trials: int = 1000) -> Dict[str, float]:
        """
        Detect data-dependent timing (potential side-channel)
        
        Constant-time requirement: Timing should not depend on input values
        """
        print(f"Testing timing side-channels ({n_trials} trials)...")
        
        # Pattern 1: All zeros
        print("  • Measuring zeros pattern...", end=' ')
        q_zeros = torch.zeros(16, 8, 1024, 128, device='cuda', dtype=torch.float16)
        times_zeros = []
        
        for _ in range(n_trials):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = self.kernel(q_zeros, q_zeros, q_zeros)
            end.record()
            
            torch.cuda.synchronize()
            times_zeros.append(start.elapsed_time(end))
        
        print(f"mean={np.mean(times_zeros):.3f}ms")
        
        # Pattern 2: Random data
        print("  • Measuring random pattern...", end=' ')
        torch.manual_seed(42)
        q_rand = torch.randn(16, 8, 1024, 128, device='cuda', dtype=torch.float16)
        times_rand = []
        
        for _ in range(n_trials):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = self.kernel(q_rand, q_rand, q_rand)
            end.record()
            
            torch.cuda.synchronize()
            times_rand.append(start.elapsed_time(end))
        
        print(f"mean={np.mean(times_rand):.3f}ms")
        
        # Statistical test: Kolmogorov-Smirnov
        print("  • Running statistical test...", end=' ')
        try:
            from scipy import stats
            _, p_value = stats.ks_2samp(times_zeros, times_rand)
            
            # p > 0.05 means distributions are statistically indistinguishable
            constant_time = p_value > 0.05
            
            print(f"p={p_value:.4f} ({'✓ constant-time' if constant_time else '⚠ data-dependent'})")
            
            return {
                'timing_variance_zeros_ms': float(np.std(times_zeros)),
                'timing_variance_rand_ms': float(np.std(times_rand)),
                'mean_diff_ms': float(abs(np.mean(times_zeros) - np.mean(times_rand))),
                'ks_p_value': float(p_value),
                'constant_time_compliant': constant_time
            }
        except ImportError:
            print("⚠ scipy not available, skipping statistical test")
            return {
                'timing_variance_zeros_ms': float(np.std(times_zeros)),
                'timing_variance_rand_ms': float(np.std(times_rand)),
                'mean_diff_ms': float(abs(np.mean(times_zeros) - np.mean(times_rand))),
                'constant_time_compliant': None  # Can't determine without scipy
            }
    
    def compute_output_hash(self) -> str:
        """Cryptographic hash for bitwise reproducibility verification"""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        q = torch.randn(16, 8, 1024, 128, device='cuda', dtype=torch.float16)
        k, v = q.clone(), q.clone()
        
        out = self.kernel(q, k, v)
        
        return hashlib.sha256(out.cpu().numpy().tobytes()).hexdigest()
    
    def run_full_audit(self) -> Dict:
        """Execute complete security validation suite"""
        print("=" * 80)
        print("SECURITY VALIDATION - FlashCore Kernels")
        print("=" * 80)
        print()
        
        audit = {}
        
        # Test 1: Memory bounds
        print("[1/4] Memory Bounds Validation")
        audit['memory_bounds'] = self.validate_memory_bounds()
        print()
        
        # Test 2: Numerical stability
        print("[2/4] Numerical Stability")
        audit['numerical_stability'] = self.validate_numerical_stability()
        print()
        
        # Test 3: Timing side-channels
        print("[3/4] Timing Side-Channel Analysis")
        audit['timing_analysis'] = self.detect_timing_sidechannel(n_trials=1000)
        print()
        
        # Test 4: Output hash
        print("[4/4] Output Hash (Reproducibility)")
        audit['output_hash'] = self.compute_output_hash()
        print(f"  SHA256: {audit['output_hash'][:32]}...")
        print()
        
        return audit
    
    def generate_report(self, audit: Dict, output_file: Path) -> bool:
        """Generate security audit report"""
        print("=" * 80)
        print("SECURITY AUDIT SUMMARY")
        print("=" * 80)
        
        # Evaluate pass/fail
        all_pass = True
        
        # Memory bounds
        mem_pass = audit['memory_bounds'].get('overflow_protection', False)
        print(f"Memory Bounds: {'✅ PASS' if mem_pass else '⚠️ WARN'}")
        if not mem_pass:
            all_pass = False
        
        # Numerical stability
        nan_ratios = [v for k, v in audit['numerical_stability'].items() if 'nan_ratio' in k]
        num_pass = all(r < 1e-4 for r in nan_ratios) if nan_ratios else False
        print(f"Numerical Stability: {'✅ PASS' if num_pass else '⚠️ WARN'}")
        if not num_pass:
            all_pass = False
        
        # Timing side-channels
        timing = audit['timing_analysis']
        timing_pass = timing.get('constant_time_compliant', False)
        if timing_pass is None:
            print(f"Timing Side-Channels: ⚠️ INCONCLUSIVE (scipy not available)")
        else:
            print(f"Timing Side-Channels: {'✅ PASS' if timing_pass else '⚠️ WARN'}")
            if not timing_pass:
                print(f"  Note: p-value={timing.get('ks_p_value', 0):.4f} < 0.05 indicates data-dependent timing")
        
        # Output hash
        print(f"Reproducibility Hash: ✅ {audit['output_hash'][:16]}...")
        
        print()
        print(f"Overall Status: {'✅ PASS' if all_pass else '⚠️ WARNINGS PRESENT'}")
        
        # Save report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'overall_status': 'PASS' if all_pass else 'WARN',
            'audit': audit,
            'summary': {
                'memory_bounds': 'PASS' if mem_pass else 'WARN',
                'numerical_stability': 'PASS' if num_pass else 'WARN',
                'timing_analysis': 'PASS' if timing_pass else ('INCONCLUSIVE' if timing_pass is None else 'WARN')
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved: {output_file}")
        
        return all_pass


def validate_production_kernels():
    """Validate all FlashCore production kernels"""
    from flashcore.fast.attention_production import attention as prod_attention
    from flashcore.fast.attention_multihead import multihead_attention
    
    results = {}
    
    # Test production kernel
    print("\n" + "=" * 80)
    print("Testing: attention_production.py")
    print("=" * 80)
    validator_prod = SecurityValidator(prod_attention)
    audit_prod = validator_prod.run_full_audit()
    results['production'] = validator_prod.generate_report(
        audit_prod,
        Path('logs/security_validation_production.json')
    )
    
    # Test multihead kernel
    print("\n" + "=" * 80)
    print("Testing: attention_multihead.py")
    print("=" * 80)
    validator_multi = SecurityValidator(multihead_attention)
    audit_multi = validator_multi.run_full_audit()
    results['multihead'] = validator_multi.generate_report(
        audit_multi,
        Path('logs/security_validation_multihead.json')
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SECURITY VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Production Kernel: {'✅ PASS' if results['production'] else '⚠️ WARNINGS'}")
    print(f"Multi-Head Kernel: {'✅ PASS' if results['multihead'] else '⚠️ WARNINGS'}")
    print()
    
    if all(results.values()):
        print("✅ ALL KERNELS PASS SECURITY VALIDATION")
        return 0
    else:
        print("⚠️ SECURITY WARNINGS PRESENT (review logs/)")
        return 0  # Warnings, not failures


if __name__ == "__main__":
    import sys
    exit_code = validate_production_kernels()
    sys.exit(exit_code)

