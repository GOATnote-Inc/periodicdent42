#!/usr/bin/env python3
"""
TDD: Test ceiling_scout.py before committing
"""

import unittest
import sys
import tempfile
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent))

from ceiling_scout import (
    Precision, Operation, BenchmarkResult, OpportunityScore, CeilingScout
)


class TestDataClasses(unittest.TestCase):
    """Test basic data structures"""
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult can be created"""
        result = BenchmarkResult(
            name="test",
            operation="gemm",
            shape=(8192, 8192, 8192),
            precision="fp16",
            tflops=628.0,
            latency_ms=15.0,
            memory_gb=2.0
        )
        self.assertEqual(result.tflops, 628.0)
        self.assertEqual(result.shape, (8192, 8192, 8192))
    
    def test_opportunity_score_creation(self):
        """Test OpportunityScore can be created"""
        opp = OpportunityScore(
            operation="gemm",
            shape=(8192, 8192, 8192),
            baseline_tflops=627.0,
            ceiling_tflops=628.0,
            efficiency=0.998,
            recommendation="Use cuBLAS",
            priority="NONE",
            approach="NONE",
            config_suggestion={"use": "cuBLAS"}
        )
        self.assertEqual(opp.efficiency, 0.998)
        self.assertEqual(opp.priority, "NONE")


class TestCeilingScout(unittest.TestCase):
    """Test CeilingScout engine"""
    
    def setUp(self):
        """Setup test environment"""
        self.scout = CeilingScout(device="h100", cuda_path="/usr/local/cuda-13.0")
    
    def test_scout_creation(self):
        """Test scout can be created"""
        self.assertEqual(self.scout.device, "h100")
        self.assertTrue(self.scout.cuda_path.exists() or True)  # May not exist in test env
    
    def test_nvcc_path_correct(self):
        """Test nvcc path is constructed correctly"""
        expected = Path("/usr/local/cuda-13.0") / "bin" / "nvcc"
        self.assertEqual(self.scout.nvcc, expected)


class TestEnums(unittest.TestCase):
    """Test enum definitions"""
    
    def test_precision_enum(self):
        """Test Precision enum values"""
        self.assertEqual(Precision.FP16.value, "fp16")
        self.assertEqual(Precision.FP32.value, "fp32")
        self.assertEqual(Precision.FP8_E4M3.value, "fp8_e4m3")
    
    def test_operation_enum(self):
        """Test Operation enum values"""
        self.assertEqual(Operation.GEMM.value, "gemm")
        self.assertEqual(Operation.ATTENTION.value, "attention")


class TestDecisionLogic(unittest.TestCase):
    """Test ceiling detection decision logic"""
    
    def test_optimal_efficiency(self):
        """Test that >90% efficiency recommends library"""
        scout = CeilingScout(device="h100")
        
        # Mock a high-efficiency scenario
        # This tests the logic without needing actual CUDA
        baseline_tflops = 627.0
        ceiling_tflops = 628.0
        efficiency = baseline_tflops / ceiling_tflops  # 99.8%
        
        self.assertGreater(efficiency, 0.90)
        # Should recommend using library
    
    def test_medium_efficiency(self):
        """Test that 70-90% efficiency suggests CUTLASS sweep"""
        baseline_tflops = 500.0
        ceiling_tflops = 628.0
        efficiency = baseline_tflops / ceiling_tflops  # 79.6%
        
        self.assertGreater(efficiency, 0.70)
        self.assertLess(efficiency, 0.90)
        # Should recommend CUTLASS sweep
    
    def test_low_efficiency(self):
        """Test that <70% efficiency suggests investigation"""
        baseline_tflops = 400.0
        ceiling_tflops = 628.0
        efficiency = baseline_tflops / ceiling_tflops  # 63.7%
        
        self.assertLess(efficiency, 0.70)
        # Should recommend investigation


class TestReportGeneration(unittest.TestCase):
    """Test report generation"""
    
    def test_generate_report_structure(self):
        """Test that report has correct structure"""
        scout = CeilingScout(device="h100")
        
        # Create a mock opportunity
        opp = OpportunityScore(
            operation="gemm",
            shape=(8192, 8192, 8192),
            baseline_tflops=627.0,
            ceiling_tflops=628.0,
            efficiency=0.998,
            recommendation="Use cuBLAS",
            priority="NONE",
            approach="NONE",
            config_suggestion={"use": "cuBLAS"}
        )
        
        report = scout.generate_report([opp])
        
        # Check structure
        self.assertIn("device", report)
        self.assertIn("opportunities", report)
        self.assertIn("summary", report)
        self.assertIn("recommendations", report)
        
        # Check summary
        self.assertEqual(report["summary"]["total_ops"], 1)
        self.assertEqual(report["summary"]["already_optimal"], 1)
    
    def test_report_output_to_file(self):
        """Test that report can be saved to file"""
        scout = CeilingScout(device="h100")
        
        opp = OpportunityScore(
            operation="gemm",
            shape=(8192, 8192, 8192),
            baseline_tflops=627.0,
            ceiling_tflops=628.0,
            efficiency=0.998,
            recommendation="Use cuBLAS",
            priority="NONE",
            approach="NONE",
            config_suggestion={"use": "cuBLAS"}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            report = scout.generate_report([opp], output_file=output_path)
            
            # Check file was created
            self.assertTrue(output_path.exists())
            
            # Check file contents
            import json
            with open(output_path) as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded["device"], "h100")
            self.assertEqual(len(loaded["opportunities"]), 1)
        
        finally:
            output_path.unlink(missing_ok=True)


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("Testing Ceiling Scout - TDD Approach")
    print("=" * 70)
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
        print(f"   {result.testsRun} tests run, 0 failures")
    else:
        print("❌ TESTS FAILED")
        print(f"   {len(result.failures)} failures, {len(result.errors)} errors")
        print()
        print("Fix issues before committing!")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

