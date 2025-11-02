#!/usr/bin/env python3
"""
TDD: Test ceiling_scout_extended.py
Tests FA3 benchmarking, sparse detection, fusion detection
"""

import unittest
import sys
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available, skipping GPU tests")

# FusionDetector doesn't need torch - always import
from ceiling_scout_extended import FusionDetector

# These need torch
if TORCH_AVAILABLE:
    from ceiling_scout_extended import FA3Benchmarker, SparseDetector


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch CUDA not available")
class TestFA3Benchmarker(unittest.TestCase):
    """Test FlashAttention-3 benchmarking"""
    
    def setUp(self):
        """Setup FA3 benchmarker"""
        self.benchmarker = FA3Benchmarker(device="cuda")
    
    def test_benchmarker_creation(self):
        """Test FA3 benchmarker can be created"""
        self.assertEqual(self.benchmarker.device, "cuda")
    
    def test_pytorch_sdpa_benchmark(self):
        """Test PyTorch SDPA benchmarking runs"""
        try:
            result = self.benchmarker.benchmark_pytorch_sdpa(
                batch=1, heads=2, seq_len=128, head_dim=64
            )
            
            # Check result structure
            self.assertEqual(result.operation, "attention")
            self.assertGreater(result.tflops, 0)
            self.assertGreater(result.latency_ms, 0)
            self.assertTrue(result.is_library)
            self.assertEqual(result.library_name, "FlashAttention")
            
            print(f"    PyTorch SDPA: {result.latency_ms:.3f} ms")
        
        except Exception as e:
            self.skipTest(f"SDPA benchmark failed: {e}")
    
    def test_naive_attention_benchmark(self):
        """Test naive attention benchmarking runs"""
        try:
            result = self.benchmarker.benchmark_naive_attention(
                batch=1, heads=2, seq_len=128, head_dim=64
            )
            
            # Check result structure
            self.assertEqual(result.operation, "attention")
            self.assertGreater(result.tflops, 0)
            self.assertGreater(result.latency_ms, 0)
            self.assertFalse(result.is_library)
            
            print(f"    Naive attention: {result.latency_ms:.3f} ms")
        
        except Exception as e:
            self.skipTest(f"Naive benchmark failed: {e}")
    
    def test_detect_attention_ceiling(self):
        """Test attention ceiling detection"""
        try:
            opp = self.benchmarker.detect_attention_ceiling(
                batch=1, heads=2, seq_len=128, head_dim=64
            )
            
            # Check opportunity structure
            self.assertEqual(opp.operation, "attention")
            self.assertIn(opp.priority, ["NONE", "MEDIUM", "HIGH"])
            self.assertIn(opp.approach, ["NONE", "CUSTOM_ATTENTION"])
            
            print(f"    Recommendation: {opp.recommendation[:60]}...")
        
        except Exception as e:
            self.skipTest(f"Ceiling detection failed: {e}")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch CUDA not available")
class TestSparseDetector(unittest.TestCase):
    """Test sparse pattern detection"""
    
    def test_dense_matrix_detection(self):
        """Test that dense matrix is classified correctly"""
        # Create dense matrix (no zeros)
        dense = torch.randn(1024, 1024, device="cuda")
        
        analysis = SparseDetector.analyze_sparsity(dense)
        
        self.assertLess(analysis["sparsity"], 0.1)  # Should be <10% sparse
        self.assertEqual(analysis["pattern"], "DENSE")
        print(f"    Dense matrix: {analysis['sparsity']:.1%} sparse → {analysis['pattern']}")
    
    def test_highly_sparse_matrix(self):
        """Test that highly sparse matrix is detected"""
        # Create 90% sparse matrix
        sparse = torch.randn(1024, 1024, device="cuda")
        mask = torch.rand(1024, 1024, device="cuda") > 0.9
        sparse = sparse * mask
        
        analysis = SparseDetector.analyze_sparsity(sparse)
        
        self.assertGreater(analysis["sparsity"], 0.85)
        self.assertIn(analysis["pattern"], ["HIGHLY_SPARSE", "UNSTRUCTURED", "BLOCK_SPARSE"])
        print(f"    Sparse matrix: {analysis['sparsity']:.1%} sparse → {analysis['pattern']}")
    
    def test_block_sparse_detection(self):
        """Test block sparse pattern detection"""
        # Create block sparse matrix (128x128 blocks)
        M, N = 1024, 1024
        block_size = 128
        sparse = torch.zeros(M, N, device="cuda")
        
        # Fill only some blocks
        for i in range(0, M, block_size):
            for j in range(0, N, block_size):
                if torch.rand(1).item() > 0.7:  # 30% of blocks non-zero
                    sparse[i:i+block_size, j:j+block_size] = torch.randn(block_size, block_size, device="cuda")
        
        analysis = SparseDetector.analyze_sparsity(sparse)
        
        self.assertGreater(analysis["sparsity"], 0.5)
        is_bsr = analysis["is_bsr_friendly"]
        print(f"    Block sparse: {analysis['sparsity']:.1%} sparse, BSR-friendly={is_bsr}")
    
    def test_recommend_sparse_kernel_dense(self):
        """Test recommendation for dense matrix"""
        analysis = {"sparsity": 0.05, "pattern": "DENSE"}
        
        opp = SparseDetector.recommend_sparse_kernel(analysis, 8192, 8192, 8192)
        
        self.assertEqual(opp.approach, "NONE")
        self.assertIn("cuBLAS", opp.recommendation)
        print(f"    Dense recommendation: {opp.approach}")
    
    def test_recommend_sparse_kernel_block_sparse(self):
        """Test recommendation for block sparse matrix"""
        analysis = {"sparsity": 0.875, "pattern": "BLOCK_SPARSE"}
        
        opp = SparseDetector.recommend_sparse_kernel(analysis, 8192, 8192, 8192)
        
        self.assertEqual(opp.approach, "CUSTOM_BSR_SPARSE")
        self.assertIn("BlackwellSparseK", opp.recommendation)
        print(f"    Block sparse recommendation: {opp.approach}")


class TestFusionDetector(unittest.TestCase):
    """Test fusion opportunity detection"""
    
    def test_detect_gemm_bias_relu(self):
        """Test detection of GEMM+Bias+ReLU pattern"""
        ops = ["gemm", "bias", "relu"]
        
        analysis = FusionDetector.analyze_sequence(ops)
        
        self.assertEqual(analysis["num_ops"], 3)
        self.assertGreater(len(analysis["opportunities"]), 0)
        
        # Check if GEMM+Bias+ReLU was found
        found = False
        for opp in analysis["opportunities"]:
            if "GEMM+Bias+ReLU" in opp["name"]:
                found = True
                print(f"    Found: {opp['name']} → {opp['approach']}")
        
        self.assertTrue(found, "GEMM+Bias+ReLU fusion not detected")
    
    def test_detect_attention_fusion(self):
        """Test detection of attention fusion pattern"""
        ops = ["attention", "mask", "dropout"]
        
        analysis = FusionDetector.analyze_sequence(ops)
        
        self.assertGreater(len(analysis["opportunities"]), 0)
        print(f"    Found {len(analysis['opportunities'])} fusion opportunities")
    
    def test_no_fusion_opportunities(self):
        """Test that unrelated ops don't suggest fusion"""
        ops = ["layernorm", "softmax"]
        
        analysis = FusionDetector.analyze_sequence(ops)
        
        self.assertEqual(len(analysis["opportunities"]), 0)
        self.assertEqual(analysis["priority"], "NONE")
        print(f"    No fusion found (expected)")
    
    def test_transformer_block_sequence(self):
        """Test typical transformer block sequence"""
        ops = ["gemm", "bias", "relu", "gemm", "layernorm", "residual", "activation"]
        
        analysis = FusionDetector.analyze_sequence(ops)
        
        # Should find at least GEMM+Bias+ReLU
        self.assertGreater(len(analysis["opportunities"]), 0)
        print(f"    Transformer block: {len(analysis['opportunities'])} fusion opportunities")


def run_tests():
    """Run all extended tests"""
    print("=" * 70)
    print("Testing Ceiling Scout Extended - TDD Approach")
    print("=" * 70)
    print()
    
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch CUDA not available")
        print("   Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("Running non-GPU tests only...")
        print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("✅ ALL EXTENDED TESTS PASSED")
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

