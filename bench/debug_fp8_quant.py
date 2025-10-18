#!/usr/bin/env python3
"""
Debug FP8 quantization/dequantization.
Test: Quant → Dequant → Compare (should be identity within error).
"""

import torch
import numpy as np

def quantize_to_fp8_debug(tensor, per_channel=True):
    """Quantize with detailed logging."""
    print("=" * 80)
    print("QUANTIZATION")
    print("=" * 80)
    
    B, H, S, D = tensor.shape
    print(f"Input shape: {tensor.shape}")
    print(f"Input range: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print(f"Input mean: {tensor.mean():.6f}, std: {tensor.std():.6f}")
    print()
    
    # Per-head quantization
    abs_max = tensor.abs().view(B, H, -1).max(dim=2, keepdim=True)[0]
    abs_max = abs_max.view(B, H, 1, 1)
    print(f"Per-head abs_max: {abs_max.view(H).cpu().numpy()}")
    
    fp8_max = 448.0
    scale = abs_max / fp8_max
    scale = scale.clamp(min=1e-12)
    print(f"Per-head scale: {scale.view(H).cpu().numpy()}")
    print()
    
    # Quantize: map [-448, 448] → [0, 255]
    tensor_scaled = tensor / scale
    print(f"After scaling: [{tensor_scaled.min():.6f}, {tensor_scaled.max():.6f}]")
    
    tensor_clipped = tensor_scaled.clamp(-fp8_max, fp8_max)
    print(f"After clipping: [{tensor_clipped.min():.6f}, {tensor_clipped.max():.6f}]")
    
    # Map to [0, 255]
    tensor_uint8 = ((tensor_clipped + fp8_max) / (2 * fp8_max) * 255).round().to(torch.uint8)
    print(f"uint8 range: [{tensor_uint8.min()}, {tensor_uint8.max()}]")
    print()
    
    scale_per_head = scale[0, :, 0, 0].to(dtype=torch.float32, device='cuda')
    
    return tensor_uint8, scale_per_head

def dequantize_from_fp8_debug(fp8_data, scale):
    """Dequantize with detailed logging."""
    print("=" * 80)
    print("DEQUANTIZATION (Python)")
    print("=" * 80)
    
    fp8_max = 448.0
    
    print(f"uint8 range: [{fp8_data.min()}, {fp8_data.max()}]")
    print(f"Scale: {scale.cpu().numpy()}")
    print()
    
    # Reverse: [0, 255] → [-448, 448]
    tensor_float = fp8_data.float() / 255.0 * (2 * fp8_max) - fp8_max
    print(f"After unmap: [{tensor_float.min():.6f}, {tensor_float.max():.6f}]")
    
    scale = scale.view(1, -1, 1, 1)
    tensor_rescaled = tensor_float * scale
    print(f"After rescale: [{tensor_rescaled.min():.6f}, {tensor_rescaled.max():.6f}]")
    print()
    
    return tensor_rescaled.to(torch.float16)

def test_quantization_roundtrip():
    """Test: Original → Quant → Dequant → Compare."""
    print("\n" + "=" * 80)
    print("TEST: Quantization Roundtrip")
    print("=" * 80)
    print()
    
    # Small test tensor
    torch.manual_seed(42)
    B, H, S, D = 1, 2, 4, 8
    original = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    
    print(f"Original tensor:")
    print(original[0, 0, :, :4])  # Print first head, first 4 dims
    print()
    
    # Quantize
    fp8_data, scale = quantize_to_fp8_debug(original, per_channel=True)
    
    # Dequantize
    reconstructed = dequantize_from_fp8_debug(fp8_data, scale)
    
    print(f"Reconstructed tensor:")
    print(reconstructed[0, 0, :, :4])
    print()
    
    # Compare
    diff = (original - reconstructed).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"Max relative error: {(diff / (original.abs() + 1e-8)).max().item():.6f}")
    print()
    
    # Expected error for 8-bit: ~1% of range
    expected_error = 448.0 / 255.0 * 2  # Quantization step size
    print(f"Expected max error (quant step): {expected_error:.6f}")
    
    if max_diff < expected_error * 2:
        print("✅ PASS: Quantization roundtrip works!")
        return True
    else:
        print("❌ FAIL: Quantization roundtrip broken!")
        
        # Debug: Show where it fails
        max_idx = diff.argmax()
        b, h, s, d = np.unravel_index(max_idx.cpu().numpy(), original.shape)
        print()
        print(f"Worst case at [{b}, {h}, {s}, {d}]:")
        print(f"  Original: {original[b, h, s, d].item():.6f}")
        print(f"  uint8: {fp8_data[b, h, s, d].item()}")
        print(f"  Reconstructed: {reconstructed[b, h, s, d].item():.6f}")
        print(f"  Diff: {diff[b, h, s, d].item():.6f}")
        return False

def test_cuda_dequant_formula():
    """Test the CUDA dequantization formula matches Python."""
    print("\n" + "=" * 80)
    print("TEST: CUDA Dequant Formula")
    print("=" * 80)
    print()
    
    # Test a few values
    test_vals = [0, 1, 127, 128, 254, 255]
    scale = 0.01
    fp8_max = 448.0
    
    print("Testing CUDA formula: (uint8 / 255 * 896 - 448) * scale")
    print()
    
    for val in test_vals:
        # Python formula
        python_result = (float(val) / 255.0 * (2 * fp8_max) - fp8_max) * scale
        
        # What we expect
        print(f"uint8={val:3d} → {python_result:+8.4f}")
    
    print()
    print("Expected behavior:")
    print("  0 → -4.48 (min)")
    print("  127 → ~0.00 (mid)")
    print("  255 → +4.48 (max)")

if __name__ == '__main__':
    print("FP8 QUANTIZATION DEBUG")
    print("=" * 80)
    print()
    
    # Test 1: Roundtrip
    success = test_quantization_roundtrip()
    
    # Test 2: Formula verification
    test_cuda_dequant_formula()
    
    print()
    if success:
        print("✅ Quantization logic is CORRECT")
        print("Bug must be in CUDA kernel itself")
    else:
        print("❌ Quantization logic is BROKEN")
        print("Fix Python quant/dequant first")

