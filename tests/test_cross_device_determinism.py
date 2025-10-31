"""
Test cross-device determinism (requires 2+ GPUs)
"""
import subprocess
import hashlib
import pytest

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs 2+ GPUs")
def test_cross_device_determinism():
    """Verify same kernel gives identical results on different GPUs"""
    vec = secrets.token_bytes(64)
    
    results = []
    for device_id in range(2):
        # Set device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        
        # Run kernel
        out = subprocess.check_output(
            ["./bin/chacha20_kernel_bench", "--vec", vec.hex()],
            stderr=subprocess.DEVNULL
        )
        
        results.append(hashlib.sha256(out).hexdigest())
    
    # Must be bitwise identical across devices
    assert len(set(results)) == 1, \
        f"Cross-device mismatch: GPU0={results[0][:16]}... GPU1={results[1][:16]}..."
