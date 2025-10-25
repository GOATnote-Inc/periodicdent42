import math
import shutil
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).parent.parent / "cudadent42"))

from bench.sdpa_fp8_stage_c_wmma import (  # noqa: E402
    quantize_sim_fp8_per_head,
    sdpa_fp8_stage_c_wmma_forward,
)


CUDA_AVAILABLE = torch.cuda.is_available()
NVCC_AVAILABLE = shutil.which("nvcc") is not None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device not available")
@pytest.mark.skipif(not NVCC_AVAILABLE, reason="nvcc compiler not found")
def test_quantizer_maps_zero_to_midpoint():
    zeros = torch.zeros(1, 2, 16, 64, device="cuda", dtype=torch.float16)
    encoded, scales = quantize_sim_fp8_per_head(zeros)

    assert torch.all(encoded == 128)
    assert torch.allclose(scales.cpu(), torch.ones(2), atol=1e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device not available")
@pytest.mark.skipif(not NVCC_AVAILABLE, reason="nvcc compiler not found")
def test_stage_c_wmma_matches_sdpa_fp16():
    torch.manual_seed(123)

    B, H, S, D = 1, 4, 128, 64
    Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    out = sdpa_fp8_stage_c_wmma_forward(Q, K, V)

    ref = F.scaled_dot_product_attention(
        Q.float(),
        K.float(),
        V.float(),
        is_causal=False,
        scale=1.0 / math.sqrt(float(D)),
    ).to(torch.float16)

    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)

