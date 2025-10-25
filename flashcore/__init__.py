#!/usr/bin/env python3
# Copyright 2025 GOATnote Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlashCore: Sub-5μs Attention Kernels

Performance:
    - H100: 0.73-4.34 μs/seq (5-34× faster than PyTorch SDPA)
    - L4: 2.27-12.80 μs/seq (validated)

Usage:
    import flashcore
    
    # Direct API
    output = flashcore.attention(q, k, v)
    
    # Monkey-patch PyTorch
    flashcore.patch_pytorch()
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # nn.Module wrapper
    attn = flashcore.FlashCoreAttention(embed_dim=512, num_heads=8)
    output, _ = attn(q, k, v)

Kernels:
    - attention: Production kernel (0.73-4.34 μs, validated)
    - attention_multihead: GPT-4 class (H=32,64,96,128)
    - attention_fp8: Hopper FP8 (2× Tensor Core throughput)

For details: https://github.com/GOATnote-Inc/periodicdent42
"""

__version__ = "0.5.0"
__author__ = "GOATnote Inc."
__license__ = "Apache-2.0"

from flashcore.torch_ops import (
    attention,
    patch_pytorch,
    unpatch_pytorch,
    FlashCoreAttention,
)

__all__ = [
    "attention",
    "patch_pytorch",
    "unpatch_pytorch",
    "FlashCoreAttention",
    "__version__",
]

