# Working Build Recipe - Session N+2

**Date**: October 12, 2025 3:51 AM  
**Result**: 0.10× speedup baseline (FP16, L4 GPU)  
**Time to working benchmark**: 110 minutes  

## Prerequisites

- Commit: 5b4c0c8 (Session N state)
- L4 GPU with CUDA 12.8
- PyTorch 2.7.1+cu128
- Python 3.10

## Step-by-Step Recipe

### 1. Checkout Working Commit
```bash
cd ~/periodicdent42/cudadent42
git fetch origin
git checkout 5b4c0c8
```

### 2. Apply L4 Configuration
```bash
# Reduce tiles and warps for L4's 48KB shared memory
sed -i 's/NUM_WARPS_PER_BLOCK = 12/NUM_WARPS_PER_BLOCK = 8/' python/flashmoe_science/csrc/build_config.h
sed -i 's/NUM_WARPGROUPS = 3/NUM_WARPGROUPS = 2/' python/flashmoe_science/csrc/build_config.h
sed -i 's/TILE_SIZE_M = 128/TILE_SIZE_M = 64/' python/flashmoe_science/csrc/build_config.h
sed -i 's/TILE_SIZE_N = 128/TILE_SIZE_N = 64/' python/flashmoe_science/csrc/build_config.h
sed -i 's/TILE_SIZE_K = 128/TILE_SIZE_K = 64/' python/flashmoe_science/csrc/build_config.h
sed -i 's/THREADS_PER_BLOCK == 384/THREADS_PER_BLOCK == 256 || THREADS_PER_BLOCK == 384/' python/flashmoe_science/csrc/build_config.h
```

### 3. Create Combined Bindings (KEY STEP!)
```bash
cat > python/flashmoe_science/csrc/bindings_native.cu << 'BINDINGS'
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// Include CUDA kernel directly - causes implicit instantiation
#include "flash_attention_science.cu"

torch::Tensor flash_attention_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool causal, float softmax_scale
) {
    c10::cuda::CUDAGuard device_guard(Q.device());
    
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int head_dim = Q.size(3);
    
    auto O = torch::empty_like(Q);
    auto softmax_lse = torch::empty({batch_size, num_heads, seq_len},
                                     torch::dtype(torch::kFloat32).device(Q.device()));
    
    if (Q.dtype() == torch::kFloat16) {
        // Use native CUDA half type
        flashmoe::flash_attention_forward<half>(
            reinterpret_cast<const half*>(Q.data_ptr()),
            reinterpret_cast<const half*>(K.data_ptr()),
            reinterpret_cast<const half*>(V.data_ptr()),
            reinterpret_cast<half*>(O.data_ptr()),
            softmax_lse.data_ptr<float>(),
            batch_size, num_heads, seq_len, head_dim,
            softmax_scale, causal
        );
    } else {
        throw std::runtime_error("Unsupported dtype (only FP16 supported)");
    }
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_forward", &flash_attention_forward_cuda,
          "FlashAttention-Science forward pass");
}
BINDINGS
```

### 4. Create Setup Script
```bash
cat > setup_native.py << 'SETUP'
from setuptools import setup
from torch.utils import cpp_extension

sources = [
    'python/flashmoe_science/csrc/bindings_native.cu',
]

setup(
    name='flashmoe-science',
    ext_modules=[
        cpp_extension.CUDAExtension(
            'flashmoe_science._C',
            sources,
            include_dirs=['python/flashmoe_science/csrc'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_89,code=sm_89',
                    '--use_fast_math',
                ]
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
SETUP
```

### 5. Build Extension
```bash
python3 setup_native.py build_ext --inplace
```

**Expected output**: `flashmoe_science/_C.cpython-310-x86_64-linux-gnu.so` (~14 MB)

### 6. Fix Benchmark
```bash
# Benchmark uses 4D tensors and passes causal + softmax_scale
sed -i 's/Q_flat, K_flat, V_flat, False, 1\.0 \/ (64 \*\* 0\.5)/Q, K, V, False, 1.0 \/ (D ** 0.5)/g' benches/bench_correctness_and_speed.py
```

### 7. Run Benchmark
```bash
export PYTHONPATH=".:${PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

python3 benches/bench_correctness_and_speed.py
```

**Expected result**: 0.10× average speedup, 0.06× @ S=128

## Key Insights

1. **DO NOT** use separate compilation + explicit instantiation
2. **DO** include .cu file in bindings for implicit instantiation  
3. **DO** use native CUDA types (half, not c10::Half)
4. **DO** pass 4D tensors [B, H, S, D] to kernel

## Benchmark Results

| Config | Speedup |
|--------|---------|
| S=32 | 0.28× |
| S=64 | 0.15× |
| S=128 | 0.06× |
| S=256 | 0.02× |
| Average | 0.10× |

Memory: 79.8% less than PyTorch ✅

## Known Issues

- Slow (10% of PyTorch) - needs profiling
- Only supports FP16 (BF16 crashes)
- Single kernel implementation (no optimizations yet)
