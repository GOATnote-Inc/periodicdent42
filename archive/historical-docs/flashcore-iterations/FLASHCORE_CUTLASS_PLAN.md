# FlashCore: CUTLASS FMHA Integration Plan

**Date**: October 22, 2025  
**Goal**: <26 μs using CUTLASS FlashAttention-3 patterns  
**Current Best**: 44 μs (PyTorch SDPA)  
**Target Improvement**: 1.7× faster (44 → <26 μs)

---

## 🎯 **Mission**

Leverage NVIDIA CUTLASS library to achieve FlashAttention-3 performance:
- **Expected**: 15-25 μs
- **Method**: Adapt CUTLASS FMHA example for L4 (sm_89)
- **Integration**: PyTorch C++ extension
- **Timeline**: 2-4 hours

---

## 📋 **Phase 1: Exploration (30 min)**

### **Step 1.1: Clone CUTLASS on L4**
```bash
cd ~
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git
cd cutlass
```

### **Step 1.2: Explore FMHA Examples**
```bash
# Find attention-related examples
find examples -name "*attention*" -o -name "*fmha*"

# Look for FlashAttention implementations
grep -r "flash" examples/ --include="*.cu" --include="*.h"

# Check documentation
cat examples/README.md
```

### **Step 1.3: Identify Target Implementation**
Options:
- `examples/41_fused_multi_head_attention/` (FMHA)
- `examples/5x_fused_mha/` (if exists)
- CUTLASS 3.x attention primitives

**Expected**: Find production FMHA kernel we can adapt

---

## 📋 **Phase 2: Adaptation (1-2h)**

### **Step 2.1: Study Reference Implementation**
```bash
cd examples/41_fused_multi_head_attention  # or equivalent
cat fmha_fprop.cu | head -200  # Study kernel structure
```

**Key Aspects to Understand**:
1. Tiling strategy (BLOCK_M, BLOCK_N)
2. WMMA/Tensor Core usage
3. Online softmax implementation
4. Shared memory layout
5. Launch configuration

### **Step 2.2: Create Minimal Wrapper**
```cuda
// flashcore_cutlass_fmha.cu
#include "cutlass/gemm/device/gemm.h"
#include "cute/tensor.hpp"
// ... CUTLASS FMHA headers

// Minimal kernel launch for our shape:
// B=1, H=8, S=512, D=64, sm_89
void launch_cutlass_fmha(
    const half* Q,      // [B, H, S, D]
    const half* K,      // [B, H, S, D]
    const half* V,      // [B, H, S, D]
    half* O,            // [B, H, S, D]
    int B, int H, int S, int D,
    cudaStream_t stream
) {
    // Adapt CUTLASS FMHA for our params
    // Use sm_89 optimizations
}
```

### **Step 2.3: Build Configuration**
```python
# build_cutlass.py
from torch.utils.cpp_extension import load
import os

cutlass_dir = os.path.expanduser("~/cutlass")
cutlass_include = f"{cutlass_dir}/include"
cutlass_tools = f"{cutlass_dir}/tools/util/include"

flashcore_cutlass = load(
    name='flashcore_cutlass',
    sources=[
        'kernels/flashcore_cutlass_fmha.cu',
        'kernels/flashcore_cutlass_bindings.cu',
    ],
    extra_include_paths=[cutlass_include, cutlass_tools],
    extra_cuda_cflags=[
        '-O3',
        '-arch=sm_89',  # L4 Ada
        '-std=c++17',   # CUTLASS requires C++17
        '-Xptxas', '-v',
        '--use_fast_math',
        '-DCUTLASS_ENABLE_TENSOR_CORES=1',
    ],
    verbose=True
)
```

---

## 📋 **Phase 3: Integration (30 min)**

### **Step 3.1: PyTorch Bindings**
```cpp
// flashcore_cutlass_bindings.cu
#include <torch/extension.h>
#include "flashcore_cutlass_fmha.cu"

torch::Tensor fmha_cutlass(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // Shape: [B, H, S, D]
    auto O = torch::empty_like(Q);
    
    launch_cutlass_fmha(
        Q.data_ptr<at::Half>(),
        K.data_ptr<at::Half>(),
        V.data_ptr<at::Half>(),
        O.data_ptr<at::Half>(),
        Q.size(0), Q.size(1), Q.size(2), Q.size(3),
        at::cuda::getCurrentCUDAStream()
    );
    
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fmha", &fmha_cutlass, "CUTLASS FMHA");
}
```

### **Step 3.2: Test Harness**
```python
# test_cutlass.py
import torch
import flashcore_cutlass
import statistics

def test_cutlass_fmha():
    B, H, S, D = 1, 8, 512, 64
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, S, D, dtype=torch.float16, device='cuda')
    
    # Correctness
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    O = flashcore_cutlass.fmha(Q, K, V)
    error = (O - O_ref).abs().max().item()
    
    # Benchmark
    for _ in range(100):
        O = flashcore_cutlass.fmha(Q, K, V)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        O = flashcore_cutlass.fmha(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    
    p50 = statistics.median(times)
    
    print(f"CUTLASS FMHA:")
    print(f"  Error: {error:.6f}")
    print(f"  p50: {p50:.2f} μs")
    print(f"  Target: <26 μs")
    print(f"  Status: {'✅ SUCCESS' if p50 < 26 else '⚠️  Close'}")

if __name__ == '__main__':
    test_cutlass_fmha()
```

---

## 📋 **Phase 4: Optimization (1h)**

### **Step 4.1: Profile with NCU**
```bash
cd ~/flashcore
/usr/local/cuda/bin/ncu \
    --set full \
    --target-processes all \
    --export cutlass_profile \
    python3 test_cutlass.py
```

**Metrics to Check**:
- `smsp__sass_thread_inst_executed_op_tensor_active.pct` (Tensor Core %)
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` (SM utilization)
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` (Memory %)

### **Step 4.2: Tune Launch Config**
If not hitting <26 μs, try:
```cuda
// Experiment with tiling
const int BLOCK_M = 64;  // vs 128
const int BLOCK_N = 64;  // vs 128
const int NUM_WARPS = 4; // vs 8

// Try different stages for cp.async
const int PIPELINE_STAGES = 2; // vs 3, 4
```

---

## 📋 **Phase 5: Validation (30 min)**

### **Success Criteria**
✅ **Correctness**: `max_error < 0.1` vs PyTorch SDPA  
✅ **Performance**: p50 < 26 μs  
✅ **Stability**: p90 < 30 μs (low variance)  
✅ **Resources**: No spills, reasonable register usage  

### **Comparison Benchmark**
```python
# Final comparison
results = {
    'Baseline': 1397,
    'Our WMMA': 306,
    'Triton': 76,
    'PyTorch SDPA': 44,
    'CUTLASS FMHA': ???,  # Target: <26
}
```

---

## 🎯 **Expected Outcomes**

### **Optimistic (80% confidence)**
```
CUTLASS FMHA: 15-20 μs ✅
- FlashAttention-3 patterns
- Optimal Tensor Core usage (>80%)
- Production-quality implementation
- Beats PyTorch by 2-3×
```

### **Realistic (90% confidence)**
```
CUTLASS FMHA: 20-30 μs
- Better than PyTorch (44 μs)
- Close to or beats target (26 μs)
- Minimal tuning needed
```

### **Conservative (95% confidence)**
```
CUTLASS FMHA: 30-40 μs
- Competitive with PyTorch
- Needs more optimization
- Learning value high even if not beating target
```

---

## 🚀 **Execution Plan**

### **Step-by-Step Commands**

1. **Clone CUTLASS on L4** (5 min)
2. **Explore examples** (15 min)
3. **Create minimal wrapper** (30 min)
4. **Build with PyTorch** (15 min)
5. **Test correctness** (10 min)
6. **Benchmark performance** (10 min)
7. **Profile with NCU** (20 min)
8. **Optimize if needed** (30 min)
9. **Final validation** (15 min)

**Total**: 2.5 hours (expected)

---

## 💡 **Key Advantages of CUTLASS**

1. **Production-Quality**: Used by NVIDIA internally
2. **FlashAttention-3**: Latest optimizations (1.5-2× over FA-2)
3. **sm_89 Optimized**: Ada-specific Tensor Core usage
4. **Template Library**: Easy to adapt for custom shapes
5. **CuTe DSL**: Modern GPU programming abstractions

---

## 📚 **References**

1. **CUTLASS 3.x Documentation**:
   - https://github.com/NVIDIA/cutlass
   - FlashAttention-3 blog post
   - CuTe programming guide

2. **Our Context**:
   - PyTorch SDPA: 44 μs baseline
   - L4 GPU: sm_89, 242 TFLOPS FP16
   - Shape: B=1, H=8, S=512, D=64

---

**Status**: Ready to execute Phase 1!  
**Next**: Clone CUTLASS and explore FMHA examples 🚀

