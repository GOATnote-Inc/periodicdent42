import torch
import subprocess
import os
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

def get_cuda_arch_list():
    """Robust arch detection with fallbacks."""
    # CI override: Use deterministic arch list if set
    ci_archs = os.getenv('TORCH_CUDA_ARCH_LIST', '')
    if ci_archs:
        archs = ci_archs.replace('.', '').replace(';', ' ').split()
        print(f"✅ Using TORCH_CUDA_ARCH_LIST (CI mode): {archs}")
        return archs
    
    # Method 1: PyTorch runtime detection
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
            print(f"✅ Detected arch from PyTorch: {arch}")
            return [arch]
        except Exception as e:
            print(f"⚠️  PyTorch detection failed: {e}")
    
    # Method 2: nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, timeout=5
        )
        caps = [line.strip().replace('.', '') for line in result.stdout.split('\n') if line.strip()]
        if caps:
            print(f"✅ Detected archs from nvidia-smi: {caps}")
            return caps
    except Exception as e:
        print(f"⚠️  nvidia-smi failed: {e}")
    
    # Fallback: L4 (SM89) - our current GPU
    print("⚠️  Could not detect GPU, defaulting to SM89 (L4)")
    print("    Tip: Set TORCH_CUDA_ARCH_LIST='75;80;89;90' for multi-arch build")
    return ['89']

def setup_cuda_extension():
    archs = get_cuda_arch_list()
    print(f"🎯 Target CUDA architectures: {archs}")
    
    # Determine which dtypes to support
    has_bf16 = any(int(arch) >= 80 for arch in archs)
    print(f"{'✅' if has_bf16 else '❌'} BF16 support: {'enabled' if has_bf16 else 'disabled'}")
    
    # Base sources (always compiled)
    sources = [
        'python/flashmoe_science/csrc/flash_attention_fp16_sm75.cu',
        'python/flashmoe_science/csrc/bindings_new.cpp',
    ]
    
    # Add BF16 source only if supported
    if has_bf16:
        sources.append('python/flashmoe_science/csrc/flash_attention_bf16_sm80.cu')
    
    # Generate gencode flags (real SASS targets for deterministic perf)
    gencode_flags = []
    for arch in archs:
        arch_num = int(arch)
        # SASS target (not PTX-only)
        gencode_flags.append(f'-gencode=arch=compute_{arch_num},code=sm_{arch_num}')
    
    # Optional: Add PTX fallback for future cards
    max_arch = max(int(a) for a in archs)
    gencode_flags.append(f'-gencode=arch=compute_{max_arch},code=compute_{max_arch}')
    print(f"📦 Added PTX fallback for compute_{max_arch}")
    
    # Base CUDA flags (order matters per NVCC docs!)
    base_cuda_flags = [
        '-O3',
        '--use_fast_math',
        '-lineinfo',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        *gencode_flags,
        '-Xcompiler=-fno-strict-aliasing',
        '-Xcompiler=-fPIC',
        '-Xcompiler=-fno-omit-frame-pointer',  # Better profiling
    ]
    
    # ABI compatibility: Match PyTorch's glibc++ ABI (prevents symbol mismatch)
    abi_flag = '-D_GLIBCXX_USE_CXX11_ABI=' + ('1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0')
    print(f"🔗 ABI flag: {abi_flag} (matching PyTorch)")
    
    # C++ flags (host compiler - always hide BF16)
    cxx_flags = [
        '-DCUDA_NO_BFLOAT16',
        '-D__CUDA_NO_BFLOAT16_OPERATORS__',
        abi_flag,  # Critical: Match PyTorch ABI
        '-O3',
        '-fPIC',
    ]
    
    # Preprocessor defines
    define_macros = [
        ('FLASHMOE_HAS_BF16', '1' if has_bf16 else '0'),
    ]
    
    # Final compilation args
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': base_cuda_flags + [
            '-DCUDA_NO_BFLOAT16',  # Default for FP16 files
            '-D__CUDA_NO_BFLOAT16_OPERATORS__',
        ],
    }
    
    return CUDAExtension(
        name='flashmoe_science._C',
        sources=sources,
        extra_compile_args=extra_compile_args,
        define_macros=define_macros,
        include_dirs=['python/flashmoe_science/csrc'],
    ), BuildExtension

ext, build_ext_class = setup_cuda_extension()

setup(
    name='flashmoe-science',
    version='0.1.0',
    description='Production-grade FlashAttention CUDA kernels with BF16 support',
    author='GOATnote Autonomous Research Lab Initiative',
    author_email='b@thegoatnote.com',
    ext_modules=[ext],
    cmdclass={'build_ext': build_ext_class},
    python_requires='>=3.8',
    install_requires=['torch>=2.0.0'],
)

