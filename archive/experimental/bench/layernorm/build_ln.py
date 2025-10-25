from torch.utils.cpp_extension import load
import os
def build_ln(name="layernorm_v1", extra=[]):
    cflags = [
        "-O3","-use_fast_math","-Xptxas","-v","-std=c++17",
        "-gencode=arch=compute_89,code=sm_89",
    ] + extra
    srcs = ["kernels/layernorm/layernorm.cu","kernels/layernorm/binding.cpp"]
    return load(name=name, sources=srcs, extra_cuda_cflags=cflags, verbose=False)
if __name__ == "__main__": build_ln()

