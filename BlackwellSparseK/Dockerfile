# H100 + CUDA 13.0.2 + CUTLASS 4.3.0 + Nsight Compute CLI
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential python3 python3-pip wget && \
    rm -rf /var/lib/apt/lists/*

# CUTLASS 4.3.0
RUN git clone --depth=1 --branch v4.3.0 https://github.com/NVIDIA/cutlass.git /opt/cutlass
ENV CUTLASS_PATH=/opt/cutlass

# Nsight Compute CLI (headless)
RUN wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub \
 | gpg --dearmor > /usr/share/keyrings/cuda-archive-keyring.gpg && \
 echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
 > /etc/apt/sources.list.d/cuda.list && \
 apt-get update && apt-get install -y nvidia-nsight-compute && \
 rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# Build H100 binary
RUN nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo -Xptxas -v \
    -I${CUTLASS_PATH}/include -I${CUTLASS_PATH}/tools/util/include \
    -o sparse_h100 src/sparse_bsr_gemm_h100.cu

CMD ["./sparse_h100"]

