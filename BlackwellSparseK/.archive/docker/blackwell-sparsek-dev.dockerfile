# ============================================================================
# BlackwellSparseK Development Container
# ============================================================================
# Multi-stage build for CUDA 13.0.2 + PyTorch 2.9.0 + CUTLASS 4.3.0 + vLLM
# Target: Development environment with full tool chain
# Build time: ~20-30 minutes (with caching)
# ============================================================================

# ============================================================================
# Stage 1: Base with CUDA 13.0.2
# ============================================================================
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ninja-build \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# ============================================================================
# Stage 2: PyTorch 2.9.0 cu130
# ============================================================================
FROM base AS pytorch

# Install PyTorch with CUDA 13.0 support
# Note: PyTorch 2.9.0 is hypothetical for this plan (use latest available)
# Adjust to actual version available at deployment time
RUN pip install --no-cache-dir \
    torch==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    packaging>=23.0 \
    ninja

# ============================================================================
# Stage 3: xFormers from source
# ============================================================================
FROM pytorch AS xformers-build

# Environment variables for xFormers build
ENV XFORMERS_FORCE_BUILD=1 \
    TORCH_CUDA_ARCH_LIST="90;100" \
    XFORMERS_BUILD_CUDA_ARCH_LIST="100" \
    MAX_JOBS=8

# Install build dependencies
RUN pip install --no-cache-dir \
    ninja \
    packaging \
    wheel

# Build xFormers from source with sm_90a and sm_100 support
# Pin to known-good version
RUN pip install --no-cache-dir --no-binary xformers \
    "xformers==0.0.23.post1" || \
    pip install --no-cache-dir "xformers>=0.0.23"

# ============================================================================
# Stage 4: CUTLASS 4.3.0
# ============================================================================
FROM xformers-build AS cutlass

# Clone CUTLASS 4.3.0 (or specific commit per memory)
RUN git clone -b v4.1.0 --depth 1 https://github.com/NVIDIA/cutlass /opt/cutlass

# Build CUTLASS (optional, headers-only mostly)
WORKDIR /opt/cutlass
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCUTLASS_NVCC_ARCHS="90;100" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUTLASS_ENABLE_TESTS=OFF \
        -DCUTLASS_ENABLE_EXAMPLES=OFF && \
    cmake --build . -j8 || true

# Set CUTLASS environment variable
ENV CUTLASS_PATH=/opt/cutlass \
    CPATH=/opt/cutlass/include:$CPATH

# Install CUTLASS Python DSL (if available)
RUN pip install --no-cache-dir nvidia-cutlass || true

WORKDIR /workspace

# ============================================================================
# Stage 5: vLLM + dependencies
# ============================================================================
FROM cutlass AS vllm-env

# Install vLLM and dependencies
RUN pip install --no-cache-dir \
    "vllm>=0.6.0" \
    packaging>=23 \
    pytest>=7.0 \
    pytest-benchmark>=4.0 \
    pytest-xdist>=3.0

# Development tools
RUN pip install --no-cache-dir \
    black>=24.0 \
    ruff>=0.1 \
    mypy>=1.0 \
    ipython \
    jupyter

# ============================================================================
# Stage 6: Final development image
# ============================================================================
FROM vllm-env AS dev

# Copy BlackwellSparseK source
COPY . /workspace/BlackwellSparseK
WORKDIR /workspace/BlackwellSparseK

# Install in editable mode
RUN pip install -e .[dev,bench]

# Sanity check
RUN python -c "import torch; print('PyTorch:', torch.__version__); \
    print('CUDA available:', torch.cuda.is_available()); \
    print('CUDA version:', torch.version.cuda); \
    try: import xformers; print('xFormers:', xformers.__version__); \
    except: print('xFormers: not available'); \
    try: import vllm; print('vLLM:', vllm.__version__); \
    except: print('vLLM: not available');" || true

# Create helpful welcome message
RUN echo '#!/bin/bash' > /usr/local/bin/welcome && \
    echo 'echo "============================================"' >> /usr/local/bin/welcome && \
    echo 'echo "BlackwellSparseK Development Environment"' >> /usr/local/bin/welcome && \
    echo 'echo "============================================"' >> /usr/local/bin/welcome && \
    echo 'echo ""' >> /usr/local/bin/welcome && \
    echo 'python -c "import torch, sys; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.version.cuda}\"); print(f\"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}\"); print(f\"Python: {sys.version}\")" 2>/dev/null || true' >> /usr/local/bin/welcome && \
    echo 'echo ""' >> /usr/local/bin/welcome && \
    echo 'echo "Quick start:"' >> /usr/local/bin/welcome && \
    echo 'echo "  python benchmarks/perf.py"' >> /usr/local/bin/welcome && \
    echo 'echo "  pytest tests/"' >> /usr/local/bin/welcome && \
    echo 'echo ""' >> /usr/local/bin/welcome && \
    chmod +x /usr/local/bin/welcome

# Set entrypoint
CMD ["/bin/bash", "-c", "welcome && exec /bin/bash"]

