# FlashCore RunPod Endpoint - CUDA 13.0 + CUTLASS 4.3.0
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy repository contents
COPY . .

# Install Python dependencies if requirements.txt exists
RUN if [ -f requirements.txt ]; then pip3 install --no-cache-dir -r requirements.txt; fi

# Install PyTorch with CUDA 13.0 support
RUN pip3 install --no-cache-dir \
    torch==2.4.1 \
    triton==3.0.0 \
    numpy

# Clone and setup CUTLASS 4.3.0 (main branch)
RUN git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass && \
    cd /opt/cutlass && \
    git checkout main

# Set CUTLASS environment variables
ENV CUTLASS_HOME=/opt/cutlass
ENV CPATH=${CUTLASS_HOME}/include:${CPATH}

# Verify CUDA installation
RUN nvcc --version && \
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Expose default port for HTTP endpoints
EXPOSE 8000

# Default command - can be overridden by RunPod
CMD ["bash", "-c", "nvcc --version && nvidia-smi && python3 main.py"]

