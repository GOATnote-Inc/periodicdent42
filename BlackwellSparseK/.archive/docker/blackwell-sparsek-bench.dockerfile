# ============================================================================
# BlackwellSparseK Benchmark Container
# ============================================================================
# Extends dev image with profiling and benchmarking tools
# Includes: Nsight Compute, Jupyter, visualization libraries
# ============================================================================

FROM blackwell-sparsek:dev

WORKDIR /workspace/benchmarks

# Install additional benchmark tools
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    tensorboard \
    plotly \
    scikit-learn

# Install Nsight Compute (CUDA 13.0 compatible)
# Note: Adjust version based on actual CUDA 13.0 release
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends nsight-compute || true && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.0-1_all.deb

# Add Nsight Compute to PATH if available
ENV PATH=/opt/nvidia/nsight-compute/2025.3.0:$PATH

# Create benchmark results directory
RUN mkdir -p /workspace/results

# Benchmark scripts
COPY benchmarks/ /workspace/benchmarks/

# Make benchmark scripts executable
RUN find /workspace/benchmarks -name "*.sh" -exec chmod +x {} \;

# Default: Run performance benchmark
CMD ["bash", "-c", "python perf.py --save-results && echo 'Benchmark complete! Results in /workspace/results'"]

