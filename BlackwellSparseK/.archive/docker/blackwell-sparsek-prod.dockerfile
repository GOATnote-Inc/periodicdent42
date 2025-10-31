# ============================================================================
# BlackwellSparseK Production Container
# ============================================================================
# Optimized runtime image for production deployment
# Base: CUDA 13.0.2 runtime (smaller than devel)
# Target: <3GB image size
# ============================================================================

FROM nvidia/cuda:13.0.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-13.0 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy pre-built wheels and dependencies from dev image
# Note: In practice, build dev image first, then copy artifacts
COPY --from=blackwell-sparsek:dev /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=blackwell-sparsek:dev /opt/cutlass /opt/cutlass

ENV CUTLASS_PATH=/opt/cutlass

# Copy application code
COPY src/ /app/src/
COPY benchmarks/ /app/benchmarks/
COPY pyproject.toml setup.py README.md LICENSE /app/

# Install package
RUN pip install --no-cache-dir .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import blackwell_sparsek; print('OK')" || exit 1

# Expose port for vLLM server
EXPOSE 8000

# Default: vLLM server with SPARSEK_XFORMERS backend
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "meta-llama/Llama-3.1-70B", \
     "--max-model-len", "4096", \
     "--attention-backend", "SPARSEK_XFORMERS"]

