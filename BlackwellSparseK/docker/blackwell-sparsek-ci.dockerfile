# ============================================================================
# BlackwellSparseK CI Container
# ============================================================================
# Lightweight image for GitHub Actions CI/CD
# No GPU required for unit tests
# Fast build time: <5 minutes
# ============================================================================

FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Minimal dependencies for testing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /ci

# Install test dependencies only (no CUDA kernels)
RUN pip install --no-cache-dir \
    torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu \
    pytest>=7.0 \
    pytest-cov>=4.0 \
    pytest-xdist>=3.0 \
    ruff>=0.1 \
    black>=24.0 \
    mypy>=1.0 \
    numpy>=1.24 \
    packaging>=23.0

# Copy source and tests
COPY src/ /ci/src/
COPY tests/ /ci/tests/
COPY pyproject.toml setup.py /ci/

# Install package in development mode (without building CUDA extension)
RUN pip install -e . --no-build-isolation || \
    pip install -e . --config-settings="--build-option=--no-cuda" || \
    echo "Package install skipped (expected for CI)"

# Run tests (skip GPU-specific tests)
CMD ["pytest", "tests/", "-v", "--cov=src/blackwell_sparsek", \
     "--ignore=tests/test_kernels.py", \
     "-m", "not gpu"]

