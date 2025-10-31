#!/bin/bash
# ============================================================================
# Build All BlackwellSparseK Containers
# ============================================================================

set -e

IMAGE_NAME="blackwell-sparsek"
VERSION="0.1.0"

echo "🐳 Building BlackwellSparseK containers..."
echo ""

# Build dev image (multi-stage, takes ~20-30 min)
echo "======================================"
echo "Building dev image..."
echo "======================================"
docker build \
    -f docker/blackwell-sparsek-dev.dockerfile \
    -t ${IMAGE_NAME}:dev \
    -t ${IMAGE_NAME}:${VERSION}-dev \
    --target dev \
    .

echo ""
echo "✅ Dev image built successfully"
echo ""

# Build prod image
echo "======================================"
echo "Building prod image..."
echo "======================================"
docker build \
    -f docker/blackwell-sparsek-prod.dockerfile \
    -t ${IMAGE_NAME}:prod \
    -t ${IMAGE_NAME}:${VERSION} \
    -t ${IMAGE_NAME}:latest \
    .

echo ""
echo "✅ Prod image built successfully"
echo ""

# Build benchmark image
echo "======================================"
echo "Building benchmark image..."
echo "======================================"
docker build \
    -f docker/blackwell-sparsek-bench.dockerfile \
    -t ${IMAGE_NAME}:bench \
    -t ${IMAGE_NAME}:${VERSION}-bench \
    .

echo ""
echo "✅ Benchmark image built successfully"
echo ""

# Build CI image
echo "======================================"
echo "Building CI image..."
echo "======================================"
docker build \
    -f docker/blackwell-sparsek-ci.dockerfile \
    -t ${IMAGE_NAME}:ci \
    -t ${IMAGE_NAME}:${VERSION}-ci \
    .

echo ""
echo "✅ CI image built successfully"
echo ""

# Summary
echo "======================================"
echo "BUILD SUMMARY"
echo "======================================"
echo ""
echo "All containers built successfully!"
echo ""
echo "Images:"
docker images | grep ${IMAGE_NAME} | head -n 20
echo ""
echo "Usage:"
echo "  docker-compose up dev                    # Development"
echo "  docker-compose --profile production up   # vLLM server"
echo "  docker-compose --profile benchmark up    # Benchmarks"
echo "  docker-compose --profile test up         # CI tests"
echo ""

