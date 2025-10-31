#!/bin/bash
# ============================================================================
# Push BlackwellSparseK Containers to GitHub Container Registry
# ============================================================================

set -e

# Configuration
REGISTRY="ghcr.io/yourusername"
IMAGE="blackwell-sparsek"
VERSION="0.1.0"

echo "======================================"
echo "Pushing to GitHub Container Registry"
echo "======================================"
echo "Registry: ${REGISTRY}"
echo "Image: ${IMAGE}"
echo "Version: ${VERSION}"
echo ""

# Check if logged in
echo "🔐 Checking registry login..."
if ! docker info | grep -q "Username:"; then
    echo "❌ Not logged in to Docker registry"
    echo ""
    echo "Login with:"
    echo "  echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
    exit 1
fi

# Tag images
echo ""
echo "🏷️  Tagging images..."

docker tag ${IMAGE}:latest ${REGISTRY}/${IMAGE}:latest
docker tag ${IMAGE}:latest ${REGISTRY}/${IMAGE}:${VERSION}
docker tag ${IMAGE}:dev ${REGISTRY}/${IMAGE}:dev
docker tag ${IMAGE}:dev ${REGISTRY}/${IMAGE}:${VERSION}-dev
docker tag ${IMAGE}:bench ${REGISTRY}/${IMAGE}:bench
docker tag ${IMAGE}:bench ${REGISTRY}/${IMAGE}:${VERSION}-bench
docker tag ${IMAGE}:ci ${REGISTRY}/${IMAGE}:ci
docker tag ${IMAGE}:ci ${REGISTRY}/${IMAGE}:${VERSION}-ci

echo "✅ Images tagged"

# Push images
echo ""
echo "📤 Pushing images to registry..."
echo ""

docker push ${REGISTRY}/${IMAGE}:latest
docker push ${REGISTRY}/${IMAGE}:${VERSION}
docker push ${REGISTRY}/${IMAGE}:dev
docker push ${REGISTRY}/${IMAGE}:${VERSION}-dev
docker push ${REGISTRY}/${IMAGE}:bench
docker push ${REGISTRY}/${IMAGE}:${VERSION}-bench
docker push ${REGISTRY}/${IMAGE}:ci
docker push ${REGISTRY}/${IMAGE}:${VERSION}-ci

echo ""
echo "======================================"
echo "✅ Push Complete!"
echo "======================================"
echo ""
echo "Images available at:"
echo "  ${REGISTRY}/${IMAGE}:latest"
echo "  ${REGISTRY}/${IMAGE}:${VERSION}"
echo "  ${REGISTRY}/${IMAGE}:dev"
echo "  ${REGISTRY}/${IMAGE}:bench"
echo "  ${REGISTRY}/${IMAGE}:ci"
echo ""

