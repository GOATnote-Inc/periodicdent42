#!/usr/bin/env bash
# Download BETE-NET model weights from upstream repository
# Copyright 2025 GOATnote Autonomous Research Lab Initiative

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$ROOT_DIR/third_party/bete_net/models"

echo "🔽 Downloading BETE-NET model weights..."

# Create models directory
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Download from BETE-NET GitHub releases or Zenodo
# Note: Replace with actual URL once available
BETE_NET_RELEASE_URL="https://github.com/henniggroup/BETE-NET/releases/download/v1.0/bete_weights.tar.gz"
ZENODO_URL="https://zenodo.org/record/XXXXX/files/bete_weights.tar.gz"

echo "📦 Fetching weights..."
if command -v wget &> /dev/null; then
    wget -O bete_weights.tar.gz "$BETE_NET_RELEASE_URL" || \
    wget -O bete_weights.tar.gz "$ZENODO_URL"
elif command -v curl &> /dev/null; then
    curl -L -o bete_weights.tar.gz "$BETE_NET_RELEASE_URL" || \
    curl -L -o bete_weights.tar.gz "$ZENODO_URL"
else
    echo "❌ Error: wget or curl required"
    exit 1
fi

echo "📂 Extracting weights..."
tar -xzf bete_weights.tar.gz

echo "🔐 Computing checksums..."
sha256sum model_*.pt > weights_checksums.txt

echo "✅ Weights downloaded successfully!"
echo ""
echo "📊 Model inventory:"
ls -lh model_*.pt
echo ""
echo "🔑 Checksums:"
cat weights_checksums.txt

# Cleanup
rm -f bete_weights.tar.gz

echo ""
echo "Next steps:"
echo "  1. Update inference.py with model loading code"
echo "  2. Run: pytest app/tests/test_bete_inference.py -v"
echo "  3. Deploy: gcloud run deploy ard-backend --memory 4Gi"

