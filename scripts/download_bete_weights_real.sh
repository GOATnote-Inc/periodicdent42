#!/usr/bin/env bash
# Download REAL BETE-NET model weights for publication-quality research
# Copyright 2025 GOATnote Autonomous Research Lab Initiative

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$ROOT_DIR/third_party/bete_net/models"

echo "════════════════════════════════════════════════════════════════"
echo "  BETE-NET REAL WEIGHTS DOWNLOAD"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "⚠️  IMPORTANT: This downloads 5.48 GB of data"
echo "   Estimated time: 30-60 minutes"
echo "   Network cost: ~\$0.50"
echo ""
echo "📊 What you're downloading:"
echo "   • 5 bootstrap ensemble models (PyTorch .pt files)"
echo "   • Model configuration and architecture specs"
echo "   • Training dataset metadata"
echo "   • SHA-256 checksums for verification"
echo ""
echo "📋 Publication Requirements:"
echo "   Real weights are REQUIRED for peer-reviewed publications."
echo "   Mock models are NOT acceptable for scientific claims."
echo ""
read -p "Continue with download? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "❌ Download cancelled."
    echo ""
    echo "To continue using mock models for development:"
    echo "  • Current API will work with mock data"
    echo "  • Tests will run but show ⚠️ MOCK warnings"
    echo "  • NOT suitable for publications"
    exit 0
fi

echo "🔽 Starting download..."
echo ""

# Create models directory
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Method 1: Try HyperAI torrent (primary source)
echo "📦 Attempting Method 1: HyperAI Dataset..."
HYPERAI_TORRENT="https://hyperai.com/datasets/bete-net/bete_weights.torrent"

if command -v transmission-cli &> /dev/null; then
    echo "✅ transmission-cli found"
    
    curl -L -o bete_weights.torrent "$HYPERAI_TORRENT" 2>/dev/null || {
        echo "⚠️  HyperAI torrent not available"
    }
    
    if [ -f "bete_weights.torrent" ]; then
        echo "📥 Downloading via BitTorrent (5.48 GB)..."
        echo "   This may take 30-60 minutes..."
        transmission-cli bete_weights.torrent --download-dir . || {
            echo "❌ Torrent download failed"
            rm -f bete_weights.torrent
        }
    fi
else
    echo "⚠️  transmission-cli not found"
    echo "   Install with: brew install transmission-cli"
fi

# Method 2: Try GitHub Releases (if available)
if [ ! -f "model_ensemble_0.pt" ]; then
    echo ""
    echo "📦 Attempting Method 2: GitHub Releases..."
    
    GITHUB_REPO="henniggroup/BETE-NET"
    RELEASE_TAG="v1.0"
    RELEASE_URL="https://github.com/$GITHUB_REPO/releases/download/$RELEASE_TAG/bete_weights.tar.gz"
    
    if command -v gh &> /dev/null; then
        echo "✅ GitHub CLI found"
        gh release download "$RELEASE_TAG" \
            --repo "$GITHUB_REPO" \
            --pattern "bete_weights.tar.gz" 2>/dev/null || {
            echo "⚠️  GitHub release not found"
        }
    elif command -v curl &> /dev/null; then
        echo "✅ curl found, attempting direct download..."
        curl -L -o bete_weights.tar.gz "$RELEASE_URL" 2>/dev/null || {
            echo "⚠️  GitHub direct download not available"
        }
    fi
    
    if [ -f "bete_weights.tar.gz" ]; then
        echo "📂 Extracting weights..."
        tar -xzf bete_weights.tar.gz
        rm -f bete_weights.tar.gz
    fi
fi

# Method 3: Manual download instructions
if [ ! -f "model_ensemble_0.pt" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ⚠️  AUTOMATIC DOWNLOAD FAILED"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Please download manually using one of these methods:"
    echo ""
    echo "METHOD A: HyperAI Website"
    echo "  1. Visit: https://hyperai.com/datasets/bete-net"
    echo "  2. Click 'Dataset Download'"
    echo "  3. Use a torrent client to download (5.48 GB)"
    echo "  4. Extract to: $MODELS_DIR"
    echo ""
    echo "METHOD B: Contact Authors"
    echo "  Email: hennig_group@ufl.edu"
    echo "  Subject: Request for BETE-NET Model Weights"
    echo "  Reference: Nature paper on superconductor discovery"
    echo ""
    echo "METHOD C: Zenodo Repository (if available)"
    echo "  1. Search Zenodo for 'BETE-NET superconductor'"
    echo "  2. Download model weights package"
    echo "  3. Extract to: $MODELS_DIR"
    echo ""
    echo "After manual download, verify with:"
    echo "  bash $SCRIPT_DIR/verify_bete_weights.sh"
    echo ""
    exit 1
fi

# Verification
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🔐 VERIFYING DOWNLOAD"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "📊 Files downloaded:"
ls -lh model_*.pt 2>/dev/null || {
    echo "❌ No model files found!"
    exit 1
}

echo ""
echo "🔑 Computing checksums..."
sha256sum model_*.pt > computed_checksums.txt

# If checksums file exists, verify
if [ -f "checksums.txt" ]; then
    echo "✅ Verifying against official checksums..."
    if sha256sum -c checksums.txt 2>/dev/null; then
        echo "✅ ALL CHECKSUMS VERIFIED!"
    else
        echo "⚠️  Checksum verification failed"
        echo "   This may indicate corrupted download"
        echo "   Consider re-downloading"
    fi
else
    echo "⚠️  Official checksums not found"
    echo "   Computed checksums saved to: computed_checksums.txt"
fi

# Test model loading
echo ""
echo "🧪 Testing model loading..."
python3 << 'EOF'
import torch
import sys

try:
    model = torch.load("model_ensemble_0.pt", map_location="cpu")
    print("✅ Model loaded successfully!")
    
    if "state_dict" in model:
        print(f"   Architecture: {model.get('config', {}).get('architecture', 'unknown')}")
        print(f"   Parameters: {sum(p.numel() for p in model['state_dict'].values()):,}")
    else:
        print("   ⚠️ Unexpected model format")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ DOWNLOAD COMPLETE"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "📊 Model Inventory:"
    ls -lh model_*.pt
    echo ""
    echo "🎯 Next Steps:"
    echo "  1. Run validation tests:"
    echo "     pytest app/tests/test_bete_golden.py -v"
    echo ""
    echo "  2. Expected results (with real weights):"
    echo "     • Nb: T_c ≈ 9.2 K"
    echo "     • MgB₂: T_c ≈ 39 K"
    echo "     • Al: T_c ≈ 1.2 K"
    echo ""
    echo "  3. If tests pass, rebuild Docker image:"
    echo "     docker buildx build --platform linux/amd64 \\"
    echo "       -t ard-backend:with-real-weights -f app/Dockerfile ."
    echo ""
    echo "  4. Deploy to production:"
    echo "     gcloud run deploy ard-backend-v2 --memory=2Gi"
    echo ""
    echo "📝 Documentation:"
    echo "   See: third_party/bete_net/WEIGHTS_INFO.md"
    echo ""
    echo "🎓 For Publications:"
    echo "   ✅ Real weights downloaded"
    echo "   ⏳ Validation required (run tests above)"
    echo "   ⏳ Cite original Nature paper"
    echo ""
else
    echo ""
    echo "❌ Download completed but model validation failed"
    echo "   Please check logs above for errors"
    exit 1
fi

