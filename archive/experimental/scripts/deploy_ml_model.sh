#!/bin/bash
# scripts/deploy_ml_model.sh
# Deploy trained ML model to Google Cloud Storage
# Phase 3 Week 7 Day 7: ML Test Selection Deployment

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
MODEL_FILE="${1:-test_selector.pkl}"
METADATA_FILE="${2:-test_selector.json}"
BUCKET="gs://periodicdent42-ml-models"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║  🚀 ML Model Deployment - Cloud Storage                                   ║"
echo "║     Deploying test selection model to production                          ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check model files exist
echo -e "${BLUE}1. Checking model files...${NC}"
if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}❌ Error: Model file not found: $MODEL_FILE${NC}"
    echo -e "${YELLOW}   Train model first: python scripts/train_test_selector.py --train --evaluate${NC}"
    exit 1
fi

if [ ! -f "$METADATA_FILE" ]; then
    echo -e "${RED}❌ Error: Metadata file not found: $METADATA_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found model files:${NC}"
echo -e "   - $MODEL_FILE ($(du -h $MODEL_FILE | cut -f1))"
echo -e "   - $METADATA_FILE ($(du -h $METADATA_FILE | cut -f1))"
echo ""

# Check bucket exists
echo -e "${BLUE}2. Checking Cloud Storage bucket...${NC}"
if gsutil ls "$BUCKET" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Bucket exists: $BUCKET${NC}"
else
    echo -e "${YELLOW}⚠️  Creating bucket...${NC}"
    gcloud storage buckets create "$BUCKET" --location=us-central1 --public-access-prevention
fi
echo ""

# Upload model
echo -e "${BLUE}3. Uploading model to Cloud Storage...${NC}"
gsutil cp "$MODEL_FILE" "$BUCKET/" 2>&1
gsutil cp "$METADATA_FILE" "$BUCKET/" 2>&1
echo -e "${GREEN}✅ Upload complete${NC}"
echo ""

# Verify upload
echo -e "${BLUE}4. Verifying deployment...${NC}"
MODEL_URL="$BUCKET/$(basename $MODEL_FILE)"
METADATA_URL="$BUCKET/$(basename $METADATA_FILE)"

if gsutil ls "$MODEL_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Model accessible: $MODEL_URL${NC}"
else
    echo -e "${RED}❌ Model upload verification failed${NC}"
    exit 1
fi

if gsutil ls "$METADATA_URL" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Metadata accessible: $METADATA_URL${NC}"
else
    echo -e "${RED}❌ Metadata upload verification failed${NC}"
    exit 1
fi
echo ""

# Show metadata
echo -e "${BLUE}5. Model Information:${NC}"
gsutil cat "$METADATA_URL" | python -m json.tool
echo ""

# Success message
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║  ✅ ML MODEL DEPLOYMENT COMPLETE!                                         ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Deployment Summary:${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Model: $MODEL_URL"
echo -e "  Metadata: $METADATA_URL"
echo -e "  Bucket: $BUCKET"
echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo -e "  1. Update CI to download model:"
echo -e "     ${YELLOW}# In .github/workflows/ci.yml, uncomment:${NC}"
echo -e "     ${YELLOW}gsutil cp $MODEL_URL test_selector.pkl${NC}"
echo -e "     ${YELLOW}gsutil cp $METADATA_URL test_selector.json${NC}"
echo ""
echo -e "  2. Enable ML prediction:"
echo -e "     ${YELLOW}echo \"skip_ml=false\" >> \$GITHUB_OUTPUT${NC}"
echo ""
echo -e "  3. Monitor CI time reduction:"
echo -e "     ${YELLOW}Target: 70% reduction${NC}"
echo ""
echo "© 2025 GOATnote Autonomous Research Lab Initiative"
echo "ML Model Deployment Complete: $(date)"
