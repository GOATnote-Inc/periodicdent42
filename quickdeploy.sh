#!/bin/bash
# Quick deployment script for Autonomous R&D Intelligence Layer
# Run from project root: bash quickdeploy.sh

set -e  # Exit on error

PROJECT_ID=${PROJECT_ID:-periodicdent42}
REGION=${REGION:-us-central1}

echo "🚀 Deploying Autonomous R&D Intelligence Layer"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Step 1: Set project
echo "📋 Step 1/6: Setting GCP project..."
gcloud config set project $PROJECT_ID

# Step 2: Enable APIs
echo "🔧 Step 2/6: Enabling APIs..."
bash infra/scripts/enable_apis.sh

# Step 3: Setup IAM
echo "🔒 Step 3/6: Setting up IAM..."
bash infra/scripts/setup_iam.sh

# Step 4: Run tests
echo "✅ Step 4/6: Running tests..."
cd app
if command -v python3 &> /dev/null; then
    python3 -m pip install -q -r requirements.txt
    python3 -m pytest -q 2>&1 | tail -5
else
    echo "⚠️  Python 3 not found, skipping tests"
fi
cd ..

# Step 5: Build image
echo "🏗️  Step 5/6: Building Docker image..."
cd app
gcloud builds submit --tag gcr.io/$PROJECT_ID/ard-backend --quiet
cd ..

# Step 6: Deploy to Cloud Run
echo "🚀 Step 6/6: Deploying to Cloud Run..."
export PROJECT_ID=$PROJECT_ID
export REGION=$REGION
bash infra/scripts/deploy_cloudrun.sh

# Get service URL
SERVICE_URL=$(gcloud run services describe ard-backend \
  --region $REGION \
  --format 'value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test health check:"
echo "  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $SERVICE_URL/healthz"
echo ""
echo "View logs:"
echo "  gcloud run services logs tail ard-backend --region=$REGION"
echo ""
echo "🎉 Your Autonomous R&D Intelligence Layer is live!"

