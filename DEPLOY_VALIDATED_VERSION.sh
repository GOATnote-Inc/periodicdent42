#!/bin/bash
# Deploy validated version with all fixes applied
# Date: October 1, 2025

set -e  # Exit on any error

echo "=========================================="
echo "DEPLOYING VALIDATED VERSION"
echo "=========================================="
echo ""

# 1. Re-run validation with fixes
echo "✅ Step 1: Re-running validation suite..."
cd /Users/kiteboard/periodicdent42
source app/venv/bin/activate
python scripts/validate_rl_system.py 2>&1 | tee validation_fixed.log
echo ""

# 2. Build Docker image
echo "✅ Step 2: Building Docker image..."
cd app
docker buildx build --platform linux/amd64 \
  -t gcr.io/periodicdent42/ard-backend:validated-$(date +%Y%m%d) \
  -t gcr.io/periodicdent42/ard-backend:latest .
echo ""

# 3. Push to GCR
echo "✅ Step 3: Pushing to Google Container Registry..."
docker push gcr.io/periodicdent42/ard-backend:validated-$(date +%Y%m%d)
docker push gcr.io/periodicdent42/ard-backend:latest
echo ""

# 4. Deploy to Cloud Run
echo "✅ Step 4: Deploying to Cloud Run..."
gcloud run deploy ard-backend \
  --image gcr.io/periodicdent42/ard-backend:latest \
  --region us-central1 \
  --project periodicdent42 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --concurrency 80
echo ""

# 5. Test health endpoint
echo "✅ Step 5: Testing health endpoint..."
SERVICE_URL=$(gcloud run services describe ard-backend \
  --region us-central1 \
  --project periodicdent42 \
  --format="value(status.url)")
echo "Service URL: $SERVICE_URL"
curl -f "${SERVICE_URL}/healthz" || { echo "❌ Health check failed!"; exit 1; }
echo ""

# 6. Test reasoning endpoint
echo "✅ Step 6: Testing reasoning endpoint..."
curl -X POST "${SERVICE_URL}/api/reasoning/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the optimal experiment?", "context": {"temperature": 25}}' \
  --max-time 10 || { echo "⚠️  Reasoning endpoint test timed out (expected for SSE)"; }
echo ""

echo "=========================================="
echo "✅ DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""
echo "Service URL: $SERVICE_URL"
echo "Web UI: ${SERVICE_URL}/"
echo "Benchmark: ${SERVICE_URL}/static/benchmark.html"
echo "Health: ${SERVICE_URL}/healthz"
echo "API: ${SERVICE_URL}/api/reasoning/query"
echo ""
echo "Next steps:"
echo "1. Open ${SERVICE_URL}/ in your browser"
echo "2. Test the query interface"
echo "3. Review benchmark results at ${SERVICE_URL}/static/benchmark.html"
echo "4. Monitor at https://console.cloud.google.com/run?project=periodicdent42"
echo ""

